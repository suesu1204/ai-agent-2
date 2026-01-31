import json
from typing import Any, Dict, List
from state import GraphState
from utils import llm_client, get_coordinates_kakao

select_candidate_tool = {
    "type": "function",
    "function": {
        "name": "select_final_candidate",
        "description": "todo item의 후보 장소 중 하나를 선택",
        "parameters": {
            "type": "object",
            "properties": {
                "todo_id": {
                    "type": "string",
                    "description": "todo item id"
                },
                "candidate_id": {
                    "type": "string",
                    "description": "선택된 후보 장소의 id"
                },
                "reason": {
                    "type": "string",
                    "description": "선택 이유 (간단히)"
                }
            },
            "required": ["todo_id", "candidate_id"]
        }
    }
}

def select_candidate_with_llm(
    client,
    todo: dict,
    model: str = "solar-pro2"
):
    system_prompt = """
너는 사용자의 할 일을 가장 잘 수행할 장소를
이미 주어진 후보 목록 중에서 고르는 의사결정 전문가다.

규칙:
- 반드시 후보 목록에 있는 장소만 선택
- 새로운 장소를 만들어내지 말 것
- 장소 id(candidate_id)만 선택
- 반드시 function call로 응답
"""

    # 후보 목록을 LLM이 읽기 좋은 형태로 정리
    candidates_text = "\n".join([
        f"- id: {c['id']}, 이름: {c['name']}, 주소: {c['address']}"
        for c in todo["candidates"]
    ])

    user_prompt = f"""
[todo 정보]
- id: {todo['id']}
- 제목: {todo['title']}
- 소요 시간: {todo['duration']}분
- 중심 위치 힌트: {todo['center_place']}

[후보 장소 목록]
{candidates_text}

위 후보 중 가장 적합한 하나를 선택하라.
"""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        tools=[select_candidate_tool],
        tool_choice={
            "type": "function",
            "function": {"name": "select_final_candidate"}
        }
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)

    return args

def attach_final_choice_with_llm(client, todo_items):
    for todo in todo_items:
        if not todo.get("candidates"):
            continue

        result = select_candidate_with_llm(client, todo)

        todo["final_choice"] = result["candidate_id"]
        todo["selection_reason"] = result.get("reason", "")

    return todo_items

def selection_node(state: GraphState):
    print("\n--- [NODE 3] 장소 최종 선택 및 좌표 확정 시작 ---")
    
    todo_items = state["todo_items"]
    # 💡 고정 일정 유실 방지를 위해 미리 확보
    fixed_events = state.get("fixed_events", [])
    
    updated_todos = attach_final_choice_with_llm(llm_client, todo_items)
    new_selection_history = []
    
    for todo in updated_todos:
        final_id = todo.get("final_choice")
        if not final_id: continue
            
        selected_cand = next((c for c in todo["candidates"] if c["id"] == final_id), None)
        
        if selected_cand:
            # 💡 주소 데이터 정제 (리스트/문자열 모두 대응)
            raw_addr = selected_cand.get("address", "")
            if isinstance(raw_addr, list):
                valid_addr = [a for a in raw_addr if a and len(a.strip()) > 0]
                addr = valid_addr[0] if valid_addr else ""
            else:
                addr = raw_addr if raw_addr else ""

            # 좌표 변환 로직
            if not addr.strip():
                print(f" '{selected_cand['name']}': 주소 없음")
                selected_cand["coordinates"] = {"x": "0.0", "y": "0.0"}
            else:
                # 좌표가 없거나 초기값인 경우에만 갱신
                if not selected_cand.get("coordinates") or str(selected_cand["coordinates"].get("x")) == "0.0":
                    print(f" '{selected_cand['name']}' 좌표 변환 중: {addr}")
                    selected_cand["coordinates"] = get_coordinates_kakao(addr)
            
            todo["status"] = "confirmed"
            new_selection_history.append({
                "todo_id": todo["id"],
                "selected_place": selected_cand["name"]
            })

    print(f"--- [NODE 3] 완료: {len(new_selection_history)}개 장소 확정 ---")
    
    return {
        "todo_items": updated_todos,
        "selection_history": new_selection_history,
        "meta": state["meta"],
        "fixed_events": fixed_events # 안전하게 원본 데이터 유지
    }