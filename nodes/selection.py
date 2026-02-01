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
    updated_todos = attach_final_choice_with_llm(llm_client, todo_items)
    
    for todo in updated_todos:
        final_id = todo.get("final_choice")
        if not final_id: continue
            
        selected_cand = next((c for c in todo["candidates"] if c["id"] == final_id), None)
        
        if selected_cand:
            # 이미 candidate 노드에서 검증된 좌표가 넘어왔으므로 
            # 주소 유무와 상관없이 좌표를 그대로 유지
            print(f"   ✅ 확정: {selected_cand['name']} (좌표: {selected_cand['coordinates']['x']}, {selected_cand['coordinates']['y']})")
            todo["status"] = "confirmed"

    return {
        "todo_items": updated_todos,
        "selection_history": [
            {"todo_id": t["id"], "selected_place": next((c["name"] for c in t["candidates"] if c["id"] == t["final_choice"]), "N/A")} 
            for t in updated_todos if t.get("final_choice")
        ],
    }