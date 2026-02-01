import json
import asyncio
import nest_asyncio
from tqdm import tqdm
from state import GraphState
from utils import llm_client, attach_candidates_with_crawling_async
from typing import Dict, List, Any
from openai import OpenAI

#1. todo 기반 검색어 생성 과정 
generate_search_words_tool = {
    "type": "function",
    "function": {
        "name": "generate_search_words",
        "description": "todo item을 네이버 지도 검색에 적합한 한국어 검색어로 변환",
        "parameters": {
            "type": "object",
            "properties": {
                "todo_id": {
                    "type": "string",
                    "description": "todo item의 id"
                },
                "search_words": {
                    "type": "string",
                    "description": "네이버 지도에서 바로 사용할 검색어"
                }
            },
            "required": ["todo_id", "search_words"]
        }
    }
}

def generate_search_words_with_llm(
    client: OpenAI,
    meta: Dict[str, Any],
    todo_items: List[Dict[str, Any]],
    model: str = "solar-pro2"
) -> List[Dict[str, Any]]:
    """
    각 todo item에 대해 search_words를 생성하고
    todo_items를 업데이트하여 반환
    """

    updated_items = []

    for todo in tqdm(todo_items, desc="Generating search words"):
        system_prompt = system_prompt = """
너는 대한민국 지도 검색 키워드 최적화 전문가다.
사용자의 할 일을 분석하여 네이버 지도에서 '가장 많은 장소 결과'를 얻을 수 있는 보편적인 키워드를 생성하라.

[키워드 생성 규칙]
1. 형식: '[지역명] [핵심장소]' (예: '서대문구 서점', '한남동 카페')
2. 지역명 설정: 
   - '중심 위치'가 특정 장소이면 해당 장소가 속한 '구' 또는 '동' 단위로 확장하라.
   - 너무 좁은 지역(예: 이화여대길)보다는 '이대역' 또는 '신촌'처럼 유동인구가 많은 명칭을 사용하라.
3. 핵심장소 설정: 
   - 할 일의 목적에 맞는 가장 일반적인 명사형 단어를 선택하라. (예: '스터디룸', '카페', '서점')
4. 금지어: '근처', '추천', '하기 좋은', '맛있는' 등 검색 품질을 떨어뜨리는 형용사는 절대 금지한다.
"""
        user_prompt = f"""
        [사용자 기본 위치 정보]
        - 집: {meta.get("user_house_address")}
        - 직장/학교: {meta.get("user_workplace_address")}

        [할 일 정보]
        - 할 일 제목: {todo.get("task") or todo.get("title")} 
        - 사용자가 지정한 중심 위치: {todo["center_place"]}

        위 정보를 분석하여 네이버 지도 검색 결과가 가장 잘 나올 키워드 1개를 생성하라.
        """
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tools=[generate_search_words_tool],
            tool_choice={"type": "function", "function": {"name": "generate_search_words"}},
        )

        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)

        # todo item 업데이트
        todo["search_words"] = args["search_words"]
        todo["status"] = "need_recommendation"

        updated_items.append(todo)

    return updated_items


#2번노드 (candidate) 
def candidate_node(state: GraphState):
    """
    할 일(Todo) 목록에 대해 검색어를 생성하고 
    네이버 지도를 크롤링하여 후보 장소(candidates)를 수집하는 노드
    """
    print("\n--- [NODE 2] 후보지 탐색 및 크롤링 프로세스 시작 ---")
    
    # 1. State 데이터 추출
    meta = state["meta"]
    todo_items = state["todo_items"]
    
    # --- 단계 1: LLM 기반 검색어 생성 ---
    print(f"> {len(todo_items)}개의 할 일에 대한 검색어 생성 중...")
    
    # 기존에 정의하신 generate_search_words_with_llm 함수 호출
    # client는 전역 변수로 설정되어 있거나 이 노드 안에서 선언되어야 합니다.
    updated_todos = generate_search_words_with_llm(
        client=llm_client,
        meta=meta,
        todo_items=todo_items
    )
    
    # --- 단계 2: Playwright 기반 크롤링 ---
    print(f"> 네이버 지도 크롤링 및 장소 후보 수집 시작...")
    
    # 코랩/주피터 환경의 이벤트 루프 대응
    import nest_asyncio
    nest_asyncio.apply()
    
    # 기존에 정의하신 attach_candidates_with_crawling_async 함수 호출
    # 비동기 함수이므로 asyncio.run으로 실행합니다.
    final_updated_todos = asyncio.run(attach_candidates_with_crawling_async(updated_todos))
    
    print("--- [NODE 2] 후보지 탐색 완료 ---")
    
    # 업데이트된 todo_items를 반환하여 전역 state를 갱신합니다.
    return {
        "todo_items": final_updated_todos,
    }