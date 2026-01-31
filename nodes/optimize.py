import json
from state import GraphState
from utils import llm_client
from datetime import datetime

def optimization_node(state: GraphState):
    input_json = state

    optimization_input = {
        "meta": state["meta"],
        "fixed_events": state["fixed_events"],
        "todo_items": [item for item in state["todo_items"] if item.get("status") == "confirmed"],
        "distance_matrix": state["distance_matrix"]
    }
    print(f"DEBUG [Optimize]: 전달받은 고정 일정 개수 = {len(state['fixed_events'])}")

    system_prompt = """너는 'Route_Optimizer_Function'이라는 이름의 함수다. 
아래의 실행 순서(Step)를 엄격히 준수하여 논리적 근거를 생성하고, 지정된 스키마에 맞춰 결과를 출력하라.

[실행 알고리즘]
Step 1. 고정 일정 확인: 'fixed_events' 항목들의 시작/종료 시간을 타임라인의 절대적 뼈대로 세운다.
Step 2. 지역 군집화(Clustering): 'distance_matrix' 수치가 10 이하인 지점들을 그룹화하고 중심 fixed_events 지점을 선정한다.
Step 3. 대안 비교 및 틈새 삽입: 
    - todo_items 를 fixed_events 전/후에 배치했을 때의 거리 수치 변화를 시뮬레이션하여 최소값 선택.
    - 자취방(출발/도착) 근처 todo_items는 '출발 직후' 또는 '귀가 직전' 배치 우선순위 적용.
Step 4. 전체 수치 검증: 경로상 모든 구간의 distance 수치를 합산하여 최종 total_distance를 도출한다.

[출력 JSON 스키마 고정]
{
  "total_distance": "숫자 (전체 거리 합계)",
  "schedule": [
    "0. home_start (장소명) - 출발 [00:00]",
    "1. ID (장소명) - 활동명 ",
    "2. ... 순차적으로 리스트 구성",
    "N. home_end (장소명) - 도착"
  ],
  "reasoning": [
    "[Step 1. 고정 일정 확인] - 분석 내용 (예: 가용 시간 계산 등)",
    "[Step 2. 지역 군집화] - 분석 내용 (예: 그룹화된 항목과 거리 수치)",
    "[Step 3. 대안 비교 및 삽입] - 분석 내용 (예: A 대신 B를 선택하여 거리 5 절감 등 구체적 근거)",
    "[Step 4. 수치 검증] - 합산식 (예: 10+5+15...=30)"
  ]
}

[주의사항]
- schedule은 반드시 문자열 리스트로 작성하여 토글 없이 한눈에 보이게 할 것.
- reasoning은 생략 없이 아주 상세하게 로그 형태로 남길 것.
- 모든 텍스트는 한국어를 사용하며 JSON 형식 외의 불필요한 말은 하지 말 것."""

    user_input = f"데이터: {json.dumps(optimization_input, ensure_ascii=False)}"

    response = llm_client.chat.completions.create(
        model="solar-pro",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)

    # [수정] 파일명이 겹치지 않게 시간 정보를 추가하여 output 폴더에 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"output/result_{timestamp}.json"
    
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"로그: 결과가 {file_name}에 저장되었습니다.")

    return {"optimized_result": result}
