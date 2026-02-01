from state import GraphState
from utils import calculate_distance

def distance_matrix_node(state: GraphState):
    print("\n--- [NODE 4] 거리 행렬(Distance Matrix) 생성 시작 ---")
    nodes = []
    meta = state["meta"]
    
    # (1) Start Point
    nodes.append({
        "id": "start",
        "name": meta["start_point"]["name"],
        "coordinates": meta["start_point"]["coordinates"]
    })

    # (2) Fixed Events 처리 (에러 지점 수정)
    # enumerate를 사용하여 i(인덱스)와 item(데이터)을 분리해서 받습니다.
    for i, item in enumerate(state.get("fixed_events", [])):
        nodes.append({
            "id": item.get("id") or f"fixed_{i}", 
            "name": item.get("title") or "고정 일정",
            "coordinates": item.get("coordinates")
        })

    # (3) Confirmed Todo Items 처리
    for item in state.get("todo_items", []):
        if item.get("status") == "confirmed" and item.get("final_choice"):
            target_id = item["final_choice"]
            selected_candidate = next((c for c in item["candidates"] if c["id"] == target_id), None)
            
            if selected_candidate:
                nodes.append({
                    "id": item["id"],
                    "name": selected_candidate["name"],
                    "coordinates": selected_candidate.get("coordinates")
                })

    # (4) End Point
    nodes.append({
        "id": "end",
        "name": meta["end_point"]["name"],
        "coordinates": meta["end_point"]["coordinates"]
    })

    # 2. Matrix 생성 (All-to-All)
    distance_matrix = {}
    print(f"> 총 {len(nodes)}개 지점 간 거리 계산 중...")
    
    for origin in nodes:
        for dest in nodes:
            if origin["id"] == dest["id"]:
                continue

            # 유틸리티 함수 calculate_distance 활용
            dist = calculate_distance(origin["coordinates"], dest["coordinates"])
            
            # 키 생성 (예: 'start->fixed_1')
            key = f"{origin['id']}->{dest['id']}"
            distance_matrix[key] = {
                "origin_name": origin["name"],
                "dest_name": dest["name"],
                "distance_meter": dist
            }

    print("--- [NODE 4] 완료: 거리 행렬 생성 완료 ---")
    
    # 결과만 반환하여 state에 저장
    return {
        "distance_matrix": distance_matrix,
    }