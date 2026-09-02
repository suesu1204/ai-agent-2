from typing import Dict, Any, List
from state import GraphState
from utils import get_coordinates_kakao


def _require_coords(label: str, address: str):
    """좌표 변환 실패 시 잘못된 거리 행렬로 이어지므로 즉시 중단"""
    coords = get_coordinates_kakao(address)
    if coords.get("x") == "0.0":
        raise ValueError(f"{label} 좌표 변환 실패: '{address}'")
    return coords


# Node 1: 데이터 전처리 및 초기화 노드
def ingest_data_node(state: GraphState):
    print("\n--- [NODE 1] 데이터 처리를 시작합니다 ---")

    # (1) Meta Data 처리 및 좌표 변환
    start_coords = _require_coords("출발지", state["start_point"]["address"])
    end_coords = _require_coords("도착지", state["end_point"]["address"])
    print(f"DEBUG: Start Coords = {start_coords}, End Coords = {end_coords}")

    meta = {
        "user_id": state.get("user_id"),
        "target_date": state.get("target_date"),
        "user_house_address": state.get("user_house_address"),
        "user_workplace_address": state.get("user_workplace_address"),
        "start_point": {
            **state["start_point"],
            "coordinates": start_coords
        },
        "end_point": {
            **state["end_point"],
            "coordinates": end_coords
        }
    }

    # (2) Fixed Schedules 처리
    raw_fixed = state.get("fixed_events") or []
    
    print(f"DEBUG [Ingest]: 원본 데이터에서 찾은 일정 개수 = {len(raw_fixed)}")

    fixed_events = []
    for idx, item in enumerate(raw_fixed, 1):
        print(f"고정 일정 좌표 변환 중: {item['location']}")
        coords = _require_coords(f"고정 일정[{item['title']}]", item["location"])
        print(f"DEBUG: 고정일정[{item['title']}] 좌표 = {coords}")
        
        processed_item = {
            "id": f"fixed_{idx}",
            "type": "fixed",
            "title": item["title"],
            "location": item["location"],
            "coordinates": coords,
            "start_time": item["start_time"],
            "end_time": item["end_time"],
            "category": item["category"]
        }
        fixed_events.append(processed_item)

    # (3) Todo Items 처리
    todo_items = []
    for idx, item in enumerate(state.get("todo_list_raw", []), 1):
        processed_item = {
            "id": f"todo_{idx}",
            "type": "todo",
            "title": item["task"],
            "duration": item["user_duration"],
            "center_place": item.get("center_place", ""),
            "search_words": item.get("search_words", []),
            "status": "need_recommendation",
            "candidates": [],
            "final_choice": None
        }
        todo_items.append(processed_item)

    print("--- [NODE 1] 데이터 처리 완료 ---")
    print(f"DEBUG [Ingest]: 최종 생성된 고정 일정 개수 = {len(fixed_events)}")
    
    return {
        "meta": meta,
        "fixed_events": fixed_events,
        "todo_items": todo_items,
    }