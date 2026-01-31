from typing import Dict, Any, List
from state import GraphState
from utils import get_coordinates_kakao


# Node 1: 데이터 전처리 및 초기화 노드
def ingest_data_node(state: GraphState):
    # 💡 1. raw_input 정의
    raw_input = state 
    
    print("\n--- [NODE 1] 데이터 처리를 시작합니다 ---")

    # (1) Meta Data 처리 및 좌표 변환
    start_coords = get_coordinates_kakao(raw_input["start_point"]["address"])
    end_coords = get_coordinates_kakao(raw_input["end_point"]["address"])
    print(f"DEBUG: Start Coords = {start_coords}, End Coords = {end_coords}")

    meta = {
        "user_id": raw_input.get("user_id"),
        "target_date": raw_input.get("target_date"),
        "user_house_address": raw_input.get("user_house_address"),
        "user_workplace_address": raw_input.get("user_workplace_address"),
        "start_point": {
            **raw_input["start_point"],
            "coordinates": start_coords
        },
        "end_point": {
            **raw_input["end_point"],
            "coordinates": end_coords
        }
    }

    # (2) Fixed Schedules 처리
    # 수정: STATE에 있는 'fixed_events' 키를 직접 참조하거나 
    # main.py에서 보낸 'fixed_schedules'가 있다면 그것도 참조하도록 보강
    raw_fixed = raw_input.get("fixed_events") or raw_input.get("fixed_schedules") or []
    
    print(f"DEBUG [Ingest]: 원본 데이터에서 찾은 일정 개수 = {len(raw_fixed)}")

    fixed_events = []
    for idx, item in enumerate(raw_fixed, 1):
        print(f"고정 일정 좌표 변환 중: {item['location']}")
        coords = get_coordinates_kakao(item["location"])
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
    # 💡 수정: todo_list_raw 키 참조
    for idx, item in enumerate(raw_input.get("todo_list_raw", []), 1):
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
        "start_point": meta["start_point"],
        "end_point": meta["end_point"]
    }