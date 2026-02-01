from typing import Annotated, TypedDict, List, Dict, Any
import operator

class GraphState(TypedDict):
    # 초기 입력 및 경로의 시작/끝점
    user_id: str
    target_date: str
    start_point: Dict[str, Any]
    end_point: Dict[str, Any]
    user_house_address: str
    user_workplace_address: str
    
    # [추가] 가공 전 할 일 목록
    todo_list_raw: List[Dict[str, Any]]
    
    # 기존 필드들
    meta: Dict[str, Any]
    fixed_events: List[Dict[str, Any]]
    todo_items: List[Dict[str, Any]]
    distance_matrix: Dict[str, Any]
    optimized_result: Dict[str, Any]
    selection_history: List[Dict[str, Any]]