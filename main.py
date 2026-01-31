# 환경 변수 로드

import os
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from state import GraphState
from nodes import (
    ingest_data_node,
    candidate_node,
    selection_node,
    distance_matrix_node,
    optimization_node
)
# main.py 상단
import os

if not os.path.exists("output"):
    os.makedirs("output")

def build_graph():
    workflow = StateGraph(GraphState)

    # 1. 노드 등록
    workflow.add_node("ingest", ingest_data_node)
    workflow.add_node("candidates", candidate_node)
    workflow.add_node("selection", selection_node)
    workflow.add_node("distance", distance_matrix_node)
    workflow.add_node("optimization", optimization_node)

    # 2. 엣지 연결 (순서대로)
    workflow.set_entry_point("ingest")
    workflow.add_edge("ingest", "candidates")
    workflow.add_edge("candidates", "selection")
    workflow.add_edge("selection", "distance")
    workflow.add_edge("distance", "optimization")
    workflow.add_edge("optimization", END)

    return workflow.compile()

if __name__ == "__main__":
    # 초기 테스트 데이터 (이 구조로 에이전트 시작)
    initial_input = {
        "user_id": "user_123",
        "target_date": "2026-01-01",
        "user_house_address": "서울시 용산구 독서당로 111",
        "user_workplace_address": "서울시 서대문구 이화여대길 52",
        "start_point": {
            "name": "집",
            "address": "서울시 용산구 독서당로 111"
        },
        "end_point": {
            "name": "집",
            "address": "서울시 용산구 독서당로 111"
        },
        "fixed_events": [
            {
                "title": "컴퓨터구조 강의",
                "location": "서울시 서대문구 이화여대길 52",
                "start_time": "13:00",
                "end_time": "15:00",
                "category": "lecture"
            },
            {
                "title": "PT",
                "location": "서울시 서대문구 이화여대길 59",
                "start_time": "19:00",
                "end_time": "20:00",
                "category": "workout"
            }
        ],
        "todo_list_raw": [
            {
                "task": "서점 가서 책 사기",
                "user_duration": 60,
                "center_place": "학교 근처",
                "search_words": ["대형 서점", "전공 서적"]
            },
            {
                "task": "조별 과제 회의 준비",
                "user_duration": 90,
                "center_place": "학교 근처",
                "search_words": ["스터디룸", "회의실"]
            },
            {
                "task": "분위기 좋은 카페가서 커피 마시기",
                "user_duration": 60,
                "center_place": "집 근처",
                "search_words": []
            }
        ]
    }

    app = build_graph()
    
    final_state = app.invoke(initial_input)
    
    print("\n--- 최종 최적화 결과 ---")
    print(final_state.get("optimized_result"))