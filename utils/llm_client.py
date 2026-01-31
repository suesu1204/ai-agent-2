import os
from openai import OpenAI
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

def get_llm_client():
    """
    Upstage Solar API를 사용하기 위한 OpenAI 클라이언트를 생성하여 반환합니다.
    """
    api_key = os.getenv("UPSTAGE_API_KEY")
    base_url = "https://api.upstage.ai/v1" # Solar 모델 기본 엔드포인트

    if not api_key:
        raise ValueError("UPSTAGE_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )
    return client

# 노드에서 간편하게 사용할 수 있도록 미리 인스턴스화된 클라이언트를 제공할 수도 있습니다.
llm_client = get_llm_client()