import math
import requests
import os
from dotenv import load_dotenv

load_dotenv()
KAKAO_API_KEY = os.getenv("KAKAO_API_KEY")


def get_coordinates_kakao(address):
    headers = {"Authorization": f"KakaoAK {KAKAO_API_KEY}"}
    
    # [방법 1] 정석적인 주소 검색
    try:
        res = requests.get("https://dapi.kakao.com/v2/local/search/address.json", 
                           headers=headers, params={"query": address}, timeout=5)
        data = res.json()
        if data.get("documents"):
            # 좌표가 있다면 바로 반환
            return {"x": data["documents"][0]["x"], "y": data["documents"][0]["y"]}
    except: pass

    # [방법 2] 주소 검색 실패 시 키워드/장소 검색으로 재시도 
    try:
        res = requests.get("https://dapi.kakao.com/v2/local/search/keyword.json", 
                           headers=headers, params={"query": address}, timeout=5)
        data = res.json()
        if data.get("documents"):
            return {"x": data["documents"][0]["x"], "y": data["documents"][0]["y"]}
    except: pass

    # 둘 다 실패하면 에러 로그라도 찍어야 합니다.
    print(f" '{address}'에 대한 좌표를 카카오에서 찾지 못했습니다.")
    return {"x": "0.0", "y": "0.0"}

def calculate_distance(coord1, coord2):
    """
    하버사인(Haversine) 공식을 사용하여 두 좌표 사이의 거리(미터)를 계산합니다.
    """
    R = 6371000  # 지구 반지름 (단위: m)

    try:
        # 문자열로 들어올 경우를 대비해 float 형변환
        lon1, lat1 = float(coord1['x']), float(coord1['y'])
        lon2, lat2 = float(coord2['x']), float(coord2['y'])
    except (TypeError, ValueError, KeyError):
        return 0

    # 라디안 변환
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # 하버사인 공식 수식
    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2)**2
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # 결과 반환 (반올림하여 정수 미터 단위로)
    return round(R * c)