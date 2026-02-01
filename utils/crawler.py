import asyncio
from playwright.async_api import async_playwright
from utils import get_coordinates_kakao

# 브라우저 실행 함수 (비동기)
async def launch_browser_async():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(
        headless=False,
        args=[
            "--disable-http2",              # SSL/인증서 관련 에러 방지
            "--ignore-certificate-errors",  # 인증서 에러 무시
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-blink-features=AutomationControlled" # 자동화 탐지 우회
        ]
    )
    context = await browser.new_context(
       ignore_https_errors=True,           # 하얀 창 방지 핵심 옵션
        viewport={'width': 1280, 'height': 800},
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )

    page = await context.new_page()
    return pw, browser, page

async def crawl_naver_places_async(search_words: str, todo_id: str, max_results=3):
    pw, browser, page = await launch_browser_async()
    candidates = []

    # 제거할 불필요한 키워드 목록
    stop_words = ["네이버페이", "예약", "톡톡", "쿠폰", "카페,디저트", "스터디카페", "독서실", "독립서점", "서점"]

    try:
        url = f"https://map.naver.com/p/search/{search_words}"
        await page.goto(url, wait_until="commit", timeout=60000)
        await asyncio.sleep(4) # 로딩 대기

        await page.wait_for_selector("iframe#searchIframe", timeout=20000)
        search_iframe = await page.query_selector("iframe#searchIframe")
        frame = await search_iframe.content_frame()

        if frame:
            await frame.wait_for_selector("li", timeout=10000)

            #스크롤을 살짝 내려서 데이터를 활성화 (네이버 지도 특징)
            await frame.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(5)

            cards = await frame.query_selector_all("li")
            count = 0
            for card in cards:
                if count >= max_results: break

                # 1. 매장명 추출: .YwYLL 또는 .place_bluelink 안의 텍스트만 타겟팅
                # .TYp9e 안에서도 실제 이름만 있는 span을 찾거나 inner_text를 정제합니다.
                name_el = await card.query_selector(".TYp9e, .YwYLL, .place_bluelink, .C_N_u")
                if not name_el: continue

                full_name = await name_el.inner_text()
                # 줄바꿈으로 구분된 경우 첫 줄이 보통 매장명입니다.
                clean_name = full_name.split('\n')[0].strip()

                # 2. 불필요한 접미사 제거 로직 (예: "갈십리카페,디저트" -> "갈십리")
                for word in stop_words:
                    if clean_name.endswith(word):
                        clean_name = clean_name.replace(word, "").strip()

                # 3. 주소 추출: 주소가 비어있는 문제 해결을 위한 셀렉터 보강
                addr_el = await card.query_selector(".Pb4bU, .addr, .address, .info_item.address, .Vp_7n")
                address = ""
                if addr_el:
                    address = await addr_el.get_attribute("title") or await addr_el.inner_text()

                if clean_name:
                    candidates.append({
                        "id": f"{todo_id}_candidates_{count+1}",
                        "name": clean_name,
                        "address": [address.strip().replace("\n", " ")],
                        "coordinates": []
                    })
                    count += 1
        print(f"{search_words} 데이터 수집 완료!")

    except Exception as e:
        print(f"크롤링 에러 발생 ({search_words}): {e}")
    finally:
        print("브라우저 리소스를 해제합니다")
        if browser:
            await browser.close()
        if pw:
            await pw.stop()

    return candidates

# 메인 실행 함수
async def attach_candidates_with_crawling_async(todo_items):
    for todo in todo_items:
        print(f"Crawling: {todo['search_words']}...")
        raw_candidates = await crawl_naver_places_async(todo["search_words"], todo["id"])
        
        valid_candidates = []
        for cand in raw_candidates:
            # 1. 주소 데이터 정제 (기존 로직 유지)
            raw_addr = cand.get("address", "")
            clean_addr = ""
            if isinstance(raw_addr, list):
                valid_parts = [str(a).strip() for a in raw_addr if a and str(a).strip()]
                clean_addr = valid_parts[0] if valid_parts else ""
            else:
                clean_addr = str(raw_addr).strip() if raw_addr and raw_addr != "[]" else ""

            # 2. 좌표 체크 및 보정
            curr_coords = cand.get("coordinates", {})
            if not curr_coords or str(curr_coords.get("x")) == "0.0":
                # 주소가 유효하면 주소 사용, 아니면 이름 사용
                base_query = clean_addr if clean_addr else cand.get("name", "")
                
                if base_query:
                    # 검색어 최적화: 콤마나 공백으로 잘라 핵심 상호명만 추출
                    # 예: '돌삼카페, 레스토랑 삼각지점' -> '돌삼카페'
                    refined_query = base_query.split(',')[0].split(' ')[0]
                    
                    # 너무 짧은 경우(한 글자 등)를 대비해 안전장치 추가
                    final_query = refined_query if len(refined_query) > 1 else base_query
                    
                    print(f" 좌표 재시도 ({cand.get('name')}) -> 최종 검색어: '{final_query}'")
                    cand["coordinates"] = get_coordinates_kakao(final_query)

            # 3. 최종 필터링
            final_coords = cand.get("coordinates", {})
            if final_coords and str(final_coords.get("x")) != "0.0":
                valid_candidates.append(cand)
            else:
                print(f"   🗑️ 제외됨 (좌표 없음): {cand.get('name')}")
        
        todo["candidates"] = valid_candidates
        todo["status"] = "need_selection"
        
    return todo_items