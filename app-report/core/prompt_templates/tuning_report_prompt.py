# 뉴스 생성 프롬프트 템플릿

from ...schemas.tuning_schema import UserProfile


def build_prompt(
    category: str, chatCount: int, userA: UserProfile, userB: UserProfile
) -> str:
    def format_list(items: list[str]) -> str:
        return ", ".join(items) if items else "없음"

    def format_user(user: UserProfile, label: str) -> str:
        return (
            f"🔹 {label}\n"
            f" - 성별: {user.gender}\n"
            f" - MBTI: {user.MBTI}\n"
            f" - 종교: {user.religion}\n"
            f" - 흡연: {user.smoking}\n"
            f" - 음주: {user.drinking}\n"
            f" - 성향: {format_list(user.personality)}\n"
            f" - 선호하는 사람 유형: {format_list(user.preferredPeople)}\n"
            f" - 관심사: {format_list(user.currentInterests)}\n"
            f" - 좋아하는 음식: {format_list(user.favoriteFoods)}\n"
            f" - 좋아하는 운동: {format_list(user.likedSports)}\n"
            f" - 반려동물: {format_list(user.pets)}\n"
            f" - 자기계발 활동: {format_list(user.selfDevelopment)}\n"
            f" - 취미: {format_list(user.hobbies)}"
        )

    if category == "COUPLE":
        # 커플
        style_guide = (
            "##  스타일 가이드\n"
            "- 가십/연예 뉴스 스타일\n"
            "- 많은 이모지 활용 \n"
            "- 의문문과 감탄문 사용\n"
            "- 호기심을 자극하는 어조\n"
            "- 약간의 과장도 OK\n"
            "- 독자에게 직접 말하듯 친근하게\n"
            "- 밝고 경쾌한 톤\n"
        )
        few_shot = (
            '{{"title": "🔥MBTI 불꽃 케미?! 이 조합 실화?!🔥",\n'
            '"content": "어디선가 불꽃이 튄다?! INFP와 ESTJ의 만남은 마치 아이스크림과 고추장의 조합?! 🍦🌶️\\n\\n'
            "첫 번째 힌트! 감성 끝판왕과 현실주의의 조우!\\n\\n"
            "두 번째 힌트! 두 사람 다 운동을 좋아한다는 공통점! 같이 등산? 혹은 헬스장 러닝머신에서 조우? 🏋️‍♂️⛰️\\n\\n"
            "세 번째 힌트! 벌써 백 번은 대화했을지도?  😆\\n\\n"
            'Stay Tuned!"}}\n\n'
        )
    elif category == "FRIEND":
        # 친구
        style_guide = (
            "##  스타일 가이드\n"
            "- 가십/연예 뉴스 스타일\n"
            "- 많은 이모지 활용 \n"
            "- 의문문과 감탄문 사용\n"
            "- 호기심을 자극하는 어조\n"
            "- 청춘 드라마처럼 위트 있게\n"
            "- 약간의 과장도 OK\n"
            "- 독자에게 직접 말하듯 친근하게\n"
            "- 밝고 유쾌한 일상 톤\n"
            "- 소소한 유머와 편안한 말투\n"
        )
        few_shot = (
            "## 예시 출력 \n"
            '{{"title": "🔥 이런 친구 또 없습니다!🔥",\n'
            '"content": "첫 번째 힌트! 게임을 좋아하는 두 사람! 벌써 친해질 준비 완료 🎮✨\\n\\n'
            "두 번째 힌트!ENFP와 ISTP의 조합은 약간 엉뚱+차분 콤비!\\n\\n"
            "세 번째 힌트! 둘 다 고양이 좋아한다는 소문도? 냥덕 케미 폭발 중 😺\\n\\n"
            "첫 채팅에서 바로 TMI 파티일지도?!\\n\\n"
            'Stay Tuned!"}}\n'
        )
    else:
        # 기본
        style_guide = (
            "## 스타일 가이드\n"
            "- 가볍고 유쾌한 뉴스 스타일\n"
            "- 이모지 적절히 활용\n"
            "- 명료한 정보 전달 + 유머 섞기\n"
        )
        few_shot = (
            "## 예시 출력 \n"
            '{"title": "🎓 스터디 케미 폭발?!",\n'
            '"content": "공부할 때도 케미가 중요하죠? INTJ와 ENTJ의 만남은 완전 전략적! 📚🧠\\n\\n'
            "둘 다 자기계발 열정 만렙! 같은 목표? 같은 열정?\\n\\n"
            "이번 주말부터 같이 스터디 시작?! \\n\\n"
            'Stay Tuned!"}\n'
        )
    return (
        f"[매칭 유형: {category}]\n\n"
        f" 대화 횟수: {chatCount}\n"
        f"{format_user(userA, '사용자 A')}\n\n"
        f"{format_user(userB, '사용자 B')}\n\n"
        f"다음 기준을 바탕으로 **한국어 공지문**을 마크다운 형식으로 작성해주세요:\n\n"
        f"##  공지문 작성 규칙\n"
        f"1. **제목**: 호기심을 자극하는 문장, 이모지 적극 활용\n"
        f"2. **도입부**: 익명의 두 사람이 연결되었음을 암시하며, 정체를 바로 공개하지 않고 흥미 유발\n"
        f"3. **본문**:\n"
        f"   - 힌트 #1: 두 사람의 MBTI 조합에 대한 흥미로운 해석\n"
        f"   - 힌트 #2: 공통 관심사를 바탕으로 유쾌한 상상\n"
        f"   - 힌트 #3: 대화 횟수에 대한 유머러스한 추측\n"
        f"4. **마무리**: `Stay Tuned!` 문구를 활용해 궁금증 유발\n\n"
        f"{style_guide}\n\n"
        f"{few_shot}\n"
        f"⛔ 반드시 지켜야 할 출력 형식\n"
        f'- 출력은 반드시 JSON 형식으로만! 예시: {{"title": "...", "content": "..."}}\n'
        f"- 출력에는 절대로 영어, <think>, 사고 과정, 분석 설명이 들어가면 안 됩니다.\n"
        f"- 출력에는 오직 제목과 본문만 포함하세요. 그 외 설명이 포함될 경우 정답으로 간주하지 않습니다.\n"
        f"- 사고과정, 계획, reasoning, 분석 텍스트, 메모, 영어 설명을 절대 포함하지 마세요.\n"
        f" 공지문 외에 그 어떤 설명도 하지 마세요. 사고 과정이나 영어는 포함하지 마세요.\n"
        f"응답은 오직 마크다운 형식의 공지문 하나만 출력해야 합니다.\n"
    )


test_prompt = """
# 공지문 형식 요구사항

1. 제목: 호기심을 자극하는 제목 (이모지 활용)
2. 도입부: 새로운 연결이 성사되었다는 흥미로운 소개 (바로 공개하지 않고 호기심 유발)
3. 본문:
    "힌트 #1" - MBTI 정보와 그 조합의 흥미로운 해석
    "힌트 #2" - 공통 관심사와 이를 바탕으로 한 흥미로운 상상
    "힌트 #3" - 대화 횟수와 그 의미에 대한 재미있는 해석
4. 마무리: "Stay Tuned!" 문구를 활용한 흥미를 유지하는 문구

# 작성 스타일

가십/연예 뉴스 형식으로 작성
많은 이모지 활용
의문문과 감탄문을 적극 활용
호기심을 자극하는 어조
약간의 과장된 표현 사용
독자와 대화하는 듯한 친근한 톤
20-30대 젊은 층을 대상으로 하는 밝고 경쾌한 문체
한국어로 작성
출력결과는 마크다운 형식의 한국어 문장

# 입력값(필수 입력 정보)
MBTI: INTJ, ENFP
공통관심사: 디지털 드로잉, 넷플릭스 심야 감상
대화횟수: 17회
"""
