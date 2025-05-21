# 매칭 스코어 계산
import json
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import logger

# ---------------------- 상수 정의 ----------------------
EMBEDDING_DIM = 768
EMBEDDING_WEIGHT = 0.7
RULE_WEIGHT = 0.3
MBTI_WEIGHTS = [0.5, 1.0, 1.0, 0.5]  # E/I, N/S, F/T, J/P
EMBEDDING_FIELDS = [
    "currentInterests",
    "favoriteFoods",
    "likedSports",
    "pets",
    "selfDevelopment",
    "hobbies",
]

MBTI_COMPATIBILITY = {
    "INTJ": ["ENFP", "ENTP"],
    "INTP": ["ENTJ", "ENFJ"],
    "INFJ": ["ENFP", "ENTP"],
    "INFP": ["ENFJ", "ESFJ"],
    "ISTJ": ["ESFP", "ESTP"],
    "ISTP": ["ESFJ", "ENFJ"],
    "ISFJ": ["ESTP", "ESFP"],
    "ISFP": ["ENFJ", "ESFJ"],
    "ENTJ": ["INFP", "INTP"],
    "ENTP": ["INFJ", "INTJ"],
    "ENFJ": ["INFP", "ISFP"],
    "ENFP": ["INFJ", "INTJ"],
    "ESTJ": ["ISFP", "ISTP"],
    "ESTP": ["ISFJ", "ISTJ"],
    "ESFJ": ["ISFP", "INFP"],
    "ESFP": ["ISFJ", "ISTJ"],
}

AGE_GROUPS = {
    "AGE_10S": 1,
    "AGE_20S": 2,
    "AGE_30S": 3,
    "AGE_40S": 4,
    "AGE_50S": 5,
    "AGE_60S": 6,
}
# ---------------------- 상수 정의 ----------------------

MBTI_TOTAL_WEIGHT = sum(MBTI_WEIGHTS)


# 개별 필드 임베딩들을 평균 내어, 하나의 통합 벡터로 만드는 함수
def average_field_embedding(field_embeddings: dict, fields: list) -> list:
    vectors = [field_embeddings.get(f) for f in fields if f in field_embeddings]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return [0.0] * EMBEDDING_DIM
    return np.mean(np.array(vectors), axis=0).tolist()


# MBTI 매칭 스코어 계산
def mbti_weighted_score(
    mbti1: str, mbti2: str, weight_similarity: float = 0.7
) -> float:

    # MBTI: UNKWOUN일 경우 처리
    is_invalid1 = not mbti1 or mbti1 not in MBTI_COMPATIBILITY or len(mbti1) != 4
    is_invalid2 = not mbti2 or mbti2 not in MBTI_COMPATIBILITY or len(mbti2) != 4

    if is_invalid1 and is_invalid2:
        return 0.5  # 둘 다 없음 → 사용자 고립 & 패널티 부여 방지
    elif is_invalid1 or is_invalid2:
        return 0.6  # 한 쪽만 없음 → 정보 불균형 상태지만, 매칭 기회는 열어둠

    # 글자 일치 유사도 + 궁합 기반 스코어 계산 시작
    # 유사성 점수 (유형 일치 기반)
    similarity_score = (
        sum(MBTI_WEIGHTS[i] for i in range(4) if mbti1[i] == mbti2[i])
        / MBTI_TOTAL_WEIGHT
    )

    # 궁합 점수 (상보적 궁합에 포함되면 1.0, 아니면 0.0)
    compatibility_score = 1.0 if mbti2 in MBTI_COMPATIBILITY.get(mbti1, []) else 0.0

    # 가중 평균
    final_score = (weight_similarity * similarity_score) + (
        (1 - weight_similarity) * compatibility_score
    )
    return round(final_score, 6)


# 연령대 점수
# 같은 연령대: 1.0
# 한 단계 차이 (ex. 20대 vs 30대): 0.5
# 두 단계 이상 차이 (ex. 20대 vs 40대 이상): 0.0
def age_group_match_score(a: str, b: str) -> float:
    a_val = AGE_GROUPS.get(a)
    b_val = AGE_GROUPS.get(b)

    if a_val is None or b_val is None:
        return 0.0

    diff = abs(a_val - b_val)

    if diff == 0:
        return 1.0
    elif diff == 1:
        return 0.5
    else:
        return 0.0


# preferredPeople - personality 같을수록 높은 점수
def match_tags(list1: List[str], list2: List[str]) -> float:
    if not list1 or not list2:
        return 0.0
    overlap = set(list1) & set(list2)
    union = set(list1) | set(list2)
    return round(len(overlap) / len(union), 6)


# ---------------------- 룰 기반 유사도 ----------------------
# MBTI, ageGroup, preferredPeople-personalirty 는 지정된 매칭 스코어 계산 방법 적용
def rule_based_similarity(user1: dict, user2: dict) -> float:
    base_fields = ["religion", "smoking", "drinking"]
    base_score = sum(1 for f in base_fields if user1.get(f) == user2.get(f)) / len(
        base_fields
    )

    mbti_score = mbti_weighted_score(user1.get("MBTI"), user2.get("MBTI"))
    age_score = age_group_match_score(user1.get("ageGroup"), user2.get("ageGroup"))

    pref_score = match_tags(
        user1.get("preferredPeople", []), user2.get("personality", [])
    )
    rev_pref_score = match_tags(
        user2.get("preferredPeople", []), user1.get("personality", [])
    )

    # 가중치 조합
    final_score = (
        base_score * 0.3
        + mbti_score * 0.2
        + age_score * 0.2
        + (pref_score + rev_pref_score) / 2 * 0.3
    )

    return round(final_score, 6)


# ---------------------- 최종 매칭 점수 ----------------------


@logger.log_performance(operation_name="compute_matching_score", include_memory=True)
def compute_matching_score(
    user_id: str, user_embedding: List[float], user_meta: dict, all_users: dict
) -> Dict[str, float]:
    """
    코사인 + 룰 기반 유사도를 결합하여 유저와 같은 도메인의 사용자들과의 매칭 점수를 계산
    """
    similarities = {}
    all_ids = all_users["ids"]
    all_embeddings = all_users["embeddings"]
    all_metas = all_users["metadatas"]

    domain = user_meta.get("emailDomain")
    my_fields = json.loads(user_meta.get("field_embeddings", "{}"))
    my_avg_embed = average_field_embedding(my_fields, EMBEDDING_FIELDS)
    # 최종 내 임베딩 = 프로필 + 필드 평균 벡터 (결합 or 대체)
    combined_user_embedding = np.array(user_embedding) + np.array(my_avg_embed)

    for i, other_id in enumerate(all_ids):
        if other_id == user_id:
            continue

        other_meta = all_metas[i]
        if other_meta.get("emailDomain") != domain:
            continue

        # 상대방 필드 임베딩 평균
        other_fields = json.loads(other_meta.get("field_embeddings", "{}"))
        other_avg_embed = average_field_embedding(other_fields, EMBEDDING_FIELDS)
        combined_other_embedding = np.array(all_embeddings[i]) + np.array(
            other_avg_embed
        )

        # 코사인 유사도
        cosine_sim = float(
            cosine_similarity([combined_user_embedding], [combined_other_embedding])[0][
                0
            ]
        )

        # 룰 기반 유사도
        rule_sim = rule_based_similarity(user_meta, other_meta)

        # 최종 매칭 점수
        # (정량적인 유사도 측정에 더 큰 가중치 - 250501)
        final_score = EMBEDDING_WEIGHT * cosine_sim + RULE_WEIGHT * rule_sim
        similarities[other_id] = round(final_score, 6)

    return similarities
