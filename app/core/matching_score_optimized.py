"""
매칭 스코어 계산 모듈
사용자 간 유사도를 계산하기 위한 다양한 알고리즘과 유틸리티 함수 제공

주요 기능:
1. 임베딩 벡터 처리 및 평균화
2. MBTI 호환성 계산
3. 연령대 일치도 계산
4. 관심사/태그 매칭 계산
5. 규칙 기반 유사도 계산
6. 최종 매칭 점수 통합 계산
"""

import json
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.utils import logger

# ---------------------- 상수 정의 ----------------------
# 모델 임베딩 차원 및 가중치 상수
EMBEDDING_DIM = 768  # SBERT 모델의 임베딩 차원
EMBEDDING_WEIGHT = 0.7  # 임베딩 기반 유사도 가중치
RULE_WEIGHT = 0.3  # 규칙 기반 유사도 가중치

# MBTI 관련 상수
MBTI_WEIGHTS = [0.5, 1.0, 1.0, 0.5]  # E/I, N/S, F/T, J/P 각 차원별 가중치

# 임베딩 계산에 사용할 필드 목록
EMBEDDING_FIELDS = [
    "currentInterests",
    "favoriteFoods",
    "likedSports",
    "pets",
    "selfDevelopment",
    "hobbies",
]

# MBTI 유형 간 호환성 맵핑
# 각 MBTI 유형에 대해 잘 맞는 상호보완적 유형 목록 정의
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

# 연령대 그룹 정의 및 순서 설정
AGE_GROUPS = {
    "AGE_10S": 1,
    "AGE_20S": 2,
    "AGE_30S": 3,
    "AGE_40S": 4,
    "AGE_50S": 5,
    "AGE_60S": 6,
}


def average_field_embedding(field_embeddings: dict, fields: list) -> list:
    """
    개별 필드 임베딩들을 평균하여 하나의 통합 벡터로 만드는 함수

    Args:
        field_embeddings: 필드별 임베딩 벡터 딕셔너리
        fields: 평균 계산에 사용할 필드 이름 목록

    Returns:
        평균 임베딩 벡터 (리스트)
    """
    # 유효한 임베딩 벡터만 추출
    vectors = [field_embeddings.get(f) for f in fields if f in field_embeddings]
    vectors = [v for v in vectors if v is not None]

    # 벡터가 없을 경우 기본 영벡터 반환
    if not vectors:
        return [0.0] * EMBEDDING_DIM

    # 평균 계산 후 리스트로 변환하여 반환
    return np.mean(np.array(vectors), axis=0).tolist()


def mbti_weighted_score(
    mbti1: str, mbti2: str, weight_similarity: float = 0.7
) -> float:
    """
    두 MBTI 유형 간의 호환성 점수 계산
    각 MBTI 차원에 가중치를 적용하고, 궁합 정보도 고려하여 종합 점수 산출

    Args:
        mbti1: 첫 번째 사용자의 MBTI
        mbti2: 두 번째 사용자의 MBTI
        weight_similarity: 유사성 점수와 호환성 점수 간의 가중치 (기본값: 0.7)

    Returns:
        0.0~1.0 사이의 호환성 점수
    """
    # MBTI 유효성 검사 (빈 값, 알 수 없음, 형식 불일치 등)
    is_invalid1 = not mbti1 or mbti1 not in MBTI_COMPATIBILITY or len(mbti1) != 4
    is_invalid2 = not mbti2 or mbti2 not in MBTI_COMPATIBILITY or len(mbti2) != 4

    # 양쪽 다 유효하지 않은 경우
    if is_invalid1 and is_invalid2:
        return 0.5  # 중간값 반환 (둘 다 없음 → 사용자 고립 & 패널티 부여 방지)
    # 한쪽만 유효하지 않은 경우
    elif is_invalid1 or is_invalid2:
        return 0.6  # 정보 불균형 상태지만, 매칭 기회는 열어둠

    # 차원별 일치 유사성 점수 계산 (0~1)
    # 각 차원(E/I, N/S, F/T, J/P)마다 일치하면 해당 가중치 적용
    similarity_score = sum(
        MBTI_WEIGHTS[i] for i in range(4) if mbti1[i] == mbti2[i]
    ) / sum(MBTI_WEIGHTS)

    # 상호보완적 호환성 점수 계산 (0 또는 1)
    # 정의된 상호보완적 유형 목록에 포함되면 최대 점수
    compatibility_score = 1.0 if mbti2 in MBTI_COMPATIBILITY.get(mbti1, []) else 0.0

    # 가중치를 적용한 최종 점수 계산
    final_score = (weight_similarity * similarity_score) + (
        (1 - weight_similarity) * compatibility_score
    )

    return round(final_score, 6)


def age_group_match_score(a: str, b: str) -> float:
    """
    두 연령대 간의 일치도 점수 계산

    Args:
        a: 첫 번째 사용자의 연령대
        b: 두 번째 사용자의 연령대

    Returns:
        연령대 일치도 점수:
        - 완전 일치: 1.0
        - 한 단계 차이(예: 20대-30대): 0.5
        - 두 단계 이상 차이: 0.0
    """
    # 연령대 코드를 숫자 값으로 변환
    a_val = AGE_GROUPS.get(a)
    b_val = AGE_GROUPS.get(b)

    # 유효하지 않은 연령대 코드인 경우
    if a_val is None or b_val is None:
        return 0.0

    # 연령대 차이 계산
    diff = abs(a_val - b_val)

    # 차이에 따른 점수 반환
    if diff == 0:  # 동일 연령대
        return 1.0
    elif diff == 1:  # 인접 연령대 (예: 20대-30대)
        return 0.5
    else:  # 2단계 이상 차이
        return 0.0


def match_tags(list1: List[str], list2: List[str]) -> float:
    """
    두 태그 목록 간의 유사도 계산 (자카드 유사도)

    Args:
        list1: 첫 번째 태그 목록
        list2: 두 번째 태그 목록

    Returns:
        0.0~1.0 사이의 유사도 점수 (공통 태그 수 / 전체 고유 태그 수)
    """
    # 빈 목록 처리
    if not list1 or not list2:
        return 0.0

    # 교집합과 합집합 계산
    overlap = set(list1) & set(list2)
    union = set(list1) | set(list2)

    # 자카드 유사도 계산 및 반환
    return round(len(overlap) / len(union), 6)


def rule_based_similarity(user1: dict, user2: dict) -> float:
    """
    사용자 프로필 데이터를 기반으로 규칙 기반 유사도 점수 계산
    여러 속성(종교, 흡연, 음주, MBTI, 연령대, 성격 등)의 일치도를 종합

    Args:
        user1: 첫 번째 사용자 프로필 데이터
        user2: 두 번째 사용자 프로필 데이터

    Returns:
        0.0~1.0 사이의 규칙 기반 유사도 점수
    """
    # 기본 필드 일치도 (종교, 흡연, 음주 등) - 단순 일치 여부 확인
    base_fields = ["religion", "smoking", "drinking"]
    base_score = sum(1 for f in base_fields if user1.get(f) == user2.get(f)) / len(
        base_fields
    )

    # MBTI 호환성 점수
    mbti_score = mbti_weighted_score(user1.get("MBTI"), user2.get("MBTI"))

    # 연령대 일치도 점수
    age_score = age_group_match_score(user1.get("ageGroup"), user2.get("ageGroup"))

    # 선호-성격 매칭 점수 (양방향)
    # user1의 선호 특성과 user2의 성격 특성 비교
    pref_score = match_tags(
        user1.get("preferredPeople", []), user2.get("personality", [])
    )
    # user2의 선호 특성과 user1의 성격 특성 비교
    rev_pref_score = match_tags(
        user2.get("preferredPeople", []), user1.get("personality", [])
    )

    # 가중치를 적용한 최종 점수 계산
    # - 기본 필드(종교,흡연,음주): 30%
    # - MBTI 호환성: 20%
    # - 연령대 일치도: 20%
    # - 선호-성격 매칭(양방향 평균): 30%
    final_score = (
        base_score * 0.3
        + mbti_score * 0.2
        + age_score * 0.2
        + (pref_score + rev_pref_score) / 2 * 0.3
    )

    # 소수점 6자리로 반올림하여 반환
    return round(final_score, 6)


@logger.log_performance(operation_name="compute_matching_score", include_memory=True)
def compute_matching_score(
    user_id: str, user_embedding: List[float], user_meta: dict, all_users: dict
) -> Dict[str, float]:
    """
    사용자와 다른 사용자들 간의 매칭 점수를 계산
    임베딩 벡터 유사도와 규칙 기반 유사도를 결합하여 종합 점수 산출

    Args:
        user_id: 기준 사용자 ID
        user_embedding: 기준 사용자의 임베딩 벡터
        user_meta: 기준 사용자의 메타데이터
        all_users: 전체 사용자 데이터 (IDs, 임베딩, 메타데이터)

    Returns:
        사용자 ID를 키로, 매칭 점수를 값으로 하는 딕셔너리
    """
    # 결과 저장용 딕셔너리
    similarities = {}

    # 전체 사용자 데이터 추출
    all_ids = all_users["ids"]
    all_embeddings = all_users["embeddings"]
    all_metas = all_users["metadatas"]

    # 기준 사용자의 도메인(소속 조직) 추출
    domain = user_meta.get("emailDomain")

    # 기준 사용자의 필드별 임베딩 추출 및 평균 계산
    my_fields = json.loads(user_meta.get("field_embeddings", "{}"))
    my_avg_embed = average_field_embedding(my_fields, EMBEDDING_FIELDS)

    # 프로필 임베딩과 필드 임베딩을 결합
    combined_user_embedding = np.array(user_embedding) + np.array(my_avg_embed)

    # 모든 사용자에 대해 매칭 점수 계산
    for i, other_id in enumerate(all_ids):
        # 자기 자신 제외
        if other_id == user_id:
            continue

        # 다른 도메인(조직) 사용자 제외 - 같은 조직 내에서만 매칭
        other_meta = all_metas[i]
        if other_meta.get("emailDomain") != domain:
            continue

        # 상대방 필드 임베딩 추출 및 평균 계산
        other_fields = json.loads(other_meta.get("field_embeddings", "{}"))
        other_avg_embed = average_field_embedding(other_fields, EMBEDDING_FIELDS)

        # 상대방 임베딩 결합
        combined_other_embedding = np.array(all_embeddings[i]) + np.array(
            other_avg_embed
        )

        # 코사인 유사도 계산 (임베딩 기반 유사도)
        cosine_sim = float(
            cosine_similarity([combined_user_embedding], [combined_other_embedding])[0][
                0
            ]
        )

        # 규칙 기반 유사도 계산
        rule_sim = rule_based_similarity(user_meta, other_meta)

        # 최종 매칭 점수 계산 (임베딩 70% + 규칙 30%)
        final_score = EMBEDDING_WEIGHT * cosine_sim + RULE_WEIGHT * rule_sim
        similarities[other_id] = round(final_score, 6)

    return similarities


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    벡터를 L2 정규화하여 단위 벡터로 변환

    Args:
        vector: 정규화할 벡터

    Returns:
        정규화된 단위 벡터
    """
    norm = np.linalg.norm(vector)
    if norm == 0:  # 영벡터인 경우 처리
        return vector
    return vector / norm


def combine_embeddings(
    profile_embedding: List[float],
    field_embeddings: dict,
) -> np.ndarray:
    """
    프로필 임베딩과 필드별 임베딩을 가중 평균으로 결합

    Args:
       profile_embedding: 프로필 임베딩 벡터
       field_embeddings: 필드별 임베딩 사전

    Returns:
        결합된 임베딩 벡터
    """
    # 프로필 임베딩을 numpy 배열로 변환
    profile_embed = np.array(profile_embedding)

    # 필드 임베딩 추출 및 평균 계산
    avg_field_embed = np.array(
        average_field_embedding(field_embeddings, EMBEDDING_FIELDS)
    )

    # 정규화된 벡터의 가중 평균
    norm_profile = normalize_vector(profile_embed)
    norm_fields = normalize_vector(avg_field_embed)

    # 프로필(60%)과 필드 임베딩(40%)의 가중 평균
    return 0.6 * norm_profile + 0.4 * norm_fields


@logger.log_performance(
    operation_name="compute_matching_score_optimized", include_memory=True
)
def compute_matching_score_optimized(
    user_id: str,
    user_embedding: List[float],
    user_meta: dict,
    all_users: dict,
    embedding_method: str = "weighted_average",
) -> Dict[str, float]:
    """
    최적화된 매칭 점수 계산 함수
    벡터화 및 배치 처리를 통해 성능 개선

    Args:
        user_id: 기준 사용자 ID
        user_embedding: 기준 사용자의 임베딩 벡터
        user_meta: 기준 사용자의 메타데이터
        all_users: 전체 사용자 데이터 (IDs, 임베딩, 메타데이터)
        embedding_method: 임베딩 결합 방식

    Returns:
        사용자 ID를 키로, 매칭 점수를 값으로 하는 딕셔너리
    """
    # 전체 사용자 데이터 추출
    all_ids = all_users["ids"]
    all_embeddings = all_users["embeddings"]
    all_metas = all_users["metadatas"]

    # 1. 도메인 필터링 먼저 수행 (동일 도메인 사용자만 선택)
    domain = user_meta.get("emailDomain")
    domain_indices = [
        i
        for i, meta in enumerate(all_metas)
        if meta.get("emailDomain") == domain and all_ids[i] != user_id
    ]

    # 같은 도메인 사용자가 없으면 빈 결과 반환
    if not domain_indices:
        return {}

    # 2. 개선된 임베딩 결합 적용
    my_fields = json.loads(user_meta.get("field_embeddings", "{}"))
    combined_user_embedding = combine_embeddings(
        user_embedding, my_fields, method=embedding_method
    )

    # 3. 한번에 처리할 임베딩 및 메타데이터 준비
    other_ids = [all_ids[i] for i in domain_indices]
    other_embeddings = []
    other_metas_filtered = []

    # 도메인 필터링된 사용자들의 임베딩과 메타데이터 수집
    for i in domain_indices:
        other_meta = all_metas[i]
        other_fields = json.loads(other_meta.get("field_embeddings", "{}"))
        combined_other_embedding = combine_embeddings(
            all_embeddings[i], other_fields, method=embedding_method
        )
        other_embeddings.append(combined_other_embedding)
        other_metas_filtered.append(other_meta)

    # 4. 벡터화된 유사도 계산 (배치 처리)
    other_embeddings_matrix = np.vstack(other_embeddings)
    cosine_sims = cosine_similarity([combined_user_embedding], other_embeddings_matrix)[
        0
    ]

    # 5. 규칙 기반 유사도 및 최종 점수 계산
    similarities = {}
    for idx, other_id in enumerate(other_ids):
        # 각 사용자에 대한 규칙 기반 유사도 계산
        rule_sim = rule_based_similarity(user_meta, other_metas_filtered[idx])

        # 임베딩 기반 유사도와 규칙 기반 유사도를 결합하여 최종 점수 계산
        final_score = EMBEDDING_WEIGHT * cosine_sims[idx] + RULE_WEIGHT * rule_sim
        similarities[other_id] = round(final_score, 6)

    return similarities
