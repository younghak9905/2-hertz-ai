# 매칭 스코어 계산
import json

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# MBTI 가중 점수
# 축 | 설명 | 예시 점수
# E/I | 외향 ↔ 내향 | 0.5 중요
# N/S | 직관 ↔ 감각 | 1.0 중요 (정보 수용 방식 차이)
# F/T | 감정 ↔ 사고 | 1.0 중요 (판단 차이)
# J/P | 판단 ↔ 인식 | 0.5 중요
def mbti_weighted_score(mbti1, mbti2):
    MBTI_WEIGHTS = [0.5, 1.0, 1.0, 0.5]  # E/I, N/S, F/T, J/P
    if not mbti1 or not mbti2 or len(mbti1) != 4 or len(mbti2) != 4:
        return 0.0
    score = 0.0
    for i in range(4):
        if mbti1[i] == mbti2[i]:
            score += MBTI_WEIGHTS[i]
    return score / sum(MBTI_WEIGHTS)


# 연령대 점수
def age_group_match_score(a, b):
    return 1.0 if a == b else 0.0


# preferredPeople - personality 같을수록 높은 점수
def match_tags(list1, list2):
    if not list1 or not list2:
        return 0.0
    overlap = set(list1).intersection(set(list2))
    return len(overlap) / len(set(list1).union(set(list2)))


# MBTI, ageGroup, preferredPeople-personalirty 는 지정된 매칭 스코어 계산 방법 적용
def rule_based_similarity(user1: dict, user2: dict) -> float:
    # 간단한 룰 기반 유사도
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

    return final_score if final_score > 0 else 0.0


def average_field_embedding(field_embeddings: dict, fields: list) -> list:
    vectors = [field_embeddings.get(f) for f in fields if f in field_embeddings]
    vectors = [v for v in vectors if v is not None]
    if not vectors:
        return [0.0] * 768  # SBERT 임베딩 차원
    return np.mean(np.array(vectors), axis=0).tolist()


def compute_matching_score(
    user_id: str, user_embedding: list, user_meta: dict, all_users: dict
) -> dict:
    """
    코사인 + 룰 기반 유사도를 결합하여 유저와 같은 도메인의 사용자들과의 매칭 점수를 계산
    """
    similarities = {}
    all_ids = all_users["ids"]
    all_embeddings = np.array(all_users["embeddings"])
    all_users_metas = all_users["metadatas"]

    target_domain = user_meta.get("emailDomain")

    embedding_fields = [
        "currentInterests",
        "favoriteFoods",
        "likedSports",
        "pets",
        "selfDevelopment",
        "hobbies",
    ]

    my_field_embeds = json.loads(user_meta.get("field_embeddings", "{}"))
    my_avg_field_embed = average_field_embedding(my_field_embeds, embedding_fields)

    # 최종 내 임베딩 = 프로필 + 필드 평균 벡터 (결합 or 대체)
    combined_user_embedding = np.array(user_embedding) + np.array(my_avg_field_embed)

    for i, other_id in enumerate(all_ids):
        if other_id == user_id:
            continue

        other_meta = all_users_metas[i]
        if other_meta.get("emailDomain") != target_domain:
            continue

        # 상대방 필드 임베딩 평균
        other_field_embeds = json.loads(other_meta.get("field_embeddings", "{}"))
        other_avg_field_embed = average_field_embedding(
            other_field_embeds, embedding_fields
        )
        combined_other_embedding = np.array(all_embeddings[i]) + np.array(
            other_avg_field_embed
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
        final_score = 0.7 * cosine_sim + 0.3 * rule_sim
        similarities[other_id] = round(final_score, 6)

    return similarities
