import os
import re
import time
from datetime import datetime

import google.generativeai as genai
import pandas as pd
from candidates import SAMPLE_NEWS
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

model = genai.GenerativeModel("gemini-2.5-flash")


def create_evaluation_prompt(sample_data):
    """평가용 프롬프트 생성"""
    sample_json = (
        f'{{"title": "{sample_data["title"]}", "content": "{sample_data["content"]}"}}'
    )

    prompt = f"""
당신은 소셜 디스커버리 앱의 매칭 뉴스 품질을 평가하는 전문가입니다.

## 📋 평가 원칙
- **구체적 개선점 제시**: 만점이 아닌 모든 항목에 대해 명확한 개선 방향 제시 필수
- **통합적 평가**: 긍정적 요소와 부정적 요소를 종합하여 하나의 점수로 평가
- **만점 부여 기준**: 혁신적이고 완벽무결한 수준에만 만점 부여
- **스토리 중심의 평가 원칙**:'뉴스', '소설', '시' 등 어떤 창의적인 형식을 사용하더라도, 그것이 제공된 사용자 데이터를 바탕으로 매력적인 스토리를 구성하고 독자의 흥미를 유발하는 최종 목적에 기여하는지 여부를 최우선으로 평가하십시오. 형식이 독창적이더라도 데이터 기반의 스토리텔링이 빈약하다면 높은 점수를 부여할 수 없습니다.
- **상충 요소 판단 원칙**: 하나의 특징이 다른 지표에서는 장점, 다른 지표에서는 단점이 될 경우, 각 지표를 독립적으로 평가

---

## 🎯 평가지표 (총 32점)

### 1. 흥미도 및 몰입감 (6점)
독자의 관심을 끌고 끝까지 읽게 만드는 매력적인 콘텐츠 정도

**점수 기준:**
- 6점: 혁신적이고 독창적인 접근으로 독자를 완전히 몰입시킴. 업계 최고 수준의 창의성과 스토리텔링
- 5점: 매우 흥미롭고 독창적이나, 일부 예측 가능한 요소나 미세한 아쉬움 존재
- 4점: 충분히 흥미롭고 읽을 만하나, 특별함이나 독창성 부족
- 3점: 기본적인 흥미는 유발하나, 임팩트나 차별화 요소 제한적
- 2점: 다소 흥미로우나 집중력 유지 어려움, 뻔한 내용이 많음
- 1점: 흥미가 떨어지고 지루함, 독자 이탈 가능성 높음

**평가 시 고려사항:**
- ✓ 독창적이고 신선한 접근 방식, 예상을 뒤엎는 전개나 관점
- ✗ 반복적이거나 진부한 표현 사용, 어색하거나 부자연스러운 문체

### 2. 정보 풍부함 (5점)
다양하고 실용적인 정보 제공 정도

**점수 기준:**
- 5점: 매우 다양하고 심층적인 정보, 최신 트렌드와 실용적 조언이 완벽하게 조화
- 4점: 풍부하고 유용한 정보 제공, 일부 영역에서 깊이 부족
- 3점: 기본적인 정보는 충족하나, 다양성이나 깊이 제한적
- 2점: 정보량이 부족하거나 피상적, 실용성 떨어짐
- 1점: 정보가 매우 부족하고 내용이 빈약

**평가 시 고려사항:**
- ✓ 정확하고 최신의 정보 제공, 실용적이고 구체적인 조언
- ✗ 부정확하거나 오래된 정보 포함, 시의성이 떨어지는 내용

### 3. 개인화 및 스토리텔링 (4점)
개인 정보를 활용한 매력적인 스토리 구성 정도

**점수 기준:**
- 4점: 사용자 정보를 매우 창의적으로 활용하여, 두 인물의 관계나 특성이 생생하게 드러나는 하나의 완벽한 스토리를 완성함. 데이터가 서사의 핵심 동력이 되어 독자의 몰입감을 극대화시킴.
- 3점: 사용자 정보를 잘 활용하여 관계의 특징을 설명하고 스토리를 구성했으나, 일부 데이터가 단순 나열되거나 서사적 연결이 다소 부족하여 깊이가 아쉬움.
- 2점: 사용자 정보를 활용했지만, 스토리텔링보다는 단순 정보 나열이나 분석에 그침. 독자가 관계의 흐름이나 감정을 느끼기 어려움.
- 1점: 사용자 정보 활용이 매우 제한적이거나, 스토리 구성 없이 단편적인 사실만 제시.

**평가 시 고려사항:**
- ✓ 사용자 데이터(MBTI, 관심사, 대화 횟수 등)를 기반으로 인물의 개성이나 관계의 역동성을 생생하게 묘사
- ✓ 데이터와 스토리가 자연스럽게 융합되어, 독자가 '두 사람'의 이야기에 흥미를 느끼고 몰입 가능
- ✓ 단순한 정보 전달을 넘어, 관계에 대한 매력적인 서사(내러티브) 부여
- ✗ 데이터가 스토리와 분리되어 단순 나열되거나, 기계적으로 분석됨
- ✗ 누구에게나 적용될 수 있는 일반적인 설명에 그침
- ✗ 스토리의 기승전결이나 감정선 없이, 건조한 보고서 형식에 가까움

### 4. 엔터테인먼트 가치 (3점)
재미와 즐거움을 주는 오락적 요소 정도

**점수 기준:**
- 3점: 매우 재미있고 독창적인 유머, 위트가 뛰어나며 오락적 가치 최고 수준
- 2점: 재미있고 즐거운 요소가 적절히 포함되나, 임팩트나 독창성 부족
- 1점: 기본적인 재미 요소는 있으나 밋밋하고 예측 가능

**평가 시 고려사항:**
- ✓ 자연스럽고 적절한 유머와 위트, 독자를 즐겁게 하는 창의적 요소
- ✗ 부적절하거나 어색한 유머 시도

### 5. 가독성 및 구성 (2점)
읽기 쉬움과 시각적 매력 정도

**점수 기준:**
- 2점: 완벽한 가독성과 시각적 매력, 이모지와 서식 활용이 탁월하고 정보 전달 최적화
- 1점: 기본적인 가독성은 갖추었으나, 시각적 매력이나 구성에 아쉬움

**평가 시 고려사항:**
- ✓ 적절한 이모지와 서식 활용, 논리적이고 일관된 구성
- ✗ 과도한 이모지 사용으로 가독성 저해, 구성이 산만하거나 일관성 부족

### 6. 예측 및 기대감 조성 (5점)
앞으로의 관계 발전을 독자가 상상하고 기대하게 만드는 정도

**점수 기준:**
- 5점: 혁신적이고 기발한 상상력으로 독자에게 폭발적인 기대감과 호기심 유발
- 4점: 흥미로운 미래 예측과 구체적인 활동 제안으로 적절한 기대감 조성
- 3점: 기본적인 미래 예측이나 제안은 있으나, 특별함이나 창의성 부족
- 2점: 미래 예측이 모호하거나 일반적, 기대감 조성 제한적
- 1점: 미래에 대한 언급이나 기대감 조성이 거의 없음

**평가 시 고려사항:**
- ✓ 구체적이고 실현 가능한 미래 시나리오 제시, 독자의 호기심과 기대감을 자극하는 창의적 제안
- ✗ 현실성이 떨어지거나 너무 뻔한 예측

### 7. 공감대 형성 및 관계에 대한 감성적 몰입 (4점)
독자가 리포트 속 인물들의 관계와 감정선에 공감하고, 그들의 미래를 응원하고 싶게 만드는 정도

**점수 기준:**
- 4점: 인물들의 관계와 감정선에 대한 묘사가 매우 뛰어나, 독자가 마치 자신의 친구 이야기처럼 깊이 몰입하고 그들의 관계를 진심으로 응원하게 만듦.
- 3점: 인물들의 특징과 상황을 통해, 독자가 그들의 관계에 긍정적인 감정을 느끼고 공감대를 형성함.
- 2점: 인물들의 관계에 대한 기본적인 설명은 있으나, 감정 묘사가 부족하여 깊은 공감이나 몰입을 이끌어내기 어려움.
- 1점: 인물 묘사가 피상적이어서 독자가 관계에 대해 아무런 감흥을 느끼기 어려움.

**평가 시 고려사항:**
- ✓ 인물들의 감정이나 관계의 역동성(다이내믹)에 대한 섬세한 묘사
- ✓ 독자가 '이 두 사람, 정말 잘 됐으면 좋겠다'라고 느끼게 만드는 긍정적 서사
- ✓ 독자가 자신의 연애나 우정 경험을 떠올리며 보편적인 공감대를 형성할 수 있는 요소
- ✗ 인물 묘사가 피상적이거나 관계에 대한 분석이 건조함
- ✗ 독자가 관계의 발전에 대해 무관심하게 만드는 서술

### 8. 데이터 활용의 창의성 (3점)
주어진 데이터를 창의적이고 재치 있게 해석하고 활용한 정도

**점수 기준:**
- 3점: 데이터를 매우 창의적이고 혁신적인 방식으로 해석하여 스토리에 완벽하게 융합
- 2점: 데이터를 흥미롭게 활용하려는 시도가 보이나, 창의성이나 완성도 아쉬움
- 1점: 데이터 활용이 피상적이거나 기계적, 창의성 부족

**평가 시 고려사항:**
- ✓ 데이터의 창의적이고 독특한 해석, 데이터와 스토리의 자연스러운 연결
- ✗ 데이터 오해석이나 왜곡, 단순 나열식 데이터 활용

---

## 📝 평가 대상 텍스트
{sample_json}

---

## 📋 출력 형식
반드시 다음 형식으로만 답변해주세요. 다른 설명 없이 정확히 이 형식을 따라주세요:

**평가 대상:**
[title 내용]

**평가 결과:**
1. 흥미도 및 몰입감: (점수)/6점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

2. 정보 풍부함: (점수)/5점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

3. 개인화 및 스토리텔링: (점수)/4점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

4. 엔터테인먼트 가치: (점수)/3점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

5. 가독성 및 구성: (점수)/2점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

6. 예측 및 기대감 조성: (점수)/5점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

7. 공감대 형성 및 심리적 연결: (점수)/4점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

8. 데이터 활용의 창의성: (점수)/3점
   - 긍정적 분석: [콘텐츠의 장점을 구체적으로 분석]
   - 부정적 분석: [콘텐츠의 단점을 구체적으로 분석]
   - 평가 원칙 적용 및 최종 근거: [분석된 장단점에 평가 원칙을 어떻게 적용했으며, 그 결과 왜 이 점수를 부여했는지 최종 결론 서술]

**총점: (점수)/32점**

**종합 의견:**
- 전체적 강점: [3가지]
- 주요 개선 영역: [3가지]
- 우선 개선 사항: [1-2가지]
"""
    return prompt


def extract_total_score(response_text):
    """응답에서 총점 추출"""
    pattern = r"\*\*총점:\s*(\d+(?:\.\d+)?)/32\*\*"
    match = re.search(pattern, response_text)

    if match:
        return int(match.group(1))

    # 대안 패턴들 시도
    patterns = [
        r"총점:\s*(\d+(?:\.\d+)?)/32",
        r"총점\s*(\d+(?:\.\d+)?)/32",
        r"총점:\s*(\d+(?:\.\d+)?)",
        r"총점\s*(\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text)
        if match:
            return int(match.group(1))

    print(f"점수 추출 실패: {response_text[:200]}...")
    return None


def evaluate_single_sample(sample_data, retry_count=3):
    """단일 샘플 평가"""
    prompt = create_evaluation_prompt(sample_data)

    for attempt in range(retry_count):
        try:
            response = model.generate_content(prompt)
            print("\n평가내용:\n", response.text)
            score = extract_total_score(response.text)

            if score is not None:
                return score, response.text
            else:
                print(f"점수 추출 실패 (시도 {attempt + 1}/{retry_count})")

        except Exception as e:
            print(f"API 호출 실패 (시도 {attempt + 1}/{retry_count}): {e}")
            time.sleep(2)  # 잠시 대기 후 재시도

    return None, None


# 파일명 생성 함수
def generate_filename(test_name):
    """날짜와 시간이 포함된 파일명 생성"""
    results_dir = "app-report/test/llm-judge/results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{test_name}_{timestamp}.txt"
    return os.path.join(results_dir, filename)


# TXT 파일로 결과 저장하는 함수
def save_results_to_txt(results, detailed_responses, filename, test_type):
    """결과를 TXT 파일로 저장"""
    with open(filename, "w", encoding="utf-8") as f:
        # 헤더 정보
        f.write(f"=== {test_type} 결과 ===\n")
        f.write(
            f"생성 날짜: {datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')}\n"
        )
        f.write("=" * 80 + "\n\n")

        # 각 평가의 상세 내용
        f.write("=== 평가 상세 내용 ===\n\n")
        for key, response_text in detailed_responses.items():
            f.write(f"[{key}]\n")
            f.write("-" * 60 + "\n")
            f.write(response_text)
            f.write("\n" + "=" * 80 + "\n\n")

        # 결과 요약 테이블
        f.write("=== 결과 요약 ===\n\n")
        df = pd.DataFrame(results).T
        f.write(df.to_string())
        f.write("\n\n")

        # 통계 정보
        f.write("=== 통계 정보 ===\n")
        for model_name in df.index:
            valid_scores = [score for score in df.loc[model_name] if score is not None]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                max_score = max(valid_scores)
                min_score = min(valid_scores)
                f.write(
                    f"{model_name}: 평균 {avg_score:.2f}점, 최고 {max_score}점, 최저 {min_score}점\n"
                )


def run_test_1():
    """테스트 1: 30개 후보 각각 1번씩 평가"""
    print("=== 테스트 1: 전체 후보 평가 시작 ===")

    results = {}
    detailed_responses = {}  # 추가: 상세 평가 내용 저장용

    for model_name, samples in SAMPLE_NEWS.items():
        print(f"\n{model_name.upper()} 모델 평가 중...")
        results[model_name] = {}

        for i, sample in enumerate(samples, 1):
            print(f"  후보{i} 평가 중... ({sample['id']})")

            score, full_response = evaluate_single_sample(sample)
            results[model_name][f"후보{i}"] = score

            # 추가: 상세 응답 저장
            key = f"{model_name}/후보{i}"
            detailed_responses[key] = full_response if full_response else "평가 실패"

            if score is None:
                print(f"    ❌ 평가 실패")
            else:
                print(f"    ✅ 점수: {score}/32\n")

            # API 호출 간격 조절
            time.sleep(1)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results).T

    print("\n=== 테스트 1 결과 ===")
    print(df)

    # 통계 정보 추가
    print(f"\n=== 모델별 평균 점수 ===")
    for model_name in df.index:
        valid_scores = [score for score in df.loc[model_name] if score is not None]
        if valid_scores:
            avg_score = sum(valid_scores) / len(valid_scores)
            print(f"{model_name}: {avg_score:.2f}점")

    # 수정: TXT 파일로 저장
    filename = generate_filename("test1_results")
    save_results_to_txt(
        results, detailed_responses, filename, "테스트 1: 전체 후보 평가"
    )
    print(f"\n결과가 '{filename}'에 저장되었습니다.")

    return df


def run_test_2():
    """테스트 2: 각 모델별 후보1에 대해 10회 반복 평가"""
    print("\n=== 테스트 2: 반복 평가 신뢰성 테스트 시작 ===")

    results = {}
    detailed_responses = {}  # 추가: 상세 평가 내용 저장용

    for model_name, samples in SAMPLE_NEWS.items():
        print(f"\n{model_name.upper()} 모델 평가 중...")

        # 후보 1만 테스트
        for candidate_num in range(1, 2):
            sample = samples[candidate_num - 1]  # 0-indexed
            row_key = f"{model_name}/후보{candidate_num}"
            results[row_key] = {}

            print(f"  {row_key} 평가 중... ({sample['id']})")

            # 10회 반복 평가
            for repeat in range(1, 11):
                print(f"    반복 {repeat}/10...")

                score, full_response = evaluate_single_sample(sample)
                results[row_key][f"반복{repeat}"] = score

                # 추가: 상세 응답 저장
                detail_key = f"{row_key}/반복{repeat}"
                detailed_responses[detail_key] = (
                    full_response if full_response else "평가 실패"
                )

                if score is None:
                    print(f"      ❌ 평가 실패")
                else:
                    print(f"      ✅ 점수: {score}/32")

                # API 호출 간격 조절
                time.sleep(1)

    # 결과를 DataFrame으로 변환
    df = pd.DataFrame(results).T

    print("\n=== 테스트 2 결과 ===")
    print(df)

    # 각 후보별 점수 분산 분석
    print(f"\n=== 반복 평가 분산 분석 ===")
    for row_key in df.index:
        scores = [score for score in df.loc[row_key] if score is not None]
        if len(scores) >= 2:
            mean_score = sum(scores) / len(scores)
            variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
            std_dev = variance**0.5
            print(f"{row_key}: 평균 {mean_score:.2f}점, 표준편차 {std_dev:.2f}")

    # 수정: TXT 파일로 저장
    filename = generate_filename("test2_results")
    save_results_to_txt(
        results, detailed_responses, filename, "테스트 2: 반복 평가 신뢰성 테스트"
    )
    print(f"\n결과가 '{filename}'에 저장되었습니다.")

    return df


def main():
    """메인 실행 함수"""
    print("매칭 뉴스 평가 시스템을 시작합니다...")

    # 사용자 선택
    print("\n실행할 테스트를 선택하세요:")
    print("1. 테스트 1: 전체 30개 후보 평가 (각 1회)")
    print("2. 테스트 2: 각 모델별 후보1-3 반복 평가 (각 3회)")
    print("3. 둘 다 실행")

    choice = input("선택 (1, 2, 3): ").strip()

    if choice in ["1", "3"]:
        test1_results = run_test_1()

    if choice in ["2", "3"]:
        test2_results = run_test_2()

    print("\n모든 테스트가 완료되었습니다!")


if __name__ == "__main__":
    main()
