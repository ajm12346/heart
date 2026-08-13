# 데이터 안내

## 실제 사용 데이터

이 프로젝트는 UCI Machine Learning Repository의 **Heart Disease** 데이터셋에 포함된 네 기관의 processed 파일을 모두 사용합니다.

| 기관 | 파일 | 행 수 | 양성 | 결측 셀 |
|---|---|---:|---:|---:|
| Cleveland | `processed.cleveland.data` | 303 | 139 | 6 |
| Hungarian Institute of Cardiology | `processed.hungarian.data` | 294 | 106 | 782 |
| University Hospital Zurich | `processed.switzerland.data` | 123 | 115 | 273 |
| VA Long Beach | `processed.va.data` | 200 | 149 | 698 |

- 원문 출처: https://archive.ics.uci.edu/dataset/45/heart+disease
- DOI: https://doi.org/10.24432/C52P4X
- 라이선스: CC BY 4.0
- 전체 행: 920
- 목표값: 원래 `0`은 질환 없음, `1~4`는 질환 존재로 이진화
- 결합 파일: `processed/heart_disease_all_sites.csv`

네 기관의 변수 정의가 같아 모두 모델 개발에 사용했습니다. 기관별 결측과 유병률 차이가 커서 환자 단위 5겹 검증뿐 아니라 한 기관씩 완전히 제외하는 검증도 `model/evaluation.json`에 기록했습니다.

## 사용하지 않은 후보

CDC BRFSS 2023/2024는 40만 건이 넘는 대규모 공식 자료지만, 전화 설문에 의한 과거 진단 자가보고가 목표값이고 UCI의 임상검사 변수와 일치하지 않습니다. 서로 다른 목표와 변수를 억지로 합치면 결과 해석이 더 나빠지므로 현재 모델에는 혼합하지 않았습니다.

- CDC BRFSS: https://www.cdc.gov/brfss/annual_data/annual_data.htm
- `_MICHD`: 심근경색 또는 관상동맥질환을 보고한 응답자에 대한 계산변수

## 대체·합성 데이터

학습용 대체 또는 합성 데이터는 만들지 않았습니다. 공식 UCI 네 기관 원자료를 확보했기 때문입니다. `sample_input.csv`는 화면 기능을 시험하기 위한 한 건의 예시이며 모델 학습에는 사용되지 않습니다.

## 주의

자료는 1980년대 해외 환자집단에서 수집되었습니다. 국내 환자의 진단, 미래 절대위험 추정, 응급판단 또는 치료결정에 사용할 수 없습니다.
