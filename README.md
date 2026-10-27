# Depth Region VLM 실험

VLM 비전 인코더 안에서 시각 토큰을 합칠 때 **깊이(depth) 영역 정보**를 쓰면 정확도를 덜 잃는지 비교한 실험입니다. 실험 조건은 추론 전에 [protocol](experiments/2026-10-28_7b_vqav2_gqa/protocol.json)에 고정했고, 결과를 본 뒤 바꾸지 않았습니다.

## 요약

같은 문항끼리 비교한 95% 신뢰구간(paired bootstrap)으로 판단했습니다.

- **VQAv2 counting, 156개·338개 병합:** Depth-only와 Depth-RGB가 ToSA보다 높습니다. 156개 병합에서는 VQA accuracy가 9~10%p 높습니다(신뢰구간 +6.7~+12.9%p). 338개 병합에서는 3~4%p 높습니다(Depth-RGB +3.21%p [+0.37, +6.19], Depth-only +4.22%p [+1.31, +7.26]).
- **VQAv2 counting, 520개·650개 병합:** ToSA가 가장 높습니다. 520개 병합에서 Depth-RGB는 ToSA보다 3.77%p 낮습니다 [−6.77, −0.73]. Depth-only와 ToSA의 차이(−2.49%p [−5.47, +0.52])는 불확실합니다. 650개 병합에서는 Depth 방법이 55%의 문항에서 실행 불가가 되어 30%p 넘게 낮습니다.
- **GQA:** 156개 병합에서만 Depth-only가 ToSA보다 0.65%p 높습니다 [+0.13, +1.18]. Depth-RGB와 ToSA의 차이(+0.44%p [−0.07, +0.96])는 불확실합니다. 338개 병합에서는 모든 방법의 차이가 불확실합니다. 520개 병합부터는 Depth 방법이 낮습니다.
- **ToSA와 ToMe:** VQAv2 counting에서는 모든 병합 강도에서 ToSA가 4~8%p 높습니다. GQA에서는 모든 병합 강도에서 차이가 불확실합니다.

## 실험 조건

| 항목 | 내용 |
|---|---|
| 모델 | LLaVA-OneVision-7B ([ToSA 공식 저장소](https://github.com/hsiangwei0903/ToSA)의 실행 경로, 384×384 이미지 한 장 → 시각 토큰 729개) |
| 벤치마크 | VQAv2 counting 1,000문항 (val에서 새로 선정), GQA testdev-balanced 전체 12,578문항 |
| 방법 | Token merge 없음, ToMe, ToSA, Depth-only, Depth-RGB |
| 병합 강도 | 729개 중 156개(21.4%), 338개(46.4%), 520개(71.3%), 650개(89.2%) 병합 |
| 점수 | VQAv2 counting: VQA accuracy(주 지표), RCA(ToSA 논문 지표) · GQA: exact match |

코드 저장소: [depth-region-vlm](https://github.com/ondeviceteam/depth-region-vlm) (commit [`efe0a51`](https://github.com/ondeviceteam/depth-region-vlm/commit/efe0a516ffa0abe8f4947f81974aa7a90e8ab143))

## 결과: LLaVA-OneVision-7B

- 각 표의 굵은 글씨는 병합 방법 가운데 가장 높은 점수입니다. *Token merge 없음*은 모든 표에 같은 값으로 넣은 기준선입니다.
- **실행 성공**은 Depth 방법이 목표 토큰 수까지 병합할 수 있었던 문항 수입니다. 보호 토큰과 깊이 영역의 수가 목표보다 많으면 실행 불가로 기록하고 0점을 줍니다.
- 3회 반복한 조건은 첫 번째 실행의 점수입니다.

### VQAv2 counting

- **VQA accuracy:** 사람 답변 10개와 비교하는 VQAv2 공식 점수입니다. 모든 문항이 대상입니다.
- **RCA:** 정답이 1 이상인 911문항에서 `1 − |예측 − 정답| / 정답`의 평균입니다. 숫자를 얼마나 가깝게 맞췄는지를 봅니다.

#### 156개 병합 (21.4% 병합, 729 → 573개 남음)

| 방법 | VQA accuracy (%) | RCA (%) | 실행 성공 |
|---|---:|---:|---:|
| *Token merge 없음* | *76.50* | *79.51* | 1000/1000 |
| ToMe | 56.18 | 61.22 | 1000/1000 |
| ToSA | 63.52 | 68.55 | 1000/1000 |
| Depth-only | **73.73** | **77.97** | 1000/1000 |
| Depth-RGB | 72.83 | 76.61 | 1000/1000 |

#### 338개 병합 (46.4% 병합, 729 → 391개 남음)

| 방법 | VQA accuracy (%) | RCA (%) | 실행 성공 |
|---|---:|---:|---:|
| *Token merge 없음* | *76.50* | *79.51* | 1000/1000 |
| ToMe | 54.59 | 58.63 | 1000/1000 |
| ToSA | 62.87 | 67.69 | 1000/1000 |
| Depth-only | **67.09** | 71.58 | 999/1000 |
| Depth-RGB | 66.08 | **73.44** | 999/1000 |

#### 520개 병합 (71.3% 병합, 729 → 209개 남음)

| 방법 | VQA accuracy (%) | RCA (%) | 실행 성공 |
|---|---:|---:|---:|
| *Token merge 없음* | *76.50* | *79.51* | 1000/1000 |
| ToMe | 55.13 | 59.71 | 1000/1000 |
| ToSA | **60.23** | **64.80** | 1000/1000 |
| Depth-only | 57.74 | 61.64 | 950/1000 |
| Depth-RGB | 56.46 | 61.63 | 950/1000 |

#### 650개 병합 (89.2% 병합, 729 → 79개 남음)

| 방법 | VQA accuracy (%) | RCA (%) | 실행 성공 |
|---|---:|---:|---:|
| *Token merge 없음* | *76.50* | *79.51* | 1000/1000 |
| ToMe | 52.86 | 59.53 | 1000/1000 |
| ToSA | **57.05** | **63.44** | 1000/1000 |
| Depth-only | 22.77 | 25.06 | 450/1000 |
| Depth-RGB | 20.92 | 23.67 | 450/1000 |

### GQA

**Exact match:** 답변의 첫 문장을 정규화한 값이 정답과 같은 문항의 비율입니다. 정규화는 소문자 변환, 문장부호·관사 제거, 숫자 단어 변환입니다.

#### 156개 병합 (21.4% 병합, 729 → 573개 남음)

| 방법 | Exact match (%) | 실행 성공 |
|---|---:|---:|
| *Token merge 없음* | *61.78* | 12578/12578 |
| ToMe | 60.22 | 12578/12578 |
| ToSA | 60.57 | 12578/12578 |
| Depth-only | **61.23** | 12578/12578 |
| Depth-RGB | 61.01 | 12578/12578 |

#### 338개 병합 (46.4% 병합, 729 → 391개 남음)

| 방법 | Exact match (%) | 실행 성공 |
|---|---:|---:|
| *Token merge 없음* | *61.78* | 12578/12578 |
| ToMe | 59.39 | 12578/12578 |
| ToSA | **59.48** | 12578/12578 |
| Depth-only | 59.33 | 12529/12578 |
| Depth-RGB | 59.34 | 12529/12578 |

#### 520개 병합 (71.3% 병합, 729 → 209개 남음)

| 방법 | Exact match (%) | 실행 성공 |
|---|---:|---:|
| *Token merge 없음* | *61.78* | 12578/12578 |
| ToMe | 57.93 | 12578/12578 |
| ToSA | **58.11** | 12578/12578 |
| Depth-only | 54.51 | 12121/12578 |
| Depth-RGB | 54.83 | 12121/12578 |

#### 650개 병합 (89.2% 병합, 729 → 79개 남음)

| 방법 | Exact match (%) | 실행 성공 |
|---|---:|---:|
| *Token merge 없음* | *61.78* | 12578/12578 |
| ToMe | 53.62 | 12578/12578 |
| ToSA | **53.90** | 12578/12578 |
| Depth-only | 23.78 | 5741/12578 |
| Depth-RGB | 23.94 | 5741/12578 |

### 실행 불가 문항을 뺀 비교 (사후 분석)

이 분석은 protocol에 없던 추가 분석입니다. 650개 병합에서 Depth 방법의 점수가 낮은 이유가 실행 불가 때문만인지 보려고, Depth 방법이 실행된 문항만 모아 모든 방법을 다시 채점했습니다.

| 벤치마크 · 병합 | 문항 수 | Token merge 없음 | ToMe | ToSA | Depth-only | Depth-RGB |
|---|---:|---:|---:|---:|---:|---:|
| VQAv2 counting · 520개 | 950 | 76.74 | 55.46 | 60.55 | **60.78** | 59.43 |
| VQAv2 counting · 650개 | 450 | 75.42 | 48.49 | **53.29** | 50.60 | 46.49 |
| GQA · 520개 | 12,121 | 62.02 | 58.18 | **58.30** | 56.56 | 56.90 |
| GQA · 650개 | 5,741 | 62.65 | 55.09 | **55.46** | 52.10 | 52.45 |

실행된 문항만 보아도 650개 병합에서는 Depth 방법이 ToSA보다 낮습니다. 실행 불가만으로는 낮은 점수를 다 설명할 수 없습니다.

### 반복 실행

VQAv2 counting에서 Token merge 없음과 338개 병합 조건은 3회씩 실행했습니다.

| 방법 | VQA accuracy 3회 (%) | 3회 답변이 모두 같은 문항 |
|---|---|---:|
| Token merge 없음 | 76.50 / 76.50 / 76.50 | 100.0% |
| ToMe | 54.59 / 55.58 / 54.16 | 81.7% |
| ToSA | 62.87 / 62.39 / 62.75 | 85.2% |
| Depth-only | 67.09 / 67.09 / 67.09 | 100.0% |
| Depth-RGB | 66.08 / 66.08 / 66.08 | 100.0% |

ToMe와 ToSA는 입력과 seed가 같아도 실행마다 답이 달라집니다. 3회 사이의 점수 폭은 각각 1.42%p와 0.48%p입니다. 원인은 아직 확인하지 않았습니다. 338개 병합에서 Depth 방법과 ToSA의 차이(3~4%p)는 이 폭보다 큽니다.

## 방법

다섯 방법 모두 ToSA 공식 코드의 같은 ToMe 블록을 거칩니다. 비전 인코더의 26개 층마다 `r`개씩 토큰을 합치며, `r = int((384·384/14²)·(1 − keep)/27)`는 공식 코드의 식을 그대로 썼습니다. 병합한 토큰은 복원하지 않고 LLM에 넘깁니다.

| 방법 | 어떤 토큰을 합치는가 |
|---|---|
| Token merge 없음 | 합치지 않음. 같은 계산 경로에 `r = 0`을 넣었습니다. |
| ToMe | 공식 ToMe. attention key가 비슷한 토큰끼리 합칩니다. |
| ToSA | 공식 ToSA. key 유사도에 깊이와 위치를 섞으며, 깊이 비중은 층마다 커집니다. |
| Depth-only | 깊이 영역과 위치가 비슷한 토큰끼리, 같은 연결 영역 안에서만 합칩니다. 경계 토큰은 보호합니다. |
| Depth-RGB | Depth-only와 같은 제약에서, 깊이·위치 점수와 key 유사도를 0.5씩 섞어 합칩니다. |

- Depth-only도 VLM에는 RGB 이미지를 그대로 넣습니다. 이름은 병합 기준만 뜻합니다.
- 깊이는 Depth Anything V2 Small로 미리 만들었습니다.
- 답변은 greedy decoding으로 생성했습니다. prompt는 공식 `run_vqa_3d_tome.py`의 것을 그대로 썼습니다.

## 평가 데이터

- **VQAv2 counting (`datasets/vqav2_count1000.json`):**
  - val의 "how many" 숫자 문항 가운데 정답이 0 이상의 정수인 것을 골랐습니다.
  - 이미지마다 한 문항씩, seed 20261028로 1,000문항을 뽑았습니다. 정답이 0인 문항이 89개, 1 이상인 문항이 911개입니다.
  - 이전 실험에서 쓴 이미지 250장(`datasets/excluded_coco_val2014_images.json`)은 뺐습니다.
- **GQA (`datasets/gqa_testdev_balanced.json`):**
  - testdev-balanced 전체 12,578문항, 이미지 398장입니다.
  - 문항 유형별 점수는 `results/2026-10-28_7b_vqav2_gqa/gqa_by_type.csv`에 있습니다.

## 파일과 재현

| 경로 | 내용 |
|---|---|
| `experiments/2026-10-28_7b_vqav2_gqa/` | 추론 전에 고정한 protocol과 ToSA 공식 코드에 적용한 호환 패치 |
| `results/2026-10-28_7b_vqav2_gqa/raw/` | 문항별 원시 답변(jsonl), 실행 환경, 로그 |
| `results/2026-10-28_7b_vqav2_gqa/*.csv` | 점수 표, 같은 문항끼리 비교한 차이와 신뢰구간, 반복 실행, GQA 유형별 점수, 사후 분석 |
| `scripts/` | 데이터 준비, 깊이 생성, 추론, 채점 코드 |

원시 답변만으로 모든 표를 다시 만들 수 있습니다.

```bash
python scripts/summarize.py results/2026-10-28_7b_vqav2_gqa      # 채점과 CSV 생성 (bootstrap 때문에 메모리가 넉넉해야 합니다)
python scripts/feasible_subset.py results/2026-10-28_7b_vqav2_gqa
python scripts/readme_tables.py results/2026-10-28_7b_vqav2_gqa  # README 결과 표 출력
python -m pytest scripts/test_counting_rca.py
```

추론은 PBS job 안에서 `scripts/launch_eval.sh JOB RUNTIME DATA OUTPUT dataset:variant:shard:num_shards:gpu ...`로 실행했습니다. 깊이는 `scripts/launch_depth.sh`로 만들었습니다.
