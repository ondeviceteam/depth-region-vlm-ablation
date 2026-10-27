# Depth Region VLM 실험

VLM 비전 인코더 안에서 시각 토큰을 합칠 때, **깊이(depth) 영역 정보**를 쓰면 정확도를 덜 잃는지 비교하는 실험입니다.

> **현재 상태: 실험 진행 중.** 아래 조건은 추론 전에 [protocol](experiments/2026-10-28_7b_vqav2_gqa/protocol.json)에 고정했습니다. 결과는 실행이 끝나면 이 문서에 추가합니다.

## 실험 조건

| 항목 | 내용 |
|---|---|
| 모델 | LLaVA-OneVision-7B ([ToSA 공식 저장소](https://github.com/hsiangwei0903/ToSA)의 실행 경로, 384×384 이미지 한 장) |
| 벤치마크 | VQAv2 counting 1,000문항 (새로 선정), GQA testdev-balanced 12,578문항 |
| 방법 | Token merge 없음, ToMe, ToSA, Depth-only, Depth-RGB |
| 병합 강도 | 729개 중 156개(21.4%), 338개(46.4%), 520개(71.3%), 650개(89.2%) 병합 |
| 반복 | VQAv2에서 338개 병합 조건과 Token merge 없음을 3회 실행 |
| 점수 | VQAv2: VQA accuracy, RCA · GQA: exact match |

코드 저장소: [depth-region-vlm](https://github.com/ondeviceteam/depth-region-vlm) (commit [`efe0a51`](https://github.com/ondeviceteam/depth-region-vlm/commit/efe0a516ffa0abe8f4947f81974aa7a90e8ab143))
