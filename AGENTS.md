# 작업 원칙

- 보고서와 README는 ASD-STE100(Simplified Technical English)으로 쓴다. 긴 글보다 다이어그램, 그래프, 표로 설명하고, 처음 보는 사람도 읽을 수 있게 쓴다. 본문에 링크(밑줄 표시)를 넣지 않고 경로는 `code`로 적는다. `v3`, `stage7` 같은 내부 버전 번호를 본문에 쓰지 않는다.
- README에는 최신 결과만 둔다. 방법의 새 버전은 이전 버전을 대체하고, 두 버전을 한 표나 그래프에 함께 넣지 않는다. 이전 실험은 "Previous experiments" 표에 숫자 없이 둔다. 웹 설명은 `docs/explainer.html`에 둔다. 모델별(7B 등)로 나누고, 병합 강도마다 표를 따로 만든다. 표 제목에 병합한 토큰 수와 비율(예: 338개 병합, 46.4%)을 적는다.
- 기준선은 "No token merge"로 부른다. 방법 이름은 `ToMe`, `ToSA`, `Depth-only`, `Depth-RGB`를 쓴다. Depth-only도 VLM에는 RGB를 입력하며, 이름은 병합 기준만 뜻한다.
- 점수 이름을 섞지 않는다. VQAv2 counting은 VQA accuracy(주 지표)와 RCA(ToSA 논문 지표), GQA는 exact match를 쓴다.
- 실험 조건은 추론 전에 `experiments/*/protocol.json`에 고정하고 commit한다. 결과를 본 뒤 방법·임계값·prompt·seed를 바꾸지 않는다.
- 실행 불가(STRUCTURAL_INFEASIBLE)와 숫자 읽기 실패는 0점으로 분모에 남긴다. 다른 결과로 대체하지 않는다.
- 차이를 주장할 때는 같은 문항끼리 비교한 신뢰구간을 확인하고, 구간이 0을 포함하면 "차이 불확실"이라고 쓴다.
- 원시 답변(jsonl), 집계 CSV, 재생성 코드를 함께 남긴다.
- 결과 업로드 대상은 이 저장소다. 사용자 지시 없이 `depth-region-vlm` 원격으로 push하지 않는다.
