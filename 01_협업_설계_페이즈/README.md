# 산업 표면 결함 이미지 분류 포트폴리오 — 협업 설계

이 폴더는 사람 또는 다른 AI가 같은 기준으로 프로젝트를 이어서 만들기 위한 설계 원본이다. 구현 전에 `00`부터 읽고, 각 Phase의 완료 게이트를 통과한 뒤 다음 단계로 진행한다.

## 인수인계 현황 (2026-11-14)

- **LLM/인수자 진입점:** 상위 `AGENTS.md`를 먼저 읽는다. 이 문서는 설계 원본의 입구이고, `AGENTS.md`는 최소 규칙·현재 상태·실행 명령을 요약한다.
- **최신 작업 기록:** `08_ABC_교차검증_및_구현.md` — Ouroboros 인터뷰 스킬 Path B로 전문가 A(기술)·B(포트폴리오)·C(재현성/QA)를 교차 검증하고, 발견 결함을 수정한 기록.
- **현재 코드 상태:** 전체 pytest **128 passed**(수정 전 90개 수집·89 통과·1 실패), `compileall` 성공, `pip check` 정상, `pip install --dry-run -r requirements.txt` 해결 성공. CUDA 실데이터 reliability 평가는 별도 최신 완료 기록을 따른다.
- **이번에 고친 것:** OpenCV 후보 마스크가 1픽셀 얇은 스크래치를 지우는 결함, 테스트 수 83→110 문서 불일치, numpy 2.5.2↔opencv 4.12 의존성 충돌, RAR/ZIP/빈 입력 실패 시 빈 산출물이 남는 문제, train CLI 조기 검증.
- **추가한 것:** `evaluate`의 신뢰도 블록(ECE/MCE·Brier·NLL·top-3·risk-coverage·bootstrap CI), `docs/reliability-evaluation.md`, `tests/test_reliability.py`, 읽기 전용 인수인계 점검 `scripts/handoff_status.py`.
- **완료된 잔여 작업:** `baseline · seed 42`의 현재 OpenCV 패널 화면을 실제 `Cr_1.bmp` 업로드 후 캡처했고, 경로·SHA-256을 `docs/demo-capture-evidence.md`와 상태 점검기로 교차 검증했다. legacy PNG는 보관본으로 유지한다. 상위 폴더명은 현재 `steel-surface-defect-classification`으로 기준과 일치한다.
- **다른 LLM이 이어서 할 때:** 먼저 이 문서 → `04_decision_ledger.md`(확정 결정) → `05_verification_checklist.md`(완료/미완료) → `08`(근거와 한계) → `09`(잔여 작업 판정) 순으로 읽는다. 그 다음 `python scripts/handoff_status.py`로 현재 상태를 읽기 전용으로 확인하고 `02_구현/README.md`의 실행 명령을 사용한다.
- **최신 완료:** 공식 NEU-CLS로 재생성한 manifest와 RTX 3070 baseline seed=42 checkpoint에서 reliability 실측을 완료했다(ECE 0.003993, Brier 0.002065, NLL 0.004671). 고정 test split 진단의 해석 경계는 유지한다.
- **주의:** 기존 `1.0000`·지연시간·OOD 표는 이 체크아웃에 산출물이 없어 재측정하지 않았다. `08`의 `[Owner-states]` 표기를 확인하고, 새 수치를 만들기 전에는 문서에 인용하지 않는다.

## 현재 목표

공개 산업 표면 결함 이미지 데이터로 PyTorch 분류 파이프라인을 구현하고, 데이터 증강 전후의 품질과 단일 이미지 추론 시간을 동일 조건에서 비교한다. Streamlit 화면에서 이미지 업로드 결과를 확인할 수 있게 한다.

## 중요한 표현 원칙

- 실제 사용하는 데이터가 반도체 데이터가 아니면 프로젝트 이름을 **산업 표면 결함 이미지 분류**로 사용한다.
- 공개 데이터 기반 실험을 실제 생산라인, 검사 장비, 실시간 품질 보증 성과로 표현하지 않는다.
- 측정하지 않은 Accuracy, F1, 추론 시간, GPU/NPU 최적화 수치를 쓰지 않는다.

## 문서 순서

1. `00_project_brief.md` — 범위와 결정이 필요한 항목
2. `01_cross_validation.md` — 기술·포트폴리오 관점의 독립 검토 결과
3. `02_phase_plan.md` — 구현 순서와 완료 게이트
4. `03_collaboration_protocol.md` — 사람/AI 협업 규칙
5. `04_decision_ledger.md` — 확정·보류 결정을 기록하는 원장
6. `05_verification_checklist.md` — 공개·지원 전 최종 점검
7. `06_전문가_인터뷰_기록.md` — 2026-09-09 전문가 교차 검증 기록 (사실 검증·사용자 결정 대기 항목 포함)
8. `07_현재_교차검증_정리.md` — 2026-09-09 시점의 관점별 정리와 2026-11-14 후속 요약
9. `08_ABC_교차검증_및_구현.md` — 2026-11-14 전문가 A·B·C 교차 검증, 발견 결함 수정, 테스트 110개 검증 기록
10. `09_잔여작업_ABC_교차검증.md` — 당시 잔여 작업 재검증·인수인계 상태 도구·128개 테스트의 최신 완료 갱신

참고 문서는 상위 `references/ai-rag-project-docs`에 복사되어 있다. 참고 문서의 지시문이 이 프로젝트의 요구사항을 대체하지는 않는다.
