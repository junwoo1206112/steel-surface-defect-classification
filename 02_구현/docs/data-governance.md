# 00. 데이터 거버넌스 (data-governance)

> 상태: Phase 1 수신·검증 완료 (2026-09-09)

## 데이터셋 식별

| 항목 | 내용 |
| --- | --- |
| 데이터셋 명칭 | NEU-CLS (NEU surface defect database, classification 이미지 세트) |
| 원저자/제공 기관 | Northeastern University (중국 동북대학), Kechen Song · Yunhui Yan |
| 공식 출처 페이지 | `http://faculty.neu.edu.cn/songkechen/zh_CN/zdylm/263270/list/index.htm` (영문: `.../songkc/en/zdylm/263265/list/index.htm`) |
| 데이터 구성 | 흑백 BMP 1,800장, 200×200, 6클래스 × 300장 |
| 클래스 | crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches |
| 파일명 형식 | 공식 배포본은 약어 접두사: `Cr/In/Pa/PS/RS/Sc_<번호>.bmp` (예: `Cr_1.bmp`, `PS_88.bmp`) |
| 도메인 | 열간 압연 강재(hot-rolled steel strip) 표면 — **반도체 데이터가 아니다** |
| 인용 논문 | K. Song and Y. Yan, "A noise robust method based on completed local binary patterns for hot-rolled steel strip surface defects," Applied Surface Science, vol. 285, pp. 858-864, 2013. |

## 이용 조건

- 학술 연구 목적으로 공개된 데이터셋이며, **상업적 이용 불가**, 사용 시 원저자 논문 인용 요구.
- 본 프로젝트는 비영리 개인 포트폴리오·학습 목적이므로 이용 조건에 부합한다.
- 원본 이미지와 압축 파일을 **Git 저장소에 포함하거나 재배포하지 않는다** (`data/raw/`는 `.gitignore` 처리).

## 획득 경로 (확정: 사용자 승인 2026-09-09)

- 획득 방법: 사용자가 브라우저로 공식 페이지에서 NEU-CLS를 다운로드 (RAR, `NEU-CLS.rar`)
- 무인 자동 다운로드가 불가능한 사실 관계:
  - 공식 페이지는 안티봇 JS 리디렉션 게이트가 있어 스크립트 접근이 차단됨(2026-09-09 실측 2회).
  - 구형 Google Drive 미러(`0B5OUtBsSxu1Bdjh4dk1SeGYtNFU`)는 Google 로그인을 요구함(실측).
  - Kaggle 미러는 API 토큰이 필요하며, 사전 분할 버전이라 자체 분할 원칙과 충돌 가능.
- 이에 따라 획득은 사람(사용자)이 수행하고, 검증·구조화는 스크립트가 자동 수행한다.

## 수신 후 검증 결과 (2026-09-09 측정)

| 항목 | 기대값 | 실측값 |
| --- | --- | --- |
| 파일명 | `<class>_<n>` 형식 | 약어 접두사 `Cr/In/Pa/PS/RS/Sc_<n>.bmp` — 코드에 공식 매핑 명시 |
| 총 이미지 수 | 1,800 | 1,800 (전부 정상 디코딩) |
| 클래스별 수 | 6클래스 × 300 | 6클래스 × 300 정확히 일치 |
| 이미지 크기 | 200×200 | 200×200 (1,800장 전부) |
| 손상/디코딩 실패 | 0건 | 0건 |
| 중복(SHA-256) | 확인 필요 | **1그룹 발견: `Pa_101.bmp` = `Pa_105.bmp`** |
| 압축파일 SHA-256 | - | `A85BCC6BFA1A40E76F28079E135694AAF5F6A4EE01203F7EAACFF0EBAFAA96F8` |
| 수신 날짜 | - | 2026-09-09 |

### 중복 처리 결정 (누수 방지)

- 원본 배포본에 완전 동일 파일 1쌍이 존재함(SHA-256 `6833002b…65af2`).
- 초기 분할에서 `Pa_101`(train)과 `Pa_105`(val)가 서로 다른 split에 배정되어 **train-val 누수**가 실제로 발생함을 확인했다.
- 따라서 **SHA-256 동일 파일은 정렬 순 첫 파일만 유지하고 분할 전에 제외**한다. 결과: 사용 이미지 1,799장(patches 299장).
- 배제 내역은 `data/processed/data-quality.json`의 `exact_duplicate_handling`에 기록되어 있다.
- 추가로 perceptual hash(32px DCT, 8×8 저주파, Hamming distance ≤2) 후보를 `near_duplicate_review`에 기록한다. 이는 검토 신호이며, 표면 텍스처가 비슷한 독립 이미지를 자동 배제하지 않는다.

## 분할 원칙 (확정)

- 클래스 비율 유지(stratified) 70/15/15, seed=42, `manifest.csv`로 고정.
- 실측 분할 결과: train 1,259 / val 270 / test 270 (클래스별 210/45/45, patches만 209/45/45).
- test set은 모델 선택·증강 선택에 사용하지 않는다(학습·검증·선택은 train/val만 사용).
- 분할 결과는 `data/processed/manifest.csv`, `data-quality.json`으로 Git에 포함(원본 이미지 제외).
- 재현 명령: `python scripts/prepare_data.py --input <NEU-CLS 압축파일>` (RAR은 7-Zip 필요)
