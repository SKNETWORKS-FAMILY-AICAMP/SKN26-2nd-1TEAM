# 개발 규칙
## branch strategy
본 프로젝트는 **Organization에서 fork해서 개인 repo에서 push후 PR요청을 보내는 방식으로 github를 관리**합니다.

### Organization Repository
```text
main        →   배포 가능 상태     (Protected)
develop     →   통합 개발 브런치   (Pull Requests 타깃 브랜치)
```

### Personal Fork Repository
```text
main        →   기능 통합 브런치   (Pull Requests 대상 브랜치)
feature/*   →   기능 개발 브랜치
fix/*       →   버그 수정 브랜치
```

---
## Commit Convention
### 기본 원칙
* **헤더(첫 줄)는 영어로 작성**한다.
* **본문(body)은 한글로 작성**한다. (변경 이유/영향/주의사항)
* 모든 커밋은 이슈와 반드시 연결한다.
> 커밋 본문 마지막 줄에 `Refs: #이슈번호`를 무조건 포함한다.

### Type
#### feat
새로운 기능을 추가할 때 사용한다.
- 새로운 모델 추가
- Streamlit 페이지 추가
- Docker 서비스 추가
---
#### fix
버그 수정 시 사용한다.

- 예측 오류 수정
- 코드 오류 수정
---
#### docs
문서 변경 시 사용한다.
- README 수정
- 코드 주석 보강
---
#### refactor
기능 변경 없이 코드 구조를 개선할 때 사용한다.
- 함수 분리
- 파일 구조 재정리
- 중복 코드 제거
---
#### chore
환경 설정 또는 빌드 관련 변경 시 사용한다.
- .gitignore 수정
- requirements 수정
- 폴더 구조 생성
---
#### test
테스트 코드 추가 또는 수정 시 사용한다.
- pytest 추가
- 단위 테스트 작성
- 테스트 커버리지 개선
#### Type 사용 기준 요약
| 상황               | Type      |
|:------------------:|:---------:|
| 새로운 기능 추가   | feat         |
| 오류 수정       | fix          |
| 문서 변경       | docs         |
| 코드 구조 개선    | refactor     |
| 환경/설정 작업    | chore        |
| 테스트 코드 작업   | test         |

### Sample Commit Convention
```git
    <type>(scope): <subject>

    - 변경 내용 설명
    - 왜 변경했는지
    - 영향 범위 또는 주의사항

    Refs: #123
```
---
## Development Workflow

### Issue 생성 (Org Repo)
* 모든 작업은 Issue 기반으로 시작한다.
* 작업 전 반드시 Issue를 생성해서 시작한다.
---
### Branch Naming Rule

- feature/#이슈번호-간단설명
- fix/#이슈번호-간단설명

```text
예시
- feature/#12-add-model
- fix/#18-prediction-bug
```

---

### Commit
```git
<type>(scope): <subject>

- 변경 내용 설명
- 왜 변경했는지
- 영향 범위 또는 주의사항

Refs: #123
```
### Pull Request 규칙

- PR 대상 브랜치는 항상 `Organization/develop` 이다.
- `main` 브랜치로 직접 PR을 보내지 않는다.

### Merge 기준

- 최소 1회 이상 리뷰 후 병합한다.
- 기능 테스트가 완료된 경우에만 병합한다.
- 충돌 해결 후 최신 상태로 유지한다.

---

## Project Structure
### Root Directory
- .gitignore : Git에서 추적하지 않을 파일 정의
- LICENSE : 프로젝트 라이선스 파일
- structure.txt : 현재 폴더 구조 기록 파일
- README.md: 프로젝트 설명문
---
### app
Streamlit 기반 UI 레이어
- common/ : 공통 UI 컴포넌트 및 유틸리티
- pages/ : 멀티 페이지 구성 영역
※ 향후 main.py가 서비스 진입점 역할 수행
---
### artifacts
학습 및 실행 결과 저장 영역
- logs/ : 학습 및 실행 로그
- metrics/ : 모델 평가 지표 결과
- models/ : 학습 완료 모델 파일
※ 추후 MLflow 도입 시 확장 가능
---
### config
환경 설정 및 하이퍼파라미터 설정 파일 위치
---
### Database
데이터베이스 관련 파일
- 초기 스키마
- SQL 스크립트
- 마이그레이션 파일
※ Docker Compose 도입 시 초기화 스크립트 포함 가능
---
### notebooks
EDA 및 실험용 Jupyter Notebook 저장
운영 코드와 실험 코드 분리 목적
**개인 영문명으로 폴더 생성 후 안에서 개발**
---
### src
핵심 비즈니스 로직 및 ML 코드
- data/ : 데이터 로딩 및 전처리 로직
- models/ : 모델 학습 및 예측 로직
※ 향후 experiments/, utils/ 등 확장 가능
---
### 설계 원칙
- UI(app)와 ML 로직(src) 분리
- 실험 코드(notebooks)와 운영 코드 분리
- 결과물(artifacts) 분리로 재현성 확보
- Docker 및 MLflow 확장 가능 구조 유지
- Phase0는 구조 설계 중심, Phase1부터 기능 확장