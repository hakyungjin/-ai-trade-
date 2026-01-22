# 마이그레이션 가이드

## 📋 빠른 시작

### 기존 마이그레이션 적용
```bash
cd backend
alembic upgrade head
```

---

## 🔄 자동 마이그레이션 (권장)

### 1️⃣ 모델 변경 후 자동 마이그레이션 생성
```bash
cd backend

# 1. 모델 파일 수정 (예: backend/app/models/market_data.py)
# 2. 자동 마이그레이션 파일 생성
alembic revision --autogenerate -m "추가 기능 설명"

# 예시:
# alembic revision --autogenerate -m "add_price_history_index"
# alembic revision --autogenerate -m "optimize_market_candles_table"
```

### 2️⃣ 마이그레이션 검토
```bash
# 생성된 파일 확인
# backend/alembic/versions/[timestamp]_[message].py
# 
# 파일을 열어서 upgrade/downgrade 함수가 올바른지 확인
```

### 3️⃣ 마이그레이션 적용
```bash
alembic upgrade head
```

---

## 🛠️ 수동 마이그레이션 (특수한 경우)

모델로 자동 감지가 안 되는 경우 (예: 복잡한 SQL, 인덱스):

```bash
# 빈 마이그레이션 파일 생성
alembic revision -m "add_custom_index"

# 생성된 파일 수동 편집
# backend/alembic/versions/[timestamp]_add_custom_index.py

# 적용
alembic upgrade head
```

---

## 📊 자주 사용하는 명령어

| 명령어 | 설명 |
|--------|------|
| `alembic upgrade head` | 모든 미적용 마이그레이션 적용 |
| `alembic downgrade -1` | 가장 최근 마이그레이션 취소 |
| `alembic current` | 현재 적용된 마이그레이션 확인 |
| `alembic history` | 모든 마이그레이션 이력 |
| `alembic revision --autogenerate -m "메시지"` | 자동 마이그레이션 생성 |

---

## ✅ 마이그레이션 적용 이력

### 현재 적용된 마이그레이션:
1. ✅ `2a5a9ea01389` - Initial migration (2024)
2. ✅ `7d936f1f64d7` - Add AI training data models
3. ✅ `add_vector_patterns_weights` - Add vector patterns and strategy weights
4. ✅ `optimize_candle_v1` - Optimize candle indexes (MySQL compatible)
5. ✅ `add_coin_metadata_v1` - Add coin metadata tables

### 각 마이그레이션이 추가하는 것:

**2a5a9ea01389 - Initial Migration**
- `market_candles` - 캔들 데이터 (OHLCV)
- `technical_indicators` - 기술적 지표
- `ai_analyses` - AI 분석 결과
- `signal_history` - 신호 이력
- `trades` - 거래 기록

**7d936f1f64d7 - AI Training Data**
- `ai_training_data` - AI 모델 학습 데이터

**add_vector_patterns_weights**
- `vector_patterns` - 벡터 패턴 저장
- `vector_similarities` - 유사 패턴 캐시
- `strategy_weights` - 전략 가중치 설정

**optimize_candle_v1 - 인덱싱 최적화**
- `market_candles` 테이블에 인덱스 추가:
  - `idx_candle_symbol` - 심볼 검색 빠르게
  - `idx_candle_timeframe` - 타임프레임 검색
  - `idx_candle_time_desc` - 최신 캔들 조회
  - `idx_candle_symbol_tf_time_desc` - 심볼+타임프레임+시간 복합 인덱스
  - UNIQUE 제약 추가 (중복 방지)

**add_coin_metadata_v1 - 코인 메타데이터**
- `coins` - 모니터링 코인 정보
- `coin_statistics` - 코인별 통계
- `coin_analysis_configs` - 코인 분석 설정
- `coin_price_history` - 코인 가격 이력

---

## 🚀 데이터베이스 초기화 (개발 환경)

```bash
# 경고: 모든 데이터가 삭제됩니다!
# MySQL CLI에서:
mysql -u root -p
DROP DATABASE crypto_trader;
CREATE DATABASE crypto_trader;
EXIT;

# 마이그레이션 다시 적용:
cd backend
alembic upgrade head
```

---

## 🔍 문제 해결

### Q: "table already exists" 에러
```bash
# 마이그레이션 상태 확인
alembic current

# 이미 적용된 마이그레이션이면 DB에서 직접 삭제:
mysql -u root -p
USE crypto_trader;
DELETE FROM alembic_version WHERE version_num = '2a5a9ea01389';
EXIT;

# 다시 적용
alembic upgrade head
```

### Q: "IF NOT EXISTS" 구문 에러
- Alembic에서 `if_not_exists=True` 파라미터 사용 금지
- MySQL은 이 문법을 지원하지 않음
- 대신 try/except로 처리

### Q: 자동 생성이 안 됨
```bash
# 1. models/__init__.py에 모델 임포트 확인
# 2. alembic/env.py의 target_metadata 설정 확인
# 3. 모델의 __tablename__ 설정 확인
```

---

## 💡 권장 워크플로우

### 개발 중
1. 모델 파일 수정
2. `alembic revision --autogenerate -m "설명"`
3. 생성된 파일 검토
4. 테스트 DB에서 테스트
5. `alembic upgrade head`

### 프로덕션 배포
1. 마이그레이션 백업: `alembic history > migration_history.txt`
2. DB 백업
3. `alembic upgrade head`
4. 검증

---

## 📚 참고 자료

- [Alembic 공식 문서](https://alembic.sqlalchemy.org/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/en/20/orm/)
- [MySQL 인덱싱](https://dev.mysql.com/doc/refman/8.0/en/optimization-indexes.html)

---

## 🎯 다음 마이그레이션 체크리스트

새로운 마이그레이션을 만들 때:

- [ ] 모델 파일 수정 (models/*.py)
- [ ] models/__init__.py에 import 추가
- [ ] `alembic revision --autogenerate -m "설명"`
- [ ] 생성된 파일 검토 (불필요한 부분 제거)
- [ ] MySQL 호환성 확인 (IF NOT EXISTS 등)
- [ ] 개발 DB에서 테스트: `alembic upgrade head`
- [ ] 문제 없으면 커밋

---

## 🔄 현재 마이그레이션 상태

```bash
# 현재 상태 확인
alembic current

# 모든 마이그레이션 확인
alembic history --verbose
```

마이그레이션이 성공적으로 적용되면 다음 메시지를 봅니다:
```
INFO  [alembic.runtime.migration] Context impl MySQLImpl.
INFO  [alembic.runtime.migration] Will assume non-transactional DDL.
INFO  [alembic.runtime.migration] Running upgrade ... head
```
