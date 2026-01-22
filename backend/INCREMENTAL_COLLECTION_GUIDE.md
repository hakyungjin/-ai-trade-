# 증분 데이터 수집 시스템 (Incremental Data Collection)

## 📋 개요

**이전 방식의 문제:**
- 매번 차트 데이터를 Binance API에서 새로 조회
- 불필요한 API 호출 반복 (같은 데이터 재조회)
- API 레이트 리밋에 빠르게 도달
- 느린 응답 속도

**새로운 방식의 이점:**
✅ DB에 캐시된 데이터 우선 사용
✅ DB에 없는 데이터만 선택적으로 수집 (증분)
✅ API 호출 최소화
✅ 빠른 응답 속도
✅ 데이터 지속성 보장

---

## 🏗️ 아키텍처

```
┌─────────────────────┐
│  API 요청 (분석)     │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────┐
│ UnifiedDataService                   │
│ (DB 캐시 + 증분 수집 통합)           │
└──┬──────────────────────────────┬───┘
   │                              │
   ▼                              ▼
┌────────────────────┐  ┌──────────────────────┐
│ MarketCandle DB    │  │ IncrementalCollector │
│ (캐시된 데이터)     │  │ (새 데이터만 수집)    │
└────────────────────┘  └──┬───────────────────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Binance API  │
                    └──────────────┘
```

---

## 📚 주요 서비스

### 1. **IncrementalDataCollector**
위치: `app/services/incremental_collector.py`

마지막 저장된 시간 이후의 데이터만 수집합니다.

```python
# 사용 예
from app.services.incremental_collector import IncrementalDataCollector

collector = IncrementalDataCollector(db, binance_service)

# 증분 수집
success, saved_count = await collector.collect_incremental_data(
    symbol="BTCUSDT",
    timeframe="1h",
    force_full=False  # False: 증분, True: 전체 재수집
)

# 데이터 커버리지 확인
coverage = await collector.get_data_coverage("BTCUSDT", "1h")
print(f"Coverage: {coverage['coverage_percent']}%")
```

**주요 메서드:**
- `collect_incremental_data()`: 증분 수집
- `get_last_saved_time()`: 마지막 저장 시간 조회
- `get_data_coverage()`: 데이터 커버리지 정보
- `sync_all_data()`: 모든 심볼 동기화

---

### 2. **UnifiedDataService**
위치: `app/services/unified_data_service.py`

DB 캐시와 증분 수집을 통합하여 항상 최신 데이터를 제공합니다.

```python
# 사용 예
from app.services.unified_data_service import UnifiedDataService

service = UnifiedDataService(db, binance_service)

# 캐시 + 증분 수집으로 캔들 데이터 조회
candles = await service.get_klines_with_cache(
    symbol="BTCUSDT",
    timeframe="1h",
    limit=100
)

# AI 분석용 통합 데이터 (기술적 지표 포함)
market_data = await service.get_market_data_for_analysis(
    symbol="BTCUSDT",
    timeframe="1h"
)
```

**주요 메서드:**
- `get_klines_with_cache()`: 캐시 + 증분으로 캔들 조회 (가장 자주 사용)
- `get_market_data_for_analysis()`: AI 분석용 전체 데이터
- `_get_candles_from_db()`: DB에서만 조회
- `_save_klines()`: 캔들 저장

---

## 🔌 API 엔드포인트

### 데이터 수집 API
위치: `app/api/data.py`

#### 1. 단일 심볼 동기화
```bash
POST /api/v1/data/sync/{symbol}?timeframe=1h

# 예시
curl -X POST "http://localhost:8000/api/v1/data/sync/BTCUSDT?timeframe=1h"

# 응답
{
    "success": true,
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "saved_candles": 24,  # 새로 저장된 캔들 개수
    "coverage": {
        "symbol": "BTCUSDT",
        "timeframe": "1h",
        "first_time": "2025-01-01T00:00:00",
        "last_time": "2026-01-22T16:00:00",
        "total_candles": 8760,
        "coverage_percent": 100.0
    }
}
```

#### 2. 모든 심볼 동기화
```bash
POST /api/v1/data/sync-all

# Request Body
{
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "timeframes": ["1h", "4h", "1d"]
}

# 응답
{
    "success": true,
    "results": {
        "BTCUSDT": {
            "1h": 24,    # 저장된 캔들 개수
            "4h": 6,
            "1d": 1
        },
        "ETHUSDT": {...}
    }
}
```

#### 3. 데이터 커버리지 확인
```bash
GET /api/v1/data/coverage/BTCUSDT?timeframe=1h

# 응답
{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "first_time": "2025-01-01T00:00:00",
    "last_time": "2026-01-22T16:00:00",
    "total_candles": 8760,
    "expected_candles": 8760,
    "coverage_percent": 100.0,
    "gap_hours": 8760
}
```

#### 4. 마지막 저장 시간 확인
```bash
POST /api/v1/data/check-last-saved?symbol=BTCUSDT&timeframe=1h

# 응답
{
    "symbol": "BTCUSDT",
    "timeframe": "1h",
    "last_saved_time": "2026-01-22T16:00:00",
    "next_candle_time": "2026-01-22T17:00:00",
    "minutes_until_next": 44
}
```

---

## 🔄 데이터 흐름

### 첫 번째 요청 (DB 비어있을 때)
```
요청: GET /api/v1/data/sync/BTCUSDT?timeframe=1h
    ↓
마지막 저장 시간 확인 → None
    ↓
Binance에서 전체 데이터 수집 (최대 1000개)
    ↓
DB에 저장
    ↓
클라이언트에 반환
```

### 두 번째 요청 (1시간 후)
```
요청: GET /api/v1/data/sync/BTCUSDT?timeframe=1h
    ↓
마지막 저장 시간 확인 → 2026-01-22T16:00:00
    ↓
Binance에서 새 데이터만 수집
(timestamp > 2026-01-22T16:00:00 인 데이터)
    ↓
필터링된 새 데이터만 DB에 저장
    ↓
클라이언트에 반환 (저장: 1개, 기존 데이터 제외)
```

---

## 💡 사용 예시

### 1. AI 분석에서 사용
```python
# ai_signal.py에서 이미 적용됨
from app.services.unified_data_service import UnifiedDataService

async def get_prediction(request: PredictionRequest, db: AsyncSession):
    unified_service = UnifiedDataService(db, binance)
    
    # 캐시 활용 + 필요시 증분 수집
    candles = await unified_service.get_klines_with_cache(
        symbol=request.symbol,
        timeframe=request.timeframe,
        limit=100
    )
    
    # 분석 진행
    prediction = await gemini_service.analyze_chart(
        symbol=request.symbol,
        candles=candles,
        current_price=current_price
    )
```

### 2. 정기 데이터 동기화 (스케줄러)
```python
# tasks.py 또는 백그라운드 작업
import asyncio
from app.services.incremental_collector import IncrementalDataCollector

async def schedule_data_sync(db, binance_service):
    """매 1시간마다 데이터 동기화"""
    collector = IncrementalDataCollector(db, binance_service)
    
    while True:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
        timeframes = ["1h", "4h", "1d"]
        
        results = await collector.sync_all_data(symbols, timeframes)
        print(f"Sync completed: {results}")
        
        await asyncio.sleep(3600)  # 1시간 대기
```

### 3. 수동 강제 재수집
```python
# 문제 발생 시 데이터 재수집
POST /api/v1/data/sync/BTCUSDT?timeframe=1h&force_full=true

# force_full=true: DB를 무시하고 처음부터 수집
```

---

## 📊 성능 비교

### 이전 방식
```
매 요청마다:
1. Binance API 호출 (100개 캔들) ⏱️ 2-5초
2. 데이터 처리
3. 응답

레이트 리밋: 1200 요청/분 (빠르게 소진)
응답 시간: 2-5초
```

### 새 방식
```
첫 요청:
1. DB 확인 (캐시 없음) ⏱️ <100ms
2. Binance API 호출 (필요시) ⏱️ 2-5초
3. DB 저장
4. 응답

이후 요청 (1시간 이내):
1. DB 확인 ⏱️ <10ms
2. Binance API 호출 (새 데이터만, 보통 1개) ⏱️ 500ms
3. DB 저장
4. 응답

응답 시간: <100ms (대부분 캐시에서 제공)
레이트 리밋: 거의 소비 안 함 (캐시 활용)
```

---

## 🛠️ 데이터 관리

### 데이터 동기화 전략

**Option 1: 필요할 때만** (현재 기본값)
```python
# 사용자가 데이터가 필요할 때마다
candles = await service.get_klines_with_cache(...)
# → DB 확인 → 필요하면 수집
```

**Option 2: 정기 백그라운드 동기화**
```python
# 백그라운드에서 정기적으로 (스케줄러)
# → 사용자 요청 시 항상 최신 데이터
```

**Option 3: Webhook** (향후 구현)
```
Binance Stream → 새 캔들 수신 → 즉시 DB 저장
```

### 데이터 정리 (선택사항)
```python
# 30일 이상 오래된 1분 캔들 삭제
# 90일 이상 오래된 5분 캔들 삭제
async def cleanup_old_data(db):
    from datetime import datetime, timedelta
    
    cutoff_1m = datetime.utcnow() - timedelta(days=30)
    await db.execute(
        delete(MarketCandle).where(
            MarketCandle.timeframe == "1m",
            MarketCandle.open_time < cutoff_1m
        )
    )
    await db.commit()
```

---

## 🚀 권장 사항

### 1. 초기 데이터 로딩
```bash
# 처음 한 번만 실행
POST /api/v1/data/sync-all
{
    "symbols": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "timeframes": ["1h", "4h", "1d"]
}
```

### 2. 정기 업데이트
```python
# 매 1시간마다 자동으로 실행되도록 설정
# → Celery, APScheduler 등 사용
```

### 3. 모니터링
```bash
# 정기적으로 데이터 커버리지 확인
GET /api/v1/data/coverage
```

---

## ⚠️ 주의사항

1. **첫 수집 시간**: 초기에는 전체 데이터를 수집하므로 시간이 걸림
2. **API 레이트**: 여러 심볼 동시 수집 시 주의 (스케줄링 필요)
3. **DB 저장소**: 장기 운영 시 DB 크기 증가 (정리 필요)
4. **시간 동기화**: UTC 시간 기준

---

## 📋 체크리스트

- [x] IncrementalDataCollector 구현
- [x] UnifiedDataService 구현
- [x] Data API 엔드포인트 추가
- [x] ai_signal.py에 통합
- [x] 마이그레이션 설정
- [ ] 백그라운드 스케줄러 설정 (선택사항)
- [ ] 데이터 정리 스크립트 (선택사항)
- [ ] 모니터링 대시보드 (선택사항)

---

## 🔗 관련 파일

- 증분 수집: `app/services/incremental_collector.py`
- 통합 서비스: `app/services/unified_data_service.py`
- 데이터 API: `app/api/data.py`
- AI 신호: `app/api/ai_signal.py` (수정됨)
- 마켓 데이터: `app/models/market_data.py`

---

**생성일**: 2026-01-22
**상태**: ✅ 완성 및 테스트 가능
