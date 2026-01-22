# 🚀 실전 트레이딩 가이드

암호화폐 자동 트레이딩 봇 실전 운영 가이드

## 📋 목차

1. [시작하기 전에](#시작하기-전에)
2. [백테스팅](#백테스팅)
3. [페이퍼 트레이딩](#페이퍼-트레이딩)
4. [소액 실전 테스트](#소액-실전-테스트)
5. [리스크 관리](#리스크-관리)
6. [모니터링](#모니터링)

---

## 시작하기 전에

### ⚠️ 필수 체크리스트

- [ ] 백테스팅 완료 (최소 30일 이상)
- [ ] 페이퍼 트레이딩 완료 (최소 1주일)
- [ ] 리스크 관리 규칙 숙지
- [ ] 바이낸스 API 키 발급 및 권한 설정
- [ ] 거래 수수료 이해
- [ ] 긴급 상황 대응 계획 수립

### 💰 권장 초기 자본

- **최소**: $100 (테스트용)
- **적정**: $500 - $1,000
- **권장**: 전체 투자 자본의 1-5%

### ⏰ 시작 시기

- 변동성이 적은 시간대 (주말 제외)
- 중요한 경제 이벤트 직전/직후 피하기
- 시장이 안정적인 시기에 시작

---

## 백테스팅

### 1. 데이터 수집 및 백테스트 실행

```bash
cd backend
python test_strategies.py --symbol BTCUSDT --days 90 --capital 10000
```

### 2. 결과 분석

**좋은 전략의 조건:**
- 샤프 비율 > 1.0
- 승률 > 50%
- Profit Factor > 1.5
- 최대 낙폭 < 20%
- 충분한 거래 횟수 (최소 20회)

**나쁜 신호:**
- 과도한 거래 빈도 (과최적화)
- 너무 적은 거래 (신호 부족)
- 불안정한 수익 곡선
- 최근 성과 급격히 악화

### 3. 여러 시장 조건에서 테스트

```bash
# 상승장
python test_strategies.py --symbol BTCUSDT --days 60 --capital 10000

# 하락장
python test_strategies.py --symbol ETHUSDT --days 60 --capital 10000

# 횡보장
python test_strategies.py --symbol BNBUSDT --days 60 --capital 10000
```

---

## 페이퍼 트레이딩

### 바이낸스 테스트넷 사용

1. **테스트넷 계정 생성**
   - https://testnet.binance.vision/
   - 무료 테스트 자금 받기

2. **.env 설정**
   ```bash
   BINANCE_API_KEY=your_testnet_key
   BINANCE_SECRET_KEY=your_testnet_secret
   BINANCE_TESTNET=true
   ```

3. **봇 실행**
   ```bash
   uvicorn app.main:app --reload
   ```

### 페이퍼 트레이딩 체크리스트

- [ ] 1주일 이상 연속 운영
- [ ] 수익성 확인
- [ ] 예상치 못한 버그 없음
- [ ] 리스크 관리 작동 확인
- [ ] 긴급 정지 기능 테스트

---

## 소액 실전 테스트

### 1단계: 환경 설정

#### API 키 생성 (바이낸스)

1. 바이낸스 계정 로그인
2. API Management
3. Create API Key
4. **중요**: 권한 설정
   - ✅ Enable Reading
   - ✅ Enable Spot & Margin Trading
   - ❌ Enable Withdrawals (비활성화!)
5. IP 화이트리스트 설정 (선택)

#### .env 설정 (실전)

```bash
BINANCE_API_KEY=your_real_api_key
BINANCE_SECRET_KEY=your_real_secret_key
BINANCE_TESTNET=false

# 리스크 관리 설정
RISK_LEVEL=conservative
MAX_POSITION_SIZE_PCT=0.3
STOP_LOSS_PCT=2.0
TAKE_PROFIT_PCT=4.0
MAX_DAILY_LOSS_PCT=5.0
```

### 2단계: 초기 설정

```python
# risk_config.json 생성
{
  "risk_level": "conservative",
  "max_position_size_pct": 0.3,
  "stop_loss_pct": 2.0,
  "take_profit_pct": 4.0,
  "max_daily_loss_pct": 5.0,
  "max_weekly_loss_pct": 10.0,
  "max_concurrent_positions": 1,
  "use_trailing_stop": true,
  "trailing_stop_pct": 1.0
}
```

### 3단계: 봇 실행

```bash
# 백엔드 시작
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 프론트엔드 시작 (별도 터미널)
cd frontend
npm run dev
```

### 4단계: 첫 거래 전 확인사항

- [ ] 잔고 확인
- [ ] 리스크 설정 확인
- [ ] 손절/익절 자동 설정 확인
- [ ] 알림 설정 (텔레그램, 이메일 등)
- [ ] 모니터링 대시보드 확인

---

## 리스크 관리

### 포지션 크기 관리

**Conservative (보수적)**
```
- 최대 포지션: 계좌의 30%
- 손절: 2%
- 익절: 4%
- 레버리지: 1-2배
```

**Moderate (중도)**
```
- 최대 포지션: 계좌의 50%
- 손절: 2.5%
- 익절: 5%
- 레버리지: 3-5배
```

**Aggressive (공격적)** ⚠️ 초보자 비권장
```
- 최대 포지션: 계좌의 80%
- 손절: 3%
- 익절: 6%
- 레버리지: 5-10배
```

### 손실 한도

| 한도 | 조치 |
|------|------|
| 일일 -5% | 거래 중지 (당일) |
| 주간 -10% | 거래 중지 (1주일) |
| 월간 -20% | 전략 재검토 필수 |
| 총 -30% | 시스템 중단, 전면 재점검 |

### 비상 상황 대응

1. **급격한 시장 변동**
   - 모든 포지션 즉시 청산
   - 거래 중지
   - 원인 분석 후 재개

2. **API 오류**
   - 자동 재시도 (3회)
   - 실패 시 알림 발송
   - 수동 개입 필요

3. **네트워크 장애**
   - 연결 복구 시도
   - 복구 불가 시 텔레그램 알림
   - 웹 대시보드에서 수동 조작

---

## 모니터링

### 일일 체크사항

- [ ] 수익/손실 확인
- [ ] 거래 내역 검토
- [ ] 리스크 지표 확인
- [ ] 오류 로그 확인
- [ ] 시스템 상태 확인

### 주간 체크사항

- [ ] 전략 성과 분석
- [ ] 시장 조건 변화 확인
- [ ] 설정 조정 필요성 검토
- [ ] 백업 확인

### 성과 지표

```python
# 매주 확인할 지표
- 총 수익률
- 주간 수익률
- 샤프 비율
- 승률
- 평균 수익/손실
- 최대 낙폭
- 거래 횟수
- 수수료 비용
```

### 로그 확인

```bash
# 로그 파일 위치
tail -f backend/logs/trading.log

# 오류만 확인
grep "ERROR" backend/logs/trading.log

# 거래 내역만 확인
grep "Trade" backend/logs/trading.log
```

---

## 🚨 중요 경고

### 절대 하지 말아야 할 것

❌ **출금 권한 부여하지 않기**
❌ **API 키 공유하지 않기**
❌ **감당할 수 없는 금액 투자하지 않기**
❌ **손실 한도 무시하지 않기**
❌ **검증되지 않은 전략 사용하지 않기**
❌ **과도한 레버리지 사용하지 않기**
❌ **모니터링 없이 장기간 방치하지 않기**

### 권장사항

✅ **작은 금액으로 시작**
✅ **충분한 백테스팅과 페이퍼 트레이딩**
✅ **리스크 관리 규칙 엄수**
✅ **정기적인 모니터링**
✅ **시장 상황 지속 관찰**
✅ **손실 시 즉시 중단하고 분석**
✅ **감정적 개입 최소화**

---

## 📞 문제 발생 시

1. **즉시 거래 중지**
   ```bash
   # API에서 긴급 정지
   curl -X POST http://localhost:8000/api/trading/emergency-stop
   ```

2. **로그 확인**
   ```bash
   cat backend/logs/trading.log
   ```

3. **수동 청산**
   - 바이낸스 웹/앱에서 직접 포지션 청산

4. **문제 분석**
   - 로그 검토
   - 거래 내역 확인
   - 시스템 상태 점검

---

## 📚 추가 자료

- [바이낸스 API 문서](https://binance-docs.github.io/apidocs/spot/en/)
- [리스크 관리 가이드](https://www.investopedia.com/risk-management)
- [기술적 분석 학습](https://www.tradingview.com/education/)

---

**면책 조항**: 이 시스템은 교육 목적으로 제공됩니다. 실제 투자 시 발생하는 손실에 대해 개발자는 책임지지 않습니다. 투자는 신중하게 결정하시기 바랍니다.
