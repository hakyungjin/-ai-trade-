# 🤖 Crypto AI Trader

**AI 기반 암호화폐 자동매매 시스템** — 실시간 거래, LSTM/Transformer 예측 모델, Gemini AI 분석

- 🔷 **현물/선물 분리 거래**: 한 버튼으로 전환
- 🧠 **AI 예측**: PyTorch LSTM/Transformer + Gemini API
- 📊 **실시간 데이터**: WebSocket 스트리밍
- 🎯 **리스크 관리**: 스탑로스, 익절, 레버리지 제한

---

## 🏗️ 프로젝트 구조

```
crypto-ai-trader/
├── ai-model/                    # AI 학습 파이프라인
│   ├── pipeline.py              # CLI 학습 스크립트
│   ├── requirements.txt
│   ├── README.md
│   └── training/
│       ├── data_collector.py    # Binance 데이터 수집
│       ├── feature_engineering.py
│       ├── train.py
│       └── model.py
│
├── backend/                     # FastAPI 백엔드
│   ├── app/
│   │   ├── main.py
│   │   ├── config.py
│   │   ├── api/
│   │   │   ├── trading.py
│   │   │   ├── ai_signal.py
│   │   │   ├── realtime.py
│   │   │   └── settings.py
│   │   ├── models/
│   │   └── services/
│   ├── requirements.txt
│   ├── README.md
│   └── .env.example
│
├── frontend/                    # React + TypeScript + Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── Dashboard.tsx
│   │   │   ├── trading/
│   │   │   │   ├── SpotTrading.tsx
│   │   │   │   └── FuturesTrading.tsx
│   │   │   └── ui/
│   │   ├── api/
│   │   └── store/
│   ├── package.json
│   ├── README.md
│   └── vite.config.ts
│
├── .gitignore
├── README.md
└── .env.example
```

---

## 🚀 빠른 시작

### 필수 요구사항

- **Python 3.11+** (AI 모델)
- **Node.js 18+** (프론트엔드)
- **Binance API Key** (테스트넷 가능)

### 1️⃣ 저장소 클론

```bash
git clone https://github.com/YOUR_USERNAME/crypto-ai-trader.git
cd crypto-ai-trader
```

### 2️⃣ 백엔드 시작 (터미널 1)

```bash
cd backend

# 가상환경 (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
# 또는 (macOS/Linux)
source .venv/bin/activate

# 의존성
pip install --prefer-binary -r requirements.txt

# .env 설정
# BINANCE_API_KEY=your_key
# BINANCE_SECRET_KEY=your_secret
# BINANCE_TESTNET=true

# 실행
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**API 문서**: http://localhost:8000/docs

### 3️⃣ 프론트엔드 시작 (터미널 2)

```bash
cd frontend
npm install
npm run dev
```

**접속**: http://localhost:5173

### 4️⃣ AI 모델 학습 (선택, 터미널 3)

```bash
cd ai-model
python -m venv .venv
source .venv/bin/activate
pip install --prefer-binary -r requirements.txt

# 학습 시작
python pipeline.py --symbol BTCUSDT --epochs 50
```

---

## 📖 주요 기능

### 🔷 현물 거래 (Spot)

```
[매수] [매도]
시장가 / 지정가 선택
스탑로스 & 익절 설정
```

### 🟠 선물 거래 (Futures)

```
[롱] [숏]
1배 ~ 20배 레버리지
리스크 자동 계산
필수: 스탑로스 + 익절
```

### 🧠 AI 신호

- LSTM 모델 (장기 의존성)
- Transformer 모델 (병렬 처리)
- Gemini AI (자연어 분석)

---

## 🔌 API 빠른 참조

```bash
# 가격 조회
curl http://localhost:8000/api/trading/price/BTCUSDT

# AI 예측
curl -X POST http://localhost:8000/api/ai/predict \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTCUSDT","timeframe":"1h"}'

# WebSocket 실시간
ws://localhost:8000/api/realtime/ws/market/BTCUSDT
```

📚 [전체 API 문서](backend/README.md)

---

## 📚 상세 문서

- [Backend README](backend/README.md) — API, 설정, 오류 해결
- [Frontend README](frontend/README.md) — UI, 컴포넌트, 스타일
- [AI Model README](ai-model/README.md) — 학습, 데이터, 모델

---

## 🛠️ 기술 스택

**Backend**: FastAPI · PyTorch · Pydantic · SQLAlchemy
**Frontend**: React · TypeScript · Vite · Tailwind · Zustand
**AI**: LSTM · Transformer · pandas · scikit-learn

---

## 📋 상태

- ✅ 백엔드 API
- ✅ 현물/선물 UI
- ✅ AI 학습 파이프라인
- ✅ WebSocket 실시간
- 🔄 데이터베이스
- ⏳ 자동 매매 봇
- ⏳ Docker 배포

---

## 🤝 기여

1. Fork
2. Feature branch 생성: `git checkout -b feature/YourFeature`
3. Commit: `git commit -m 'Add YourFeature'`
4. Push: `git push origin feature/YourFeature`
5. Pull Request

---

## 📄 라이선스

MIT

---

**Made with ❤️ — Last updated: 2026-01-21**
