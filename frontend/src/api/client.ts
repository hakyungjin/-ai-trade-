import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const apiClient = axios.create({
  baseURL: `${API_BASE_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Trading API
export const tradingApi = {
  getBalance: () => apiClient.get('/trading/balance'),
  getPositions: () => apiClient.get('/trading/positions'),
  getPrice: (symbol: string) => apiClient.get(`/trading/price/${symbol}`),
  getHistory: (symbol?: string, limit = 50) =>
    apiClient.get('/trading/history', { params: { symbol, limit } }),
  createOrder: (order: {
    symbol: string;
    side: 'BUY' | 'SELL';
    quantity: number;
    order_type?: 'MARKET' | 'LIMIT';
    price?: number;
    stop_loss?: number;
    take_profit?: number;
  }) => apiClient.post('/trading/order', order),
  cancelOrder: (symbol: string, orderId: string) =>
    apiClient.delete(`/trading/order/${symbol}/${orderId}`),
};

// AI API
export const aiApi = {
  // AI 예측 - 버튼 클릭 시에만 호출 (5m: 학습된 모델과 일치)
  predict: (symbol: string, timeframe = '5m', market_type = 'spot') =>
    apiClient.post('/ai/predict', { symbol, timeframe, market_type }),
  // 가중치 기반 통합 분석 (AI + 기술적 지표) - 버튼 클릭 시에만 호출
  combinedAnalysis: (symbol: string, timeframe = '5m', market_type = 'spot') =>
    apiClient.post('/ai/combined-analysis', { symbol, timeframe, market_type }),
  parsePrompt: (prompt: string) =>
    apiClient.post('/ai/parse-prompt', { prompt }),
  getMarketAnalysis: (symbol: string, timeframe = '1h') =>
    apiClient.get(`/ai/market-analysis/${symbol}`, { params: { timeframe } }),
  getSignals: () => apiClient.get('/ai/signals'),
  // AI 모델 관련
  getModels: () => apiClient.get('/ai/models'),
  checkModel: (symbol: string, timeframe = '5m') =>
    apiClient.get(`/ai/models/${symbol}`, { params: { timeframe } }),
};

// Settings API
export const settingsApi = {
  get: () => apiClient.get('/settings/'),
  update: (settings: Record<string, unknown>) => apiClient.put('/settings/', settings),
  patch: (updates: Record<string, unknown>) => apiClient.patch('/settings/', updates),
  reset: () => apiClient.post('/settings/reset'),
};

// Market API - 실시간 마켓 데이터
export const marketApi = {
  // 티커 데이터 (현물/선물 지원)
  getTickers: () => apiClient.get('/market/tickers'),
  getTicker: (symbol: string, market_type = 'spot') => 
    apiClient.get(`/market/ticker/${symbol}`, { params: { market_type } }),

  // 상승/하락 코인
  getGainersLosers: (limit = 10) =>
    apiClient.get('/market/gainers-losers', { params: { limit } }),

  // 트렌딩 코인
  getTrending: (limit = 20) =>
    apiClient.get('/market/trending', { params: { limit } }),

  // 마켓 개요
  getOverview: () => apiClient.get('/market/overview'),

  // 심볼 검색
  search: (query: string) =>
    apiClient.get('/market/search', { params: { query } }),

  // 캔들스틱 차트 데이터 (현물/선물 지원)
  getKlines: (symbol: string, interval = '1h', limit = 100, market_type = 'spot') =>
    apiClient.get(`/market/klines/${symbol}`, { params: { interval, limit, market_type } }),

  // 미니 차트 데이터
  getMiniChart: (symbol: string, interval = '1h', limit = 24, market_type = 'spot') =>
    apiClient.get(`/market/mini-chart/${symbol}`, { params: { interval, limit, market_type } }),
};

// WebSocket URL 헬퍼
export const getWebSocketUrl = () => {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  return wsUrl;
};
