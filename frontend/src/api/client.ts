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
  // TODO: Gemini API 무료 티어 할당량 초과로 인해 일시 비활성화
  // predict: (symbol: string, timeframe = '1h') =>
  //   apiClient.post('/ai/predict', { symbol, timeframe }),
  predict: async (symbol: string, timeframe = '1h') => {
    // 일시 비활성화 - 더미 응답 반환 (axios 래핑 고려)
    return Promise.resolve({
      data: {
        symbol,
        signal: 'HOLD',
        confidence: 0,
        predicted_direction: 'NEUTRAL',
        current_price: 0,
        analysis: 'AI 예측이 일시적으로 비활성화되었습니다. (Gemini API 할당량 초과)',
      },
    });
  },
  // 가중치 기반 통합 분석 (AI + 기술적 지표)
  combinedAnalysis: (symbol: string, timeframe = '1h') =>
    apiClient.post('/ai/combined-analysis', { symbol, timeframe }),
  parsePrompt: (prompt: string) =>
    apiClient.post('/ai/parse-prompt', { prompt }),
  getMarketAnalysis: (symbol: string, timeframe = '1h') =>
    apiClient.get(`/ai/market-analysis/${symbol}`, { params: { timeframe } }),
  getSignals: () => apiClient.get('/ai/signals'),
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
  // 티커 데이터
  getTickers: () => apiClient.get('/market/tickers'),
  getTicker: (symbol: string) => apiClient.get(`/market/ticker/${symbol}`),

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

  // 캔들스틱 차트 데이터
  getKlines: (symbol: string, interval = '1h', limit = 100) =>
    apiClient.get(`/market/klines/${symbol}`, { params: { interval, limit } }),

  // 미니 차트 데이터
  getMiniChart: (symbol: string, interval = '1h', limit = 24) =>
    apiClient.get(`/market/mini-chart/${symbol}`, { params: { interval, limit } }),
};

// WebSocket URL 헬퍼
export const getWebSocketUrl = () => {
  const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
  return wsUrl;
};
