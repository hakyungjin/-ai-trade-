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
  predict: (symbol: string, timeframe = '1h') =>
    apiClient.post('/ai/predict', { symbol, timeframe }),
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
