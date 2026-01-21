import { create } from 'zustand';

interface Balance {
  asset: string;
  free: number;
  locked: number;
  total: number;
}

interface Position {
  symbol: string;
  quantity: number;
  current_price: number;
  value_usdt: number;
}

interface Signal {
  symbol: string;
  signal: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  predicted_direction: string;
  current_price: number;
  analysis: string;
}

interface TradingSettings {
  default_stop_loss: number;
  default_take_profit: number;
  max_position_size: number;
  auto_trading_enabled: boolean;
  prediction_threshold: number;
  max_daily_trades: number;
  max_daily_loss: number;
  trailing_stop_enabled: boolean;
  trailing_stop_percent: number;
}

interface TradingState {
  // Data
  balances: Balance[];
  positions: Position[];
  currentSignal: Signal | null;
  settings: TradingSettings;
  selectedSymbol: string;

  // Loading states
  isLoading: boolean;
  error: string | null;

  // Actions
  setBalances: (balances: Balance[]) => void;
  setPositions: (positions: Position[]) => void;
  setCurrentSignal: (signal: Signal | null) => void;
  setSettings: (settings: TradingSettings) => void;
  setSelectedSymbol: (symbol: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
}

export const useTradingStore = create<TradingState>((set) => ({
  // Initial data
  balances: [],
  positions: [],
  currentSignal: null,
  settings: {
    default_stop_loss: 0.02,
    default_take_profit: 0.05,
    max_position_size: 100,
    auto_trading_enabled: false,
    prediction_threshold: 0.6,
    max_daily_trades: 10,
    max_daily_loss: 0.1,
    trailing_stop_enabled: false,
    trailing_stop_percent: 0.01,
  },
  selectedSymbol: 'BTCUSDT',

  // Loading states
  isLoading: false,
  error: null,

  // Actions
  setBalances: (balances) => set({ balances }),
  setPositions: (positions) => set({ positions }),
  setCurrentSignal: (currentSignal) => set({ currentSignal }),
  setSettings: (settings) => set({ settings }),
  setSelectedSymbol: (selectedSymbol) => set({ selectedSymbol }),
  setLoading: (isLoading) => set({ isLoading }),
  setError: (error) => set({ error }),
}));
