-- Supabase 테이블 초기화 SQL
-- 실행: python scripts/run_supabase_init.py

-- ================================================================
-- 1️⃣ Coins 테이블 (기본 테이블 - 다른 테이블이 이것을 참조)
-- ================================================================
CREATE TABLE IF NOT EXISTS coins (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    name VARCHAR(100),
    current_price DECIMAL(20, 8),
    market_cap DECIMAL(30, 2),
    market_cap_rank INTEGER,
    volume_24h DECIMAL(30, 2),
    price_change_24h DECIMAL(10, 4),
    price_change_percentage_24h DECIMAL(10, 4),
    circulating_supply DECIMAL(30, 8),
    total_supply DECIMAL(30, 8),
    all_time_high DECIMAL(20, 8),
    all_time_low DECIMAL(20, 8),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    monitoring BOOLEAN DEFAULT FALSE,
    market_type VARCHAR(10) DEFAULT 'spot',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_coin_symbol ON coins(symbol);
CREATE INDEX IF NOT EXISTS idx_coin_market_type ON coins(market_type);
CREATE INDEX IF NOT EXISTS idx_coin_symbol_market ON coins(symbol, market_type);

-- ================================================================
-- 2️⃣ Candles 테이블 (OHLCV 캔들 데이터)
-- ================================================================
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id) ON DELETE CASCADE,
    timeframe VARCHAR(10) NOT NULL,
    open_time TIMESTAMP NOT NULL,
    close_time TIMESTAMP NOT NULL,
    open_price DECIMAL(20, 8),
    high_price DECIMAL(20, 8),
    low_price DECIMAL(20, 8),
    close_price DECIMAL(20, 8),
    volume DECIMAL(30, 8),
    quote_asset_volume DECIMAL(30, 8),
    number_of_trades INTEGER,
    taker_buy_base_asset_volume DECIMAL(30, 8),
    taker_buy_quote_asset_volume DECIMAL(30, 8),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(coin_id, timeframe, open_time)
);

CREATE INDEX IF NOT EXISTS idx_candle_coin_timeframe ON candles(coin_id, timeframe);
CREATE INDEX IF NOT EXISTS idx_candle_open_time ON candles(open_time);

-- ================================================================
-- 3️⃣ Signals 테이블 (트레이딩 신호)
-- ================================================================
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id) ON DELETE CASCADE,
    signal_type VARCHAR(20) NOT NULL,
    timeframe VARCHAR(10),
    confidence DECIMAL(5, 4),
    indicators TEXT,
    price DECIMAL(20, 8),
    signal_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_signal_coin_type ON signals(coin_id, signal_type);
CREATE INDEX IF NOT EXISTS idx_signal_time ON signals(signal_time);

-- ================================================================
-- 4️⃣ Training Data 테이블 (AI 모델 학습 데이터)
-- ================================================================
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id) ON DELETE CASCADE,
    timeframe VARCHAR(10),
    data_type VARCHAR(50),
    file_path VARCHAR(255),
    record_count INTEGER,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_data_coin ON training_data(coin_id);

-- ================================================================
-- 5️⃣ Stocks 테이블 (미국 주식)
-- ================================================================
CREATE TABLE IF NOT EXISTS stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(10) NOT NULL UNIQUE,
    name VARCHAR(100),
    sector VARCHAR(50),
    industry VARCHAR(50),
    current_price DECIMAL(20, 4),
    pe_ratio DECIMAL(10, 2),
    market_cap DECIMAL(30, 2),
    dividend_yield DECIMAL(10, 4),
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    monitoring BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_stock_symbol ON stocks(symbol);
CREATE INDEX IF NOT EXISTS idx_stock_monitoring ON stocks(monitoring);

-- ================================================================
-- ✅ 모든 테이블 생성 완료!
-- ================================================================
