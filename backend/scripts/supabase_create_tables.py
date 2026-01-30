#!/usr/bin/env python3
"""
Supabase SQL 에디터에 복사할 테이블 생성 SQL
"""

# Coins 테이블
coins_sql = """
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
"""

# Candles 테이블
candles_sql = """
CREATE TABLE IF NOT EXISTS candles (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id),
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
"""

# Signals 테이블
signals_sql = """
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id),
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
"""

# Training Data 테이블
training_data_sql = """
CREATE TABLE IF NOT EXISTS training_data (
    id SERIAL PRIMARY KEY,
    coin_id INTEGER NOT NULL REFERENCES coins(id),
    timeframe VARCHAR(10),
    data_type VARCHAR(50),
    file_path VARCHAR(255),
    record_count INTEGER,
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_training_data_coin ON training_data(coin_id);
"""

# Stocks 테이블
stocks_sql = """
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
"""

if __name__ == '__main__':
    print("=== Supabase SQL 에디터에서 다음을 순서대로 실행하세요 ===\n")
    print("1. Coins 테이블:")
    print(coins_sql)
    print("\n2. Candles 테이블:")
    print(candles_sql)
    print("\n3. Signals 테이블:")
    print(signals_sql)
    print("\n4. Training Data 테이블:")
    print(training_data_sql)
    print("\n5. Stocks 테이블:")
    print(stocks_sql)
