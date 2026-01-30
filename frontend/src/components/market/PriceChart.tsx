import { useEffect, useState, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import {
  ComposedChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from 'recharts';
import { TrendingUp, TrendingDown, Minus, RefreshCw } from 'lucide-react';
import { marketApi, getWebSocketUrl } from '@/api/client';

interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PriceChartProps {
  symbol: string;
  onSymbolChange?: (symbol: string) => void;
}

const INTERVALS = [
  { value: '1m', label: '1분' },
  { value: '5m', label: '5분' },
  { value: '15m', label: '15분' },
  { value: '1h', label: '1시간' },
  { value: '4h', label: '4시간' },
  { value: '1d', label: '1일' },
];

export function PriceChart({ symbol }: PriceChartProps) {
  const [candles, setCandles] = useState<Candle[]>([]);
  const [selectedInterval, setSelectedInterval] = useState('1m');
  const [currentPrice, setCurrentPrice] = useState(0);
  const [priceChange, setPriceChange] = useState(0);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  
  // 지표 표시 토글
  const [showRSI, setShowRSI] = useState(true);
  const [showMACD, setShowMACD] = useState(true);
  const [showBB, setShowBB] = useState(false);
  
  // 지표 데이터
  const [indicators, setIndicators] = useState<any>({});
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mountedRef = useRef(true);
  const reconnectAttemptsRef = useRef(0);

  useEffect(() => {
    mountedRef.current = true;
    reconnectAttemptsRef.current = 0;
    
    // 기존 WebSocket 정리
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    // 기존 재연결 타임아웃 정리
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    // 새 데이터 로드 및 연결
    loadInitialData();
    connectWebSocket();

    return () => {
      mountedRef.current = false;
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }
    };
  }, [symbol, selectedInterval]);

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      const response = await marketApi.getKlines(symbol, selectedInterval, 200);
      if (response.data.success && mountedRef.current) {
        setCandles(response.data.data);
        if (response.data.data.length > 0) {
          const latest = response.data.data[response.data.data.length - 1];
          const firstCandle = response.data.data[0];
          setCurrentPrice(latest.close);
          
          // 첫 캔들의 시작가를 기준으로 변화율 계산
          const change = ((latest.close - firstCandle.open) / firstCandle.open) * 100;
          setPriceChange(change);
        }
      }
    } catch (error) {
      console.error('Failed to load chart data:', error);
    } finally {
      if (mountedRef.current) {
        setIsLoading(false);
      }
    }
  };

  const connectWebSocket = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
    }

    try {
      const wsUrl = getWebSocketUrl();
      const socket = new WebSocket(`${wsUrl}/api/chart/ws/realtime/${symbol}?interval=${selectedInterval}`);

      socket.onopen = () => {
        console.log(`Chart WebSocket connected: ${symbol} ${selectedInterval}`);
        reconnectAttemptsRef.current = 0; // 연결 성공 시 재설정
        if (mountedRef.current) {
          setIsConnected(true);
        }
      };

      socket.onmessage = (event) => {
        if (!mountedRef.current) return;

        try {
          const data = JSON.parse(event.data);

          if (data.type === 'initial') {
            console.log(`[${symbol} ${selectedInterval}] Received initial data: ${data.data.length} candles`);
            setCandles(data.data);
            
            // 지표 데이터 추출 (있으면)
            if (data.indicators) {
              setIndicators(data.indicators);
            }
            
            if (data.data.length > 0) {
              const latest = data.data[data.data.length - 1];
              const firstCandle = data.data[0];
              setCurrentPrice(latest.close);
              
              // 변화율 계산
              const change = ((latest.close - firstCandle.open) / firstCandle.open) * 100;
              setPriceChange(change);
            }
          } else if (data.type === 'update' || data.type === 'kline') {
            // 최신 캔들 업데이트
            if (data.latestCandle) {
              let candleList: Candle[] = [];
              
              setCandles((prev) => {
                candleList = prev;
                const newCandles = [...prev];
                const lastIndex = newCandles.length - 1;

                if (lastIndex >= 0 && newCandles[lastIndex].timestamp === data.latestCandle.timestamp) {
                  // 같은 캔들 업데이트
                  newCandles[lastIndex] = data.latestCandle;
                } else {
                  // 새 캔들 추가
                  newCandles.push(data.latestCandle);
                  if (newCandles.length > 200) {
                    newCandles.shift();
                  }
                }

                return newCandles;
              });

              setCurrentPrice(data.latestCandle.close);
              
              // 변화율 업데이트 (첫 캔들 기준)
              if (candleList.length > 0) {
                const change = ((data.latestCandle.close - candleList[0].open) / candleList[0].open) * 100;
                setPriceChange(change);
              }
            }
          } else if (data.type === 'close') {
            // 백엔드에서 보낸 close 신호 처리
            console.log(`[${symbol} ${selectedInterval}] Received close signal: ${data.reason}`);
            if (wsRef.current) {
              wsRef.current.close();
              wsRef.current = null;
            }
          } else if (data.type === 'error') {
            console.error(`[${symbol} ${selectedInterval}] WebSocket error:`, data.message);
          }
        } catch (err) {
          console.error('Failed to parse WebSocket message:', err);
        }
      };

      socket.onerror = (error) => {
        console.error('WebSocket error:', error);
        if (mountedRef.current) {
          setIsConnected(false);
        }
      };

      socket.onclose = () => {
        console.log('Chart WebSocket disconnected');
        wsRef.current = null;
        if (mountedRef.current) {
          setIsConnected(false);
          
          // 재연결 시도 제한 (최대 5회)
          if (reconnectAttemptsRef.current < 5) {
            // exponential backoff: 1초, 2초, 4초, 8초, 15초
            const delays = [1000, 2000, 4000, 8000, 15000];
            const delay = delays[reconnectAttemptsRef.current];
            reconnectAttemptsRef.current += 1;
            console.log(`Reconnecting WebSocket in ${delay}ms (attempt ${reconnectAttemptsRef.current}/5)...`);
            
            reconnectTimeoutRef.current = setTimeout(() => {
              if (mountedRef.current && wsRef.current === null) {
                connectWebSocket();
              }
            }, delay);
          } else {
            console.error('Max reconnection attempts reached');
            setIsLoading(false);
          }
        }
      };

      wsRef.current = socket;
    } catch (error) {
      console.error('Failed to connect WebSocket:', error);
      if (mountedRef.current) {
        setIsConnected(false);
      }
    }
  }, [symbol, selectedInterval]);

  const formatPrice = (price: number) => {
    if (price >= 1000) return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    return price.toFixed(6);
  };

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    if (selectedInterval === '1d' || selectedInterval === '1w') {
      return date.toLocaleDateString('ko-KR', { month: 'short', day: 'numeric' });
    }
    return date.toLocaleTimeString('ko-KR', { hour: '2-digit', minute: '2-digit' });
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) return `${(volume / 1e9).toFixed(1)}B`;
    if (volume >= 1e6) return `${(volume / 1e6).toFixed(1)}M`;
    if (volume >= 1e3) return `${(volume / 1e3).toFixed(1)}K`;
    return volume.toFixed(0);
  };

  // 차트 데이터 변환 (캔들스틱)
  const chartData = candles.map((candle) => {
    const isGain = candle.close >= candle.open;
    return {
      time: formatTime(candle.timestamp),
      timestamp: candle.timestamp,
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
      volume: candle.volume,
      color: isGain ? '#22c55e' : '#ef4444', // 상승(초록), 하락(빨강)
      displayPrice: isGain ? candle.close : candle.open, // 캔들 표시용
    };
  });

  // 가격 범위 계산
  const prices = candles.length > 0 ? candles.map((c) => [c.high, c.low]).flat() : [0];
  const minPrice = prices.length > 0 ? Math.min(...prices) * 0.999 : 0;
  const maxPrice = prices.length > 0 ? Math.max(...prices) * 1.001 : 1;

  const TrendIcon = priceChange > 0 ? TrendingUp : priceChange < 0 ? TrendingDown : Minus;



  if (isLoading || candles.length === 0) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-96">
          <div className="flex flex-col items-center gap-2">
            <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
            <p className="text-sm text-muted-foreground">차트 데이터 로딩 중...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <CardTitle className="text-xl">{symbol.replace('USDT', '')}/USDT</CardTitle>
            <div className="flex items-center gap-2">
              <span className="text-2xl font-bold">${formatPrice(currentPrice)}</span>
              <Badge
                variant={priceChange >= 0 ? 'default' : 'destructive'}
                className="flex items-center gap-1"
              >
                <TrendIcon className="w-3 h-3" />
                {priceChange >= 0 ? '+' : ''}
                {priceChange.toFixed(2)}%
              </Badge>
            </div>
            <div className={`flex items-center gap-2 px-2 py-1 rounded-md ${isConnected ? 'bg-green-500/10' : 'bg-red-500/10'}`}>
              <div className={`w-2 h-2 rounded-full transition-all ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`} />
              <span className="text-xs font-medium text-muted-foreground">{isConnected ? '연결됨' : '연결 중...'}</span>
            </div>
          </div>

          {/* 인터벌 선택 */}
          <div className="flex gap-1">
            {INTERVALS.map((int) => (
              <Button
                key={int.value}
                variant={selectedInterval === int.value ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setSelectedInterval(int.value)}
              >
                {int.label}
              </Button>
            ))}
          </div>
        </div>
        
        {/* 지표 토글 버튼 */}
        <div className="flex gap-2 mt-3">
          <Button
            variant={showRSI ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowRSI(!showRSI)}
            className="text-xs"
          >
            RSI
          </Button>
          <Button
            variant={showMACD ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowMACD(!showMACD)}
            className="text-xs"
          >
            MACD
          </Button>
          <Button
            variant={showBB ? 'default' : 'outline'}
            size="sm"
            onClick={() => setShowBB(!showBB)}
            className="text-xs"
          >
            Bollinger Bands
          </Button>
        </div>
      </CardHeader>

      <CardContent>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 20, right: 40, left: 0, bottom: 0 }}>
              <XAxis
                dataKey="time"
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: '#6b7280' }}
                interval="preserveStartEnd"
              />
              <YAxis
                yAxisId="price"
                domain={[minPrice, maxPrice]}
                axisLine={false}
                tickLine={false}
                tick={{ fontSize: 11, fill: '#6b7280' }}
                tickFormatter={(value) => formatPrice(value)}
                orientation="right"
              />
              <YAxis
                yAxisId="volume"
                domain={[0, 'dataMax']}
                axisLine={false}
                tickLine={false}
                tick={false}
                orientation="left"
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: 'hsl(var(--card))',
                  border: '1px solid hsl(var(--border))',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelStyle={{ color: 'hsl(var(--foreground))' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload;
                    return (
                      <div className="p-2 space-y-1">
                        <p className="font-bold">{data.time}</p>
                        <p className="text-green-500">고: ${formatPrice(data.high)}</p>
                        <p className="text-red-500">저: ${formatPrice(data.low)}</p>
                        <p className={data.close >= data.open ? 'text-green-500' : 'text-red-500'}>
                          시: ${formatPrice(data.open)} → 종: ${formatPrice(data.close)}
                        </p>
                        <p className="text-gray-500">거래량: {formatVolume(data.volume)}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <ReferenceLine
                yAxisId="price"
                y={currentPrice}
                stroke="#3b82f6"
                strokeDasharray="3 3"
              />
              
              {/* 거래량 바 */}
              <Bar
                yAxisId="volume"
                dataKey="volume"
                fill="#6b728033"
                name="거래량"
                isAnimationActive={false}
              />

              {/* 캔들스틱: 상단 심지 (High - Open/Close 중 높은 값) */}
              <Bar
                yAxisId="price"
                dataKey="high"
                fill="transparent"
                isAnimationActive={false}
                shape={(props: any): React.ReactElement => {
                  const { x, y, width, payload } = props;
                  if (!payload) return <g />;
                  
                  const wickX = x + width / 2;
                  const wickTop = Math.min(y, y + 10); // high의 Y값
                  const wickBottom = Math.max(y, y + 10); // open/close 중 높은 값의 Y값
                  
                  return (
                    <line
                      x1={wickX}
                      y1={wickTop}
                      x2={wickX}
                      y2={wickBottom}
                      stroke={payload.color}
                      strokeWidth={1.5}
                    />
                  );
                }}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`wick-${index}`} fill={entry.color} />
                ))}
              </Bar>

              {/* 캔들스틱: 본체 (Open-Close) */}
              <Bar
                yAxisId="price"
                dataKey="close"
                fill="transparent"
                isAnimationActive={false}
                shape={(props: any): React.ReactElement => {
                  const { x, y, width, payload } = props;
                  if (!payload) return <g />;
                  
                  const bodyX = x + width / 4;
                  const bodyWidth = width / 2;
                  const bodyHeight = Math.max(Math.abs(payload.close - payload.open) / (maxPrice - minPrice) * 300, 2);
                  const bodyTop = Math.min(y, y + 10);
                  
                  return (
                    <rect
                      x={bodyX}
                      y={bodyTop}
                      width={bodyWidth}
                      height={bodyHeight}
                      fill={payload.color}
                      stroke={payload.color}
                      strokeWidth={1}
                    />
                  );
                }}
              >
                {chartData.map((entry, index) => (
                  <Cell key={`body-${index}`} fill={entry.color} />
                ))}
              </Bar>

              {/* RSI 지표 - 주황색 */}
              {showRSI && indicators.rsi_14 !== undefined && (
                <ReferenceLine
                  yAxisId="price"
                  y={minPrice + (indicators.rsi_14 / 100) * (maxPrice - minPrice)}
                  stroke="#f59e0b"
                  strokeWidth={2}
                  label={{ value: `RSI: ${indicators.rsi_14?.toFixed(1)}`, fill: '#f59e0b', fontSize: 11 }}
                />
              )}

              {/* MACD 지표 - 청색 */}
              {showMACD && indicators.macd !== undefined && (
                <>
                  <ReferenceLine
                    yAxisId="price"
                    y={minPrice + ((indicators.macd || 0) / 0.02) * (maxPrice - minPrice)}
                    stroke="#06b6d4"
                    strokeWidth={2}
                    label={{ value: `MACD: ${indicators.macd?.toFixed(4)}`, fill: '#06b6d4', fontSize: 11 }}
                  />
                  {indicators.macd_signal !== undefined && (
                    <ReferenceLine
                      yAxisId="price"
                      y={minPrice + ((indicators.macd_signal || 0) / 0.02) * (maxPrice - minPrice)}
                      stroke="#8b5cf6"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                    />
                  )}
                </>
              )}

              {/* Bollinger Bands - 분홍색/파랑색 */}
              {showBB && indicators.bb_upper !== undefined && (
                <>
                  <ReferenceLine
                    yAxisId="price"
                    y={indicators.bb_upper}
                    stroke="#ec4899"
                    strokeWidth={1}
                    strokeDasharray="3 3"
                  />
                  {indicators.bb_lower !== undefined && (
                    <ReferenceLine
                      yAxisId="price"
                      y={indicators.bb_lower}
                      stroke="#3b82f6"
                      strokeWidth={1}
                      strokeDasharray="3 3"
                    />
                  )}
                  {indicators.bb_middle !== undefined && (
                    <ReferenceLine
                      yAxisId="price"
                      y={indicators.bb_middle}
                      stroke="#6b7280"
                      strokeWidth={1}
                      strokeDasharray="2 2"
                    />
                  )}
                </>
              )}
            </ComposedChart>
          </ResponsiveContainer>
        </div>

        {/* 고가/저가 정보 */}
        <div className="flex justify-between mt-4 text-sm text-muted-foreground">
          <div>
            <span>고가: </span>
            <span className="text-green-500 font-medium">${formatPrice(maxPrice)}</span>
          </div>
          <div>
            <span>저가: </span>
            <span className="text-red-500 font-medium">${formatPrice(minPrice)}</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
