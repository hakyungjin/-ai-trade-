import { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { TrendingUp, TrendingDown, Activity, RefreshCw } from 'lucide-react';
import { marketApi, getWebSocketUrl } from '@/api/client';
import { MiniChart } from './MiniChart';

interface Ticker {
  symbol: string;
  price: number;
  priceChange: number;
  priceChangePercent: number;
  volume: number;
  quoteVolume: number;
  highPrice: number;
  lowPrice: number;
}

interface MarketStats {
  totalCoins: number;
  gainersCount: number;
  losersCount: number;
  totalVolume: number;
  topGainer: Ticker | null;
  topLoser: Ticker | null;
}

export function MarketOverview() {
  const [majorCoins, setMajorCoins] = useState<Ticker[]>([]);
  const [gainers, setGainers] = useState<Ticker[]>([]);
  const [losers, setLosers] = useState<Ticker[]>([]);
  const [stats, setStats] = useState<MarketStats | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // 초기 데이터 로드
    loadInitialData();

    // WebSocket 연결
    const wsUrl = getWebSocketUrl();
    const socket = new WebSocket(`${wsUrl}/api/market/ws/tickers`);

    socket.onopen = () => {
      setIsConnected(true);
    };

    socket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === 'market_update') {
          setMajorCoins(data.majorCoins || []);
          setGainers(data.gainers || []);
          setLosers(data.losers || []);
          setStats(data.stats || null);
        }
      } catch (err) {
        console.error('Failed to parse market data:', err);
      }
    };

    socket.onerror = () => {
      setIsConnected(false);
    };

    socket.onclose = () => {
      setIsConnected(false);
      // 5초 후 재연결은 상위 컴포넌트에서 처리
    };

    return () => {
      socket.close();
    };
  }, []);

  const loadInitialData = async () => {
    try {
      setIsLoading(true);
      const [overviewRes, gainersLosersRes] = await Promise.all([
        marketApi.getOverview(),
        marketApi.getGainersLosers(10),
      ]);

      if (overviewRes.data.success) {
        setMajorCoins(overviewRes.data.majorCoins || []);
        setStats(overviewRes.data.stats || null);
      }

      if (gainersLosersRes.data.success) {
        setGainers(gainersLosersRes.data.gainers || []);
        setLosers(gainersLosersRes.data.losers || []);
      }
    } catch (error) {
      console.error('Failed to load market data:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const formatPrice = (price: number) => {
    if (price >= 1000) return price.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (price >= 1) return price.toFixed(4);
    return price.toFixed(6);
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1e9) return `$${(volume / 1e9).toFixed(2)}B`;
    if (volume >= 1e6) return `$${(volume / 1e6).toFixed(2)}M`;
    if (volume >= 1e3) return `$${(volume / 1e3).toFixed(2)}K`;
    return `$${volume.toFixed(2)}`;
  };

  const formatSymbol = (symbol: string) => {
    return symbol.replace('USDT', '');
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <RefreshCw className="w-8 h-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* 연결 상태 & 통계 */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5" />
          <span className="font-semibold">마켓 개요</span>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
          <span className="text-xs text-muted-foreground">
            {isConnected ? '실시간' : '연결 끊김'}
          </span>
        </div>
        {stats && (
          <div className="flex gap-4 text-sm">
            <span className="text-green-500">상승: {stats.gainersCount}</span>
            <span className="text-red-500">하락: {stats.losersCount}</span>
            <span className="text-muted-foreground">거래량: {formatVolume(stats.totalVolume)}</span>
          </div>
        )}
      </div>

      {/* 주요 코인 */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-lg">주요 코인</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-4">
            {majorCoins.map((coin) => (
              <div
                key={coin.symbol}
                className="p-3 rounded-lg border bg-card hover:bg-accent transition-colors cursor-pointer"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold">{formatSymbol(coin.symbol)}</span>
                  <Badge
                    variant={coin.priceChangePercent >= 0 ? 'default' : 'destructive'}
                    className="text-xs"
                  >
                    {coin.priceChangePercent >= 0 ? '+' : ''}
                    {coin.priceChangePercent.toFixed(2)}%
                  </Badge>
                </div>
                <div className="text-lg font-semibold">${formatPrice(coin.price)}</div>
                <div className="mt-2">
                  <MiniChart symbol={coin.symbol} trend={coin.priceChangePercent >= 0 ? 'up' : 'down'} />
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* 상승/하락 코인 */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* 상승 TOP */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-green-500" />
              상승 TOP 10
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {gainers.map((coin, index) => (
                <div
                  key={coin.symbol}
                  className="flex items-center justify-between p-2 rounded hover:bg-accent transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground w-5">{index + 1}</span>
                    <span className="font-medium">{formatSymbol(coin.symbol)}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm">${formatPrice(coin.price)}</span>
                    <Badge variant="default" className="bg-green-500 hover:bg-green-600 min-w-[70px] justify-center">
                      +{coin.priceChangePercent.toFixed(2)}%
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* 하락 TOP */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-lg flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-red-500" />
              하락 TOP 10
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {losers.map((coin, index) => (
                <div
                  key={coin.symbol}
                  className="flex items-center justify-between p-2 rounded hover:bg-accent transition-colors cursor-pointer"
                >
                  <div className="flex items-center gap-3">
                    <span className="text-muted-foreground w-5">{index + 1}</span>
                    <span className="font-medium">{formatSymbol(coin.symbol)}</span>
                  </div>
                  <div className="flex items-center gap-4">
                    <span className="text-sm">${formatPrice(coin.price)}</span>
                    <Badge variant="destructive" className="min-w-[70px] justify-center">
                      {coin.priceChangePercent.toFixed(2)}%
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
