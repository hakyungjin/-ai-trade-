import { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  RefreshCw,
  BarChart3,
  Sparkles,
  Plus,
  Trash2,
  Search,
  Loader,
  X,
  ChevronRight,
  Coins,
  LineChart,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Input } from '@/components/ui/input';
import { cn } from '@/lib/utils';
import { marketApi, aiApi, apiClient } from '@/api/client';
import { PriceChart } from './market';

type MarketType = 'spot' | 'futures';

interface MonitoredCoin {
  id: number;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  market_type: MarketType;
  current_price: number;
  price_change_24h: number;
  volume_24h: number;
  candle_count: number;
}

interface WeightedAnalysis {
  final_signal: 'BUY' | 'SELL' | 'HOLD';
  final_confidence: number;
  weighted_signal?: {
    signal: string;
    score: number;
    confidence: number;
    indicators?: Record<string, any>;
    recommendation?: string;
  };
  ai_prediction?: {
    signal: string;
    confidence: number;
  };
}

export function CoinAnalysis() {
  // 마켓 타입
  const [marketType, setMarketType] = useState<MarketType>('spot');
  
  // 기본 메이저 코인
  const defaultSpotSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  const defaultFuturesSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  
  const [monitoredCoins, setMonitoredCoins] = useState<MonitoredCoin[]>([]);
  const [spotSymbols, setSpotSymbols] = useState<string[]>(defaultSpotSymbols);
  const [futuresSymbols, setFuturesSymbols] = useState<string[]>(defaultFuturesSymbols);
  
  const [selectedSymbol, setSelectedSymbol] = useState('BTCUSDT');
  
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  
  const [weightedAnalysis, setWeightedAnalysis] = useState<WeightedAnalysis | null>(null);
  const [analysisLoading, setAnalysisLoading] = useState(false);
  
  // 코인 추가 모달
  const [showAddModal, setShowAddModal] = useState(false);
  const [searchSymbol, setSearchSymbol] = useState('');
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [addLoading, setAddLoading] = useState(false);

  // 모니터링 코인 로드
  const loadMonitoredCoins = async () => {
    try {
      const response = await apiClient.get('/v1/coins/monitoring');
      const coinsData = response.data.coins || response.data.data || [];
      setMonitoredCoins(coinsData);
      
      // 심볼 목록 업데이트 (마켓 타입별 분리)
      const spotCoins = coinsData.filter((c: MonitoredCoin) => c.market_type === 'spot').map((c: MonitoredCoin) => c.symbol);
      const futuresCoins = coinsData.filter((c: MonitoredCoin) => c.market_type === 'futures').map((c: MonitoredCoin) => c.symbol);
      
      setSpotSymbols([...new Set([...defaultSpotSymbols, ...spotCoins])]);
      setFuturesSymbols([...new Set([...defaultFuturesSymbols, ...futuresCoins])]);
    } catch (error) {
      console.error('Failed to load monitored coins:', error);
    }
  };

  useEffect(() => {
    loadMonitoredCoins();
  }, []);

  // 마켓 타입 변경 시 심볼 초기화
  useEffect(() => {
    const symbols = marketType === 'spot' ? spotSymbols : futuresSymbols;
    if (!symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0] || 'BTCUSDT');
    }
  }, [marketType, spotSymbols, futuresSymbols]);

  // 현재가 + 분석 조회
  const fetchData = async () => {
    setLoading(true);
    setAnalysisLoading(true);
    
    try {
      const [tickerRes, analysisRes] = await Promise.all([
        marketApi.getTicker(selectedSymbol),
        aiApi.combinedAnalysis(selectedSymbol).catch((e) => {
          console.warn('분석 조회 실패:', e?.message);
          return null;
        }),
      ]);

      // 현재가
      if (tickerRes?.data) {
        const tickerData = tickerRes.data.data || tickerRes.data;
        if (tickerData?.price !== undefined) {
          setCurrentPrice(tickerData.price);
          setPriceChange(tickerData.priceChangePercent || 0);
        }
      }

      // 분석
      if (analysisRes?.data) {
        setWeightedAnalysis(analysisRes.data);
      }
    } catch (error) {
      console.error('데이터 조회 실패:', error);
    } finally {
      setLoading(false);
      setAnalysisLoading(false);
    }
  };

  useEffect(() => {
    setCurrentPrice(0);
    setPriceChange(0);
    setWeightedAnalysis(null);
    fetchData();
    
    const interval = setInterval(fetchData, 15000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  // 심볼 검색
  const searchSymbols = async (query: string) => {
    if (!query.trim()) {
      setSearchResults([]);
      return;
    }
    
    setSearchLoading(true);
    try {
      const endpoint = marketType === 'futures' 
        ? `/v1/coins/search/futures?query=${query}&limit=20`
        : `/v1/coins/search/spot?query=${query}&limit=20`;
      
      const response = await apiClient.get(endpoint);
      if (response.data.success) {
        setSearchResults(response.data.symbols || []);
      }
    } catch (error) {
      console.error('심볼 검색 실패:', error);
    } finally {
      setSearchLoading(false);
    }
  };

  useEffect(() => {
    const timer = setTimeout(() => {
      if (showAddModal && searchSymbol) {
        searchSymbols(searchSymbol);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [searchSymbol, showAddModal, marketType]);

  // 코인 추가
  const handleAddCoin = async (symbol: string) => {
    setAddLoading(true);
    try {
      await apiClient.post(`/v1/coins/add-monitoring/${symbol}?market_type=${marketType}`);
      setShowAddModal(false);
      setSearchSymbol('');
      setSearchResults([]);
      loadMonitoredCoins();
    } catch (error: any) {
      console.error('코인 추가 실패:', error);
      alert(error.response?.data?.detail || '코인 추가 실패');
    } finally {
      setAddLoading(false);
    }
  };

  // 코인 삭제
  const handleDeleteCoin = async (coinId: number, symbol: string) => {
    if (!confirm(`${symbol}을 삭제하시겠습니까?`)) return;
    
    try {
      await apiClient.delete(`/v1/coins/${coinId}`);
      loadMonitoredCoins();
    } catch (error) {
      console.error('코인 삭제 실패:', error);
    }
  };

  // 해당 심볼이 모니터링 코인인지 확인
  const getMonitoredCoin = (symbol: string) => {
    return monitoredCoins.find((c) => c.symbol === symbol && c.market_type === marketType);
  };

  const isDefaultSymbol = (symbol: string) => {
    return marketType === 'spot' 
      ? defaultSpotSymbols.includes(symbol)
      : defaultFuturesSymbols.includes(symbol);
  };

  const currentSymbols = marketType === 'spot' ? spotSymbols : futuresSymbols;
  const currentMarketCoins = monitoredCoins.filter((c) => c.market_type === marketType);

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <BarChart3 className="w-7 h-7 text-blue-600" />
            코인 분석
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            기술적 지표와 AI 분석으로 암호화폐를 분석하세요
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => setShowAddModal(true)}
            variant="default"
            size="sm"
          >
            <Plus className="w-4 h-4 mr-2" />
            코인 추가
          </Button>
          <Button
            onClick={fetchData}
            variant="outline"
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={cn('w-4 h-4 mr-2', loading && 'animate-spin')} />
            새로고침
          </Button>
        </div>
      </div>

      {/* 마켓 타입 선택 */}
      <div className="flex gap-2">
        <Button
          variant={marketType === 'spot' ? 'default' : 'outline'}
          onClick={() => setMarketType('spot')}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'spot' && 'bg-blue-600 hover:bg-blue-700'
          )}
        >
          <Coins className="w-5 h-5 mr-2" />
          현물 (Spot)
        </Button>
        <Button
          variant={marketType === 'futures' ? 'default' : 'outline'}
          onClick={() => setMarketType('futures')}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'futures' && 'bg-orange-500 hover:bg-orange-600'
          )}
        >
          <LineChart className="w-5 h-5 mr-2" />
          선물 (Futures)
        </Button>
      </div>

      {/* 심볼 선택 */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-2">
            {currentSymbols.map((symbol) => {
              const monitoredCoin = getMonitoredCoin(symbol);
              const isDefault = isDefaultSymbol(symbol);
              
              return (
                <div
                  key={symbol}
                  className={cn(
                    'flex items-center gap-1 rounded-lg border transition-colors',
                    selectedSymbol === symbol 
                      ? marketType === 'spot'
                        ? 'bg-blue-600 text-white border-blue-600'
                        : 'bg-orange-500 text-white border-orange-500'
                      : 'bg-background hover:bg-accent border-border'
                  )}
                >
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSelectedSymbol(symbol)}
                    className={cn(
                      'min-w-[70px] h-9',
                      selectedSymbol === symbol && 'text-white hover:bg-transparent hover:text-white'
                    )}
                  >
                    {symbol.replace('USDT', '')}
                  </Button>
                  {monitoredCoin && !isDefault && (
                    <Button
                      variant="ghost"
                      size="icon"
                      className={cn(
                        'h-7 w-7 mr-1',
                        selectedSymbol === symbol 
                          ? 'hover:bg-white/20 text-white' 
                          : 'hover:bg-red-100 text-red-600'
                      )}
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteCoin(monitoredCoin.id, symbol);
                      }}
                    >
                      <X className="w-3 h-3" />
                    </Button>
                  )}
                </div>
              );
            })}
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 메인 컨텐츠 */}
        <div className="lg:col-span-2 space-y-6">
          {/* 현재가 */}
          <Card>
            <CardContent className="p-4">
              <div className="flex items-center justify-between">
                <div>
                  <div className="text-sm text-muted-foreground flex items-center gap-2">
                    {selectedSymbol}
                    <Badge variant="outline" className={marketType === 'spot' ? 'text-blue-600' : 'text-orange-500'}>
                      {marketType === 'spot' ? '현물' : '선물'}
                    </Badge>
                  </div>
                  <div className="text-3xl font-bold">
                    ${currentPrice.toLocaleString(undefined, { maximumFractionDigits: 4 })}
                  </div>
                </div>
                <Badge
                  variant={priceChange >= 0 ? 'default' : 'destructive'}
                  className="text-lg py-1 px-3"
                >
                  {priceChange >= 0 ? '+' : ''}{priceChange.toFixed(2)}%
                </Badge>
              </div>
            </CardContent>
          </Card>

          {/* 차트 */}
          <PriceChart symbol={selectedSymbol} />
        </div>

        {/* 분석 패널 */}
        <div className="space-y-6">
          {/* 기술적 지표 분석 */}
          <Card className={cn(
            'border-2',
            marketType === 'spot' ? 'border-blue-500/30' : 'border-orange-500/30'
          )}>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center justify-between">
                <span className="flex items-center gap-2">
                  <Sparkles className={marketType === 'spot' ? 'w-4 h-4 text-blue-600' : 'w-4 h-4 text-orange-500'} />
                  기술적 지표 분석
                </span>
                {analysisLoading && (
                  <Loader className="w-4 h-4 animate-spin text-muted-foreground" />
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {weightedAnalysis ? (
                <>
                  {/* 최종 신호 */}
                  <div className="flex items-center justify-between">
                    <span className="font-semibold">최종 신호</span>
                    <Badge
                      variant={
                        weightedAnalysis.final_signal === 'BUY'
                          ? 'default'
                          : weightedAnalysis.final_signal === 'SELL'
                          ? 'destructive'
                          : 'secondary'
                      }
                      className="text-base py-1 px-3"
                    >
                      {weightedAnalysis.final_signal}
                    </Badge>
                  </div>

                  {/* 신뢰도 */}
                  <div>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm">신뢰도</span>
                      <span className="text-sm font-semibold">
                        {(weightedAnalysis.final_confidence * 100).toFixed(1)}%
                      </span>
                    </div>
                    <Progress
                      value={weightedAnalysis.final_confidence * 100}
                      className="h-2"
                    />
                  </div>

                  {/* AI 예측 */}
                  {weightedAnalysis.ai_prediction && (
                    <div className="pt-2 border-t">
                      <div className="text-xs font-semibold text-muted-foreground mb-2">
                        AI 예측
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">신호:</span>
                          <Badge variant="outline" className="ml-1">
                            {weightedAnalysis.ai_prediction.signal}
                          </Badge>
                        </div>
                        <div>
                          <span className="text-muted-foreground">신뢰도:</span>
                          <span className="ml-1 font-semibold">
                            {(weightedAnalysis.ai_prediction.confidence * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* 기술적 지표 */}
                  {weightedAnalysis.weighted_signal && (
                    <div className="pt-2 border-t">
                      <div className="text-xs font-semibold text-muted-foreground mb-2">
                        기술적 지표
                      </div>
                      <div className="grid grid-cols-2 gap-2 text-sm">
                        <div>
                          <span className="text-muted-foreground">신호:</span>
                          <Badge variant="outline" className="ml-1">
                            {weightedAnalysis.weighted_signal.signal}
                          </Badge>
                        </div>
                        <div>
                          <span className="text-muted-foreground">점수:</span>
                          <span
                            className={cn(
                              'ml-1 font-semibold',
                              weightedAnalysis.weighted_signal.score > 0
                                ? 'text-green-600'
                                : weightedAnalysis.weighted_signal.score < 0
                                ? 'text-red-600'
                                : 'text-gray-600'
                            )}
                          >
                            {weightedAnalysis.weighted_signal.score.toFixed(2)}
                          </span>
                        </div>
                      </div>

                      {/* 개별 지표 점수 */}
                      {weightedAnalysis.weighted_signal.indicators && 
                        Object.keys(weightedAnalysis.weighted_signal.indicators).length > 0 && (
                        <div className="mt-3 pt-2 border-t border-dashed">
                          <div className="text-xs font-semibold text-muted-foreground mb-2">
                            개별 지표 점수
                          </div>
                          <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                            {Object.entries(weightedAnalysis.weighted_signal.indicators).map(([key, value]) => (
                              <div key={key} className="flex justify-between">
                                <span className="text-muted-foreground">
                                  {key.replace('_score', '').toUpperCase()}:
                                </span>
                                <span
                                  className={cn(
                                    'font-mono',
                                    Number(value) > 0.3 ? 'text-green-600' :
                                    Number(value) < -0.3 ? 'text-red-600' : 'text-gray-500'
                                  )}
                                >
                                  {Number(value).toFixed(2)}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}

                      {weightedAnalysis.weighted_signal.recommendation && (
                        <p className="text-xs text-muted-foreground mt-2 italic">
                          {weightedAnalysis.weighted_signal.recommendation}
                        </p>
                      )}
                    </div>
                  )}
                </>
              ) : (
                <div className="text-center py-8 text-muted-foreground">
                  {analysisLoading ? (
                    <div className="flex flex-col items-center gap-2">
                      <Loader className="w-8 h-8 animate-spin" />
                      <span>분석 로딩 중...</span>
                    </div>
                  ) : (
                    <span>분석 데이터 없음</span>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* 모니터링 코인 목록 */}
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
                {marketType === 'spot' ? (
                  <Coins className="w-4 h-4 text-blue-600" />
                ) : (
                  <LineChart className="w-4 h-4 text-orange-500" />
                )}
                모니터링 중인 {marketType === 'spot' ? '현물' : '선물'} 코인
              </CardTitle>
            </CardHeader>
            <CardContent>
              {currentMarketCoins.length === 0 ? (
                <div className="text-center py-4 text-muted-foreground text-sm">
                  모니터링 중인 코인이 없습니다
                </div>
              ) : (
                <div className="space-y-2 max-h-[300px] overflow-y-auto">
                  {currentMarketCoins.map((coin) => (
                    <div
                      key={coin.id}
                      className={cn(
                        'p-2 rounded-lg border flex items-center justify-between cursor-pointer hover:bg-accent transition-colors',
                        selectedSymbol === coin.symbol && (
                          marketType === 'spot' 
                            ? 'bg-blue-500/10 border-blue-500' 
                            : 'bg-orange-500/10 border-orange-500'
                        )
                      )}
                      onClick={() => setSelectedSymbol(coin.symbol)}
                    >
                      <div className="flex items-center gap-2">
                        <div>
                          <div className="font-medium text-sm">{coin.symbol}</div>
                          <div className="text-xs text-muted-foreground">
                            {coin.candle_count} 캔들
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="font-mono text-sm">
                          ${coin.current_price?.toLocaleString(undefined, { maximumFractionDigits: 4 }) || '-'}
                        </div>
                        <div className={cn(
                          'text-xs',
                          (coin.price_change_24h || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {(coin.price_change_24h || 0) >= 0 ? '+' : ''}{(coin.price_change_24h || 0).toFixed(2)}%
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* 코인 추가 모달 */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <Card className="w-full max-w-md">
            <CardHeader>
              <CardTitle className="flex items-center justify-between">
                <span>코인 추가</span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => {
                    setShowAddModal(false);
                    setSearchSymbol('');
                    setSearchResults([]);
                  }}
                >
                  <X className="w-4 h-4" />
                </Button>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* 마켓 타입 선택 */}
              <div className="flex gap-2">
                <Button
                  variant={marketType === 'spot' ? 'default' : 'outline'}
                  className={cn('flex-1', marketType === 'spot' && 'bg-blue-600 hover:bg-blue-700')}
                  onClick={() => {
                    setMarketType('spot');
                    setSearchResults([]);
                  }}
                >
                  <Coins className="w-4 h-4 mr-2" />
                  현물
                </Button>
                <Button
                  variant={marketType === 'futures' ? 'default' : 'outline'}
                  className={cn('flex-1', marketType === 'futures' && 'bg-orange-500 hover:bg-orange-600')}
                  onClick={() => {
                    setMarketType('futures');
                    setSearchResults([]);
                  }}
                >
                  <LineChart className="w-4 h-4 mr-2" />
                  선물
                </Button>
              </div>

              {/* 검색 */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="심볼 검색 (예: BTC, ETH)"
                  value={searchSymbol}
                  onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                  className="pl-10"
                />
              </div>

              {/* 검색 결과 */}
              {searchLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Loader className="w-6 h-6 animate-spin" />
                </div>
              ) : searchResults.length > 0 ? (
                <div className="max-h-[300px] overflow-y-auto space-y-1">
                  {searchResults.map((result) => (
                    <Button
                      key={result.symbol}
                      variant="ghost"
                      className="w-full justify-between"
                      disabled={addLoading}
                      onClick={() => handleAddCoin(result.symbol)}
                    >
                      <span className="font-medium">{result.symbol}</span>
                      <div className="flex items-center gap-2">
                        <span className="text-sm text-muted-foreground">
                          ${result.price?.toLocaleString()}
                        </span>
                        <span className={cn(
                          'text-xs',
                          result.priceChangePercent >= 0 ? 'text-green-600' : 'text-red-600'
                        )}>
                          {result.priceChangePercent >= 0 ? '+' : ''}{result.priceChangePercent?.toFixed(2)}%
                        </span>
                        <ChevronRight className="w-4 h-4" />
                      </div>
                    </Button>
                  ))}
                </div>
              ) : searchSymbol && !searchLoading ? (
                <div className="text-center py-4 text-muted-foreground">
                  검색 결과가 없습니다
                </div>
              ) : null}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
