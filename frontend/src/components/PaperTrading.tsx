import { useState, useEffect, useRef } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Wallet,
  RefreshCw,
  X,
  Target,
  Percent,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  RotateCcw,
  Coins,
  LineChart,
  Plus,
  Search,
  Loader,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Tabs, TabsList, TabsTrigger, TabsContent } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { cn } from '@/lib/utils';
import { marketApi, apiClient } from '@/api/client';
import { usePaperTradingStore } from '@/store/paperTradingStore';
import { useTradingStore } from '@/store/tradingStore';
import type { MarketType, PositionType, PaperPosition } from '@/store/paperTradingStore';
import { PriceChart } from './market';

interface CoinData {
  id: number;
  symbol: string;
  market_type: 'spot' | 'futures';
}

export function PaperTrading() {
  const store = usePaperTradingStore();
  const {
    balance = 10000,
    initialBalance = 10000,
    positions = [],
    trades = [],
    totalPnl = 0,
    winCount = 0,
    loseCount = 0,
    openFuturesPosition,
    buySpot,
    sellSpot,
    closePosition,
    resetAccount,
  } = store;

  // 전역 상태에서 선택된 심볼과 마켓 타입 가져오기
  const { 
    selectedSymbol, 
    setSelectedSymbol, 
    selectedMarketType: marketType, 
    setSelectedMarketType: setMarketType 
  } = useTradingStore();

  // 기본 코인
  const defaultSpotSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  const defaultFuturesSymbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];
  
  const [spotSymbols, setSpotSymbols] = useState<string[]>(defaultSpotSymbols);
  const [futuresSymbols, setFuturesSymbols] = useState<string[]>(defaultFuturesSymbols);
  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(false);

  // 거래 폼 상태
  const [positionType, setPositionType] = useState<PositionType>('LONG');
  const [quantity, setQuantity] = useState<string>('');
  const [leverage, setLeverage] = useState<number>(10);

  // UI 상태
  const [activeTab, setActiveTab] = useState<'trade' | 'positions' | 'history'>('trade');
  const [showResetConfirm, setShowResetConfirm] = useState(false);
  const [showTradeForm, setShowTradeForm] = useState(false);  // 선물 거래 폼 표시 여부
  const [spotTradeType, setSpotTradeType] = useState<'buy' | 'sell' | null>(null);  // 현물 거래 타입

  // 코인 추가 모달 상태
  const [monitoredCoins, setMonitoredCoins] = useState<CoinData[]>([]);
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
      const spotCoins = coinsData.filter((c: CoinData) => c.market_type === 'spot').map((c: CoinData) => c.symbol);
      const futuresCoins = coinsData.filter((c: CoinData) => c.market_type === 'futures').map((c: CoinData) => c.symbol);
      
      setSpotSymbols([...new Set([...defaultSpotSymbols, ...spotCoins])]);
      setFuturesSymbols([...new Set([...defaultFuturesSymbols, ...futuresCoins])]);
    } catch (error) {
      console.error('Failed to load monitored coins:', error);
    }
  };

  useEffect(() => {
    loadMonitoredCoins();
  }, []);

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
      await loadMonitoredCoins();
      setSelectedSymbol(symbol);
    } catch (error) {
      console.error('코인 추가 실패:', error);
    } finally {
      setAddLoading(false);
    }
  };

  // 코인 삭제
  const handleDeleteCoin = async (coinId: number, symbol: string) => {
    try {
      await apiClient.delete(`/v1/coins/monitoring/${coinId}`);
      await loadMonitoredCoins();
      
      // 삭제된 코인이 선택된 코인이면 첫 번째 코인으로 변경
      if (selectedSymbol === symbol) {
        const currentList = marketType === 'spot' ? spotSymbols : futuresSymbols;
        const remaining = currentList.filter(s => s !== symbol);
        if (remaining.length > 0) {
          setSelectedSymbol(remaining[0]);
        }
      }
    } catch (error) {
      console.error('코인 삭제 실패:', error);
    }
  };

  // 모니터링 코인 정보 가져오기
  const getMonitoredCoin = (symbol: string) => {
    return monitoredCoins.find(c => c.symbol === symbol && c.market_type === marketType);
  };

  // 기본 코인 여부 확인
  const isDefaultSymbol = (symbol: string) => {
    return marketType === 'spot' 
      ? defaultSpotSymbols.includes(symbol)
      : defaultFuturesSymbols.includes(symbol);
  };

  // 마켓 타입 변경 시 심볼 초기화
  useEffect(() => {
    const symbols = marketType === 'spot' ? spotSymbols : futuresSymbols;
    if (!symbols.includes(selectedSymbol)) {
      setSelectedSymbol(symbols[0] || 'BTCUSDT');
    }
  }, [marketType, spotSymbols, futuresSymbols]);

  // 요청 ID를 추적해서 레이스 컨디션 방지
  const priceRequestIdRef = useRef(0);

  // 현재가 조회
  const fetchPrice = async (symbol: string, market: MarketType) => {
    const currentRequestId = ++priceRequestIdRef.current;
    
    setLoading(true);
    try {
      const response = await marketApi.getTicker(symbol, market);
      
      // 요청 ID가 변경되었으면 결과 무시
      if (currentRequestId !== priceRequestIdRef.current) return;
      
      const tickerData = response?.data?.data || response?.data;
      if (tickerData?.price !== undefined) {
        setCurrentPrice(tickerData.price);
        setPriceChange(tickerData.priceChangePercent || 0);
      }
    } catch (error) {
      if (currentRequestId !== priceRequestIdRef.current) return;
      console.error('Failed to fetch price:', error);
    } finally {
      if (currentRequestId === priceRequestIdRef.current) {
        setLoading(false);
      }
    }
  };

  useEffect(() => {
    fetchPrice(selectedSymbol, marketType);
    const interval = setInterval(() => fetchPrice(selectedSymbol, marketType), 5000);
    return () => clearInterval(interval);
  }, [selectedSymbol, marketType]);

  // 포지션 필터링
  const openPositions = (positions || []).filter((p) => p.status === 'OPEN');
  const spotPositions = openPositions.filter((p) => p.marketType === 'spot');
  const futuresPositions = openPositions.filter((p) => p.marketType === 'futures' || !p.marketType);
  const currentMarketPositions = marketType === 'spot' ? spotPositions : futuresPositions;
  const currentSymbolPositions = currentMarketPositions.filter((p) => p.symbol === selectedSymbol);

  // 현물 보유량 (매도 가능 수량)
  const spotHolding = spotPositions.find((p) => p.symbol === selectedSymbol);
  const availableToSell = spotHolding?.quantity || 0;

  // 선물 거래 실행
  const handleOpenFuturesPosition = () => {
    const qty = parseFloat(quantity);
    console.log('🚀 Opening futures position:', { qty, currentPrice, selectedSymbol, positionType, leverage });
    
    if (isNaN(qty) || qty <= 0) {
      alert('수량을 입력해주세요');
      return;
    }
    if (currentPrice <= 0) {
      alert('가격 로딩 중입니다. 잠시 후 다시 시도해주세요');
      return;
    }

    const success = openFuturesPosition(selectedSymbol, positionType, currentPrice, qty, leverage);
    console.log('📊 Position opened:', success);
    if (success) {
      setQuantity('');
      setShowTradeForm(false);
      alert(`✅ ${positionType} 포지션 오픈 완료!`);
    } else {
      alert('❌ 잔고가 부족합니다');
    }
  };

  // 현물 매수
  const handleBuySpot = () => {
    const qty = parseFloat(quantity);
    console.log('🛒 Buying spot:', { qty, currentPrice, selectedSymbol });
    
    if (isNaN(qty) || qty <= 0) {
      alert('수량을 입력해주세요');
      return;
    }
    if (currentPrice <= 0) {
      alert('가격 로딩 중입니다');
      return;
    }

    const success = buySpot(selectedSymbol, currentPrice, qty);
    if (success) {
      setQuantity('');
      setSpotTradeType(null);
      alert(`✅ ${selectedSymbol} 매수 완료!`);
    } else {
      alert('❌ 잔고가 부족합니다');
    }
  };

  // 현물 매도
  const handleSellSpot = () => {
    const qty = parseFloat(quantity);
    console.log('💰 Selling spot:', { qty, currentPrice, selectedSymbol, availableToSell });
    
    if (isNaN(qty) || qty <= 0) {
      alert('수량을 입력해주세요');
      return;
    }
    if (currentPrice <= 0) {
      alert('가격 로딩 중입니다');
      return;
    }
    if (qty > availableToSell) {
      alert(`보유 수량(${availableToSell})을 초과했습니다`);
      return;
    }

    const success = sellSpot(selectedSymbol, currentPrice, qty);
    if (success) {
      setQuantity('');
      setSpotTradeType(null);
      alert(`✅ ${selectedSymbol} 매도 완료!`);
    } else {
      alert('❌ 매도 실패');
    }
  };

  // 포지션 종료
  const handleClosePosition = (positionId: string) => {
    closePosition(positionId, currentPrice);
  };

  // PnL 계산 (실시간)
  const calculateUnrealizedPnl = (position: PaperPosition) => {
    if (position.marketType === 'spot') {
      return (currentPrice - position.entryPrice) * position.quantity;
    }
    if (position.type === 'LONG') {
      return (currentPrice - position.entryPrice) * position.quantity * position.leverage;
    } else {
      return (position.entryPrice - currentPrice) * position.quantity * position.leverage;
    }
  };

  // 수량 빠른 입력
  const setQuickQuantity = (percent: number) => {
    if (currentPrice <= 0) return;
    
    if (marketType === 'futures') {
      const maxQuantity = (balance * leverage * percent) / 100 / currentPrice;
      setQuantity(maxQuantity.toFixed(6));
    } else {
      const maxQuantity = (balance * percent) / 100 / currentPrice;
      setQuantity(maxQuantity.toFixed(6));
    }
  };

  // 매도 수량 빠른 입력
  const setQuickSellQuantity = (percent: number) => {
    const sellQty = (availableToSell * percent) / 100;
    setQuantity(sellQty.toFixed(6));
  };

  // 계정 리셋
  const handleResetAccount = () => {
    resetAccount();
    setShowResetConfirm(false);
  };

  // 통계
  const totalTrades = winCount + loseCount;
  const winRate = totalTrades > 0 ? (winCount / totalTrades) * 100 : 0;

  // 필요 금액/마진
  const requiredAmount = currentPrice > 0 && quantity 
    ? marketType === 'futures'
      ? (currentPrice * parseFloat(quantity || '0')) / leverage
      : currentPrice * parseFloat(quantity || '0')
    : 0;

  const currentSymbols = marketType === 'spot' ? spotSymbols : futuresSymbols;

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* 헤더 */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Target className="w-7 h-7 text-emerald-600" />
            모의투자
          </h1>
          <p className="text-sm text-muted-foreground mt-1">
            실제 자금 없이 가상으로 암호화폐 거래를 연습하세요
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            onClick={() => fetchPrice(selectedSymbol, marketType)}
            variant="outline"
            size="sm"
            disabled={loading}
          >
            <RefreshCw className={cn('w-4 h-4 mr-2', loading && 'animate-spin')} />
            새로고침
          </Button>
          <Button
            onClick={() => setShowResetConfirm(true)}
            variant="outline"
            size="sm"
            className="text-red-600 hover:text-red-700"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            리셋
          </Button>
        </div>
      </div>

      {/* 계정 요약 */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <Wallet className="w-4 h-4" />
              잔고
            </div>
            <div className="text-xl font-bold">
              ${balance.toLocaleString(undefined, { maximumFractionDigits: 2 })}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <DollarSign className="w-4 h-4" />
              총 PnL
            </div>
            <div className={cn(
              'text-xl font-bold',
              totalPnl > 0 ? 'text-green-600' : totalPnl < 0 ? 'text-red-600' : ''
            )}>
              {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <Percent className="w-4 h-4" />
              수익률
            </div>
            <div className={cn(
              'text-xl font-bold',
              totalPnl > 0 ? 'text-green-600' : totalPnl < 0 ? 'text-red-600' : ''
            )}>
              {((balance - initialBalance) / initialBalance * 100).toFixed(2)}%
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <CheckCircle className="w-4 h-4" />
              승률
            </div>
            <div className="text-xl font-bold">
              {winRate.toFixed(1)}%
            </div>
            <div className="text-xs text-muted-foreground">
              {winCount}승 {loseCount}패
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardContent className="p-4">
            <div className="flex items-center gap-2 text-muted-foreground text-sm mb-1">
              <TrendingUp className="w-4 h-4" />
              열린 포지션
            </div>
            <div className="text-xl font-bold">
              {openPositions.length}
            </div>
            <div className="text-xs text-muted-foreground">
              현물 {spotPositions.length} / 선물 {futuresPositions.length}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* 마켓 타입 선택 */}
      <div className="flex gap-2">
        <Button
          variant={marketType === 'spot' ? 'default' : 'outline'}
          onClick={() => {
            setMarketType('spot');
            setShowTradeForm(false);
            setSpotTradeType(null);
            setQuantity('');
          }}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'spot' && 'bg-blue-600 hover:bg-blue-700'
          )}
        >
          <Coins className="w-5 h-5 mr-2" />
          현물 (Spot)
          {spotPositions.length > 0 && (
            <Badge variant="secondary" className="ml-2">{spotPositions.length}</Badge>
          )}
        </Button>
        <Button
          variant={marketType === 'futures' ? 'default' : 'outline'}
          onClick={() => {
            setMarketType('futures');
            setShowTradeForm(false);
            setSpotTradeType(null);
            setQuantity('');
          }}
          className={cn(
            'flex-1 sm:flex-none h-12',
            marketType === 'futures' && 'bg-orange-500 hover:bg-orange-600'
          )}
        >
          <LineChart className="w-5 h-5 mr-2" />
          선물 (Futures)
          {futuresPositions.length > 0 && (
            <Badge variant="secondary" className="ml-2">{futuresPositions.length}</Badge>
          )}
        </Button>
      </div>

      {/* 심볼 선택 */}
      <Card>
        <CardContent className="p-4">
          <div className="flex flex-wrap gap-2 items-center">
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
            <Button
              onClick={() => setShowAddModal(true)}
              variant="outline"
              size="sm"
              className="h-9"
            >
              <Plus className="w-4 h-4 mr-1" />
              추가
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 차트 & 거래 패널 */}
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
          <PriceChart symbol={selectedSymbol} marketType={marketType} />

          {/* 현재 심볼 포지션 */}
          {currentSymbolPositions.length > 0 && (
            <Card className={cn(
              'border-2',
              marketType === 'spot' ? 'border-blue-500/30' : 'border-orange-500/30'
            )}>
              <CardHeader className="pb-2">
                <CardTitle className="text-sm flex items-center gap-2">
                  <AlertTriangle className={marketType === 'spot' ? 'w-4 h-4 text-blue-500' : 'w-4 h-4 text-orange-500'} />
                  {selectedSymbol} 열린 포지션
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                {currentSymbolPositions.map((position) => {
                  const unrealizedPnl = calculateUnrealizedPnl(position);
                  const unrealizedPnlPercent = (unrealizedPnl / (position.entryPrice * position.quantity)) * 100;
                  
                  return (
                    <div key={position.id} className="p-3 bg-muted rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          {marketType === 'spot' ? (
                            <Badge variant="default" className="bg-blue-600">보유중</Badge>
                          ) : (
                            <Badge variant={position.type === 'LONG' ? 'default' : 'destructive'}>
                              {position.type}
                            </Badge>
                          )}
                          {position.leverage > 1 && (
                            <Badge variant="outline">{position.leverage}x</Badge>
                          )}
                        </div>
                        <Button
                          size="sm"
                          variant="outline"
                          className="text-red-600 hover:bg-red-50"
                          onClick={() => handleClosePosition(position.id)}
                        >
                          <X className="w-4 h-4 mr-1" />
                          {marketType === 'spot' ? '전량 매도' : '종료'}
                        </Button>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-sm">
                        <div>
                          <div className="text-muted-foreground">평균단가</div>
                          <div className="font-mono">${position.entryPrice.toLocaleString()}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">수량</div>
                          <div className="font-mono">{position.quantity}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">미실현 PnL</div>
                          <div className={cn(
                            'font-mono font-bold',
                            unrealizedPnl > 0 ? 'text-green-600' : unrealizedPnl < 0 ? 'text-red-600' : ''
                          )}>
                            {unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}
                            <span className="text-xs ml-1">({unrealizedPnlPercent.toFixed(2)}%)</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </CardContent>
            </Card>
          )}
        </div>

        {/* 사이드 패널 */}
        <div className="space-y-6">
          <Card>
            <Tabs value={activeTab} onValueChange={(v) => setActiveTab(v as any)}>
              <TabsList className="w-full grid grid-cols-3">
                <TabsTrigger value="trade">거래</TabsTrigger>
                <TabsTrigger value="positions">
                  포지션
                  {currentMarketPositions.length > 0 && (
                    <Badge variant="secondary" className="ml-1 text-xs">
                      {currentMarketPositions.length}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="history">내역</TabsTrigger>
              </TabsList>

              {/* 거래 탭 */}
              <TabsContent value="trade" className="p-4 space-y-4">
                {marketType === 'futures' ? (
                  /* 선물 거래 */
                  <>
                    {!showTradeForm ? (
                      /* 롱/숏 버튼만 표시 */
                      <div className="grid grid-cols-2 gap-3">
                        <Button
                          className="h-20 bg-green-600 hover:bg-green-700 text-white flex flex-col items-center justify-center gap-1"
                          onClick={() => {
                            setPositionType('LONG');
                            setShowTradeForm(true);
                          }}
                        >
                          <TrendingUp className="w-8 h-8" />
                          <span className="text-lg font-bold">롱 (매수)</span>
                        </Button>
                        <Button
                          className="h-20 bg-red-600 hover:bg-red-700 text-white flex flex-col items-center justify-center gap-1"
                          onClick={() => {
                            setPositionType('SHORT');
                            setShowTradeForm(true);
                          }}
                        >
                          <TrendingDown className="w-8 h-8" />
                          <span className="text-lg font-bold">숏 (매도)</span>
                        </Button>
                      </div>
                    ) : (
                      /* 거래 폼 */
                      <>
                        {/* 롱/숏 선택 (변경 가능) */}
                        <div className="flex items-center justify-between">
                          <div className="grid grid-cols-2 gap-2 flex-1">
                            <Button
                              variant={positionType === 'LONG' ? 'default' : 'outline'}
                              className={cn(
                                'h-10',
                                positionType === 'LONG' && 'bg-green-600 hover:bg-green-700'
                              )}
                              onClick={() => setPositionType('LONG')}
                            >
                              <TrendingUp className="w-4 h-4 mr-1" />
                              롱
                            </Button>
                            <Button
                              variant={positionType === 'SHORT' ? 'default' : 'outline'}
                              className={cn(
                                'h-10',
                                positionType === 'SHORT' && 'bg-red-600 hover:bg-red-700'
                              )}
                              onClick={() => setPositionType('SHORT')}
                            >
                              <TrendingDown className="w-4 h-4 mr-1" />
                              숏
                            </Button>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="ml-2"
                            onClick={() => {
                              setShowTradeForm(false);
                              setQuantity('');
                            }}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        </div>

                        {/* 레버리지 */}
                        <div>
                          <label className="text-sm font-medium mb-2 block">레버리지</label>
                          <Select
                            value={leverage.toString()}
                            onValueChange={(v) => setLeverage(parseInt(v))}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              {[1, 2, 3, 5, 10, 20, 50, 100].map((l) => (
                                <SelectItem key={l} value={l.toString()}>
                                  {l}x
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                        </div>

                        {/* 수량 */}
                        <div>
                          <label className="text-sm font-medium mb-2 block">수량</label>
                          <Input
                            type="text"
                            inputMode="decimal"
                            placeholder="0.00"
                            value={quantity}
                            onChange={(e) => {
                              // 숫자와 소수점만 허용
                              const val = e.target.value.replace(/[^0-9.]/g, '');
                              setQuantity(val);
                            }}
                          />
                          <div className="flex gap-1 mt-2">
                            {[25, 50, 75, 100].map((p) => (
                              <Button
                                key={p}
                                variant="outline"
                                size="sm"
                                className="flex-1"
                                onClick={() => setQuickQuantity(p)}
                              >
                                {p}%
                              </Button>
                            ))}
                          </div>
                        </div>

                        {/* 주문 정보 */}
                        <div className="p-3 bg-muted rounded-lg space-y-2 text-sm">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">예상 가격</span>
                            <span className="font-mono">${currentPrice.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">필요 마진</span>
                            <span className="font-mono">${requiredAmount.toFixed(2)}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">레버리지</span>
                            <span className="font-mono">{leverage}x</span>
                          </div>
                        </div>

                        {/* 주문 버튼 */}
                        <Button
                          className={cn(
                            'w-full h-12 text-lg font-bold',
                            positionType === 'LONG' 
                              ? 'bg-green-600 hover:bg-green-700' 
                              : 'bg-red-600 hover:bg-red-700'
                          )}
                          disabled={!quantity || parseFloat(quantity) <= 0 || requiredAmount > balance}
                          onClick={() => {
                            handleOpenFuturesPosition();
                            setShowTradeForm(false);
                            setQuantity('');
                          }}
                        >
                          {positionType === 'LONG' ? '롱 진입' : '숏 진입'}
                        </Button>

                        {requiredAmount > balance && (
                          <p className="text-sm text-red-500 text-center">
                            잔고가 부족합니다
                          </p>
                        )}
                      </>
                    )}
                  </>
                ) : (
                  /* 현물 거래 */
                  <>
                    {spotTradeType === null ? (
                      /* 매수/매도 버튼만 표시 */
                      <div className="grid grid-cols-2 gap-3">
                        <Button
                          className="h-20 bg-green-600 hover:bg-green-700 text-white flex flex-col items-center justify-center gap-1"
                          onClick={() => setSpotTradeType('buy')}
                        >
                          <TrendingUp className="w-8 h-8" />
                          <span className="text-lg font-bold">매수</span>
                        </Button>
                        <Button
                          className="h-20 bg-red-600 hover:bg-red-700 text-white flex flex-col items-center justify-center gap-1"
                          onClick={() => setSpotTradeType('sell')}
                          disabled={availableToSell <= 0}
                        >
                          <TrendingDown className="w-8 h-8" />
                          <span className="text-lg font-bold">매도</span>
                          {availableToSell > 0 && (
                            <span className="text-xs opacity-80">보유: {availableToSell.toFixed(4)}</span>
                          )}
                        </Button>
                      </div>
                    ) : spotTradeType === 'buy' ? (
                      /* 매수 폼 */
                      <div className="space-y-3 p-3 bg-green-500/5 rounded-lg border border-green-500/20">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-green-600 font-semibold">
                            <TrendingUp className="w-4 h-4" />
                            매수
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setSpotTradeType(null);
                              setQuantity('');
                            }}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        </div>
                        <Input
                          type="text"
                          inputMode="decimal"
                          placeholder="매수 수량"
                          value={quantity}
                          onChange={(e) => {
                            const val = e.target.value.replace(/[^0-9.]/g, '');
                            setQuantity(val);
                          }}
                        />
                        <div className="flex gap-1">
                          {[25, 50, 75, 100].map((p) => (
                            <Button
                              key={p}
                              variant="outline"
                              size="sm"
                              className="flex-1"
                              onClick={() => setQuickQuantity(p)}
                            >
                              {p}%
                            </Button>
                          ))}
                        </div>
                        <div className="p-2 bg-muted rounded text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">예상 가격</span>
                            <span className="font-mono">${currentPrice.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">예상 금액</span>
                            <span className="font-mono">${requiredAmount.toFixed(2)}</span>
                          </div>
                        </div>
                        <Button
                          className="w-full h-12 text-lg font-bold bg-green-600 hover:bg-green-700"
                          disabled={!quantity || parseFloat(quantity) <= 0 || requiredAmount > balance}
                          onClick={() => {
                            handleBuySpot();
                            setSpotTradeType(null);
                            setQuantity('');
                          }}
                        >
                          매수
                        </Button>
                        {requiredAmount > balance && (
                          <p className="text-sm text-red-500 text-center">
                            잔고가 부족합니다
                          </p>
                        )}
                      </div>
                    ) : (
                      /* 매도 폼 */
                      <div className="space-y-3 p-3 bg-red-500/5 rounded-lg border border-red-500/20">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2 text-red-600 font-semibold">
                            <TrendingDown className="w-4 h-4" />
                            매도
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm text-muted-foreground">
                              보유: {availableToSell.toFixed(6)}
                            </span>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => {
                                setSpotTradeType(null);
                                setQuantity('');
                              }}
                            >
                              <X className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                        <Input
                          type="text"
                          inputMode="decimal"
                          placeholder="매도 수량"
                          value={quantity}
                          onChange={(e) => {
                            const val = e.target.value.replace(/[^0-9.]/g, '');
                            setQuantity(val);
                          }}
                        />
                        <div className="flex gap-1">
                          {[25, 50, 75, 100].map((p) => (
                            <Button
                              key={p}
                              variant="outline"
                              size="sm"
                              className="flex-1"
                              onClick={() => setQuickSellQuantity(p)}
                            >
                              {p}%
                            </Button>
                          ))}
                        </div>
                        <div className="p-2 bg-muted rounded text-sm space-y-1">
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">예상 가격</span>
                            <span className="font-mono">${currentPrice.toLocaleString()}</span>
                          </div>
                          <div className="flex justify-between">
                            <span className="text-muted-foreground">예상 수익</span>
                            <span className="font-mono">${(parseFloat(quantity || '0') * currentPrice).toFixed(2)}</span>
                          </div>
                        </div>
                        <Button
                          className="w-full h-12 text-lg font-bold bg-red-600 hover:bg-red-700"
                          disabled={!quantity || parseFloat(quantity) <= 0 || parseFloat(quantity) > availableToSell}
                          onClick={() => {
                            handleSellSpot();
                            setSpotTradeType(null);
                            setQuantity('');
                          }}
                        >
                          매도
                        </Button>
                        {parseFloat(quantity || '0') > availableToSell && (
                          <p className="text-sm text-red-500 text-center">
                            보유 수량을 초과합니다
                          </p>
                        )}
                      </div>
                    )}
                  </>
                )}
              </TabsContent>

              {/* 포지션 탭 */}
              <TabsContent value="positions" className="p-4">
                {currentMarketPositions.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    열린 포지션이 없습니다
                  </div>
                ) : (
                  <div className="space-y-3 max-h-[400px] overflow-y-auto">
                    {currentMarketPositions.map((position) => {
                      const unrealizedPnl = calculateUnrealizedPnl(position);
                      const unrealizedPnlPercent = (unrealizedPnl / (position.entryPrice * position.quantity)) * 100;
                      
                      return (
                        <div key={position.id} className="p-3 border rounded-lg">
                          <div className="flex items-center justify-between mb-2">
                            <div className="font-semibold">{position.symbol}</div>
                            <div className="flex items-center gap-2">
                              {marketType === 'spot' ? (
                                <Badge className="bg-blue-600">보유</Badge>
                              ) : (
                                <Badge variant={position.type === 'LONG' ? 'default' : 'destructive'}>
                                  {position.type}
                                </Badge>
                              )}
                              {position.leverage > 1 && (
                                <Badge variant="outline">{position.leverage}x</Badge>
                              )}
                            </div>
                          </div>
                          <div className="text-sm text-muted-foreground mb-2">
                            평균단가: ${position.entryPrice.toLocaleString()} | 수량: {position.quantity}
                          </div>
                          <div className="flex items-center justify-between">
                            <div className={cn(
                              'font-bold',
                              unrealizedPnl > 0 ? 'text-green-600' : unrealizedPnl < 0 ? 'text-red-600' : ''
                            )}>
                              {unrealizedPnl >= 0 ? '+' : ''}${unrealizedPnl.toFixed(2)}
                              <span className="text-sm ml-1">({unrealizedPnlPercent.toFixed(2)}%)</span>
                            </div>
                            <Button
                              size="sm"
                              variant="outline"
                              className="text-red-600"
                              onClick={() => handleClosePosition(position.id)}
                            >
                              {marketType === 'spot' ? '매도' : '종료'}
                            </Button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </TabsContent>

              {/* 내역 탭 */}
              <TabsContent value="history" className="p-4">
                {trades.length === 0 ? (
                  <div className="text-center py-8 text-muted-foreground">
                    거래 내역이 없습니다
                  </div>
                ) : (
                  <div className="space-y-2 max-h-[400px] overflow-y-auto">
                    {[...trades]
                      .filter((t) => marketType === 'futures' ? (t.marketType === 'futures' || !t.marketType) : t.marketType === 'spot')
                      .reverse()
                      .slice(0, 20)
                      .map((trade) => (
                        <div key={trade.id} className="p-2 border rounded text-sm">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              {trade.marketType === 'spot' ? (
                                <Badge className={trade.side === 'BUY' ? 'bg-green-600' : 'bg-red-600'} style={{fontSize: '10px'}}>
                                  {trade.side}
                                </Badge>
                              ) : (
                                <Badge variant={trade.type === 'LONG' ? 'default' : 'destructive'} className="text-xs">
                                  {trade.type}
                                </Badge>
                              )}
                              <span className="font-medium">{trade.symbol}</span>
                              <Badge variant="outline" className="text-xs">
                                {trade.action}
                              </Badge>
                            </div>
                            {trade.pnl !== undefined && (
                              <span className={cn(
                                'font-bold',
                                trade.pnl > 0 ? 'text-green-600' : 'text-red-600'
                              )}>
                                {trade.pnl >= 0 ? '+' : ''}${trade.pnl.toFixed(2)}
                              </span>
                            )}
                          </div>
                          <div className="text-muted-foreground mt-1">
                            ${trade.price.toLocaleString()} x {trade.quantity}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            {new Date(trade.timestamp).toLocaleString()}
                          </div>
                        </div>
                      ))}
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </Card>
        </div>
      </div>

      {/* 리셋 확인 모달 */}
      {showResetConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-red-600">
                <AlertTriangle className="w-5 h-5" />
                계정 리셋
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <p>
                정말 계정을 리셋하시겠습니까?
                <br />
                모든 포지션과 거래 내역이 삭제되고 잔고가 $10,000로 초기화됩니다.
              </p>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => setShowResetConfirm(false)}
                >
                  취소
                </Button>
                <Button
                  variant="destructive"
                  className="flex-1"
                  onClick={handleResetAccount}
                >
                  리셋
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* 코인 추가 모달 */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <Card className="w-full max-w-md mx-4 max-h-[80vh] overflow-hidden flex flex-col">
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="flex items-center gap-2">
                  <Plus className="w-5 h-5" />
                  {marketType === 'spot' ? '현물' : '선물'} 코인 추가
                </CardTitle>
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
              </div>
            </CardHeader>
            <CardContent className="space-y-4 overflow-y-auto">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="심볼 검색 (예: BTC, ETH, BEAT...)"
                  value={searchSymbol}
                  onChange={(e) => setSearchSymbol(e.target.value.toUpperCase())}
                  className="pl-9"
                  autoFocus
                />
              </div>

              {searchLoading && (
                <div className="flex items-center justify-center py-8">
                  <Loader className="w-6 h-6 animate-spin text-muted-foreground" />
                </div>
              )}

              {!searchLoading && searchResults.length > 0 && (
                <div className="space-y-1 max-h-[300px] overflow-y-auto">
                  {searchResults.map((result) => (
                    <Button
                      key={result.symbol}
                      variant="ghost"
                      className="w-full justify-between h-auto py-3"
                      onClick={() => handleAddCoin(result.symbol)}
                      disabled={addLoading || currentSymbols.includes(result.symbol)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="text-left">
                          <div className="font-medium">{result.symbol}</div>
                          <div className="text-xs text-muted-foreground">
                            {result.baseAsset}/{result.quoteAsset}
                          </div>
                        </div>
                      </div>
                      {currentSymbols.includes(result.symbol) ? (
                        <Badge variant="secondary">추가됨</Badge>
                      ) : (
                        <Plus className="w-4 h-4" />
                      )}
                    </Button>
                  ))}
                </div>
              )}

              {!searchLoading && searchSymbol && searchResults.length === 0 && (
                <div className="text-center py-8 text-muted-foreground">
                  검색 결과가 없습니다
                </div>
              )}

              {!searchSymbol && (
                <div className="text-center py-8 text-muted-foreground text-sm">
                  심볼을 검색해서 코인을 추가하세요
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  );
}
