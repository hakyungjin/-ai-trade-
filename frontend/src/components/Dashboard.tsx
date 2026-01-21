import { useEffect, useState } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { tradingApi, aiApi } from '../api/client';
import { TrendingUp, TrendingDown, Minus, RefreshCw, Wallet, BarChart3, PiggyBank } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { cn } from '@/lib/utils';
import { SpotTrading } from './trading/SpotTrading';
import { FuturesTrading } from './trading/FuturesTrading';

export function Dashboard() {
  const {
    balances,
    positions,
    currentSignal,
    selectedSymbol,
    setBalances,
    setPositions,
    setCurrentSignal,
    setSelectedSymbol,
  } = useTradingStore();

  const [currentPrice, setCurrentPrice] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [tradingMode, setTradingMode] = useState<'spot' | 'futures'>('spot');

  const symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'];

  const fetchData = async () => {
    setLoading(true);
    try {
      const [balanceRes, positionRes, priceRes, signalRes] = await Promise.all([
        tradingApi.getBalance(),
        tradingApi.getPositions(),
        tradingApi.getPrice(selectedSymbol),
        aiApi.predict(selectedSymbol),
      ]);

      setBalances(balanceRes.data);
      setPositions(positionRes.data);
      setCurrentPrice(priceRes.data.price);
      setCurrentSignal(signalRes.data);
    } catch (error) {
      console.error('데이터 조회 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000);
    return () => clearInterval(interval);
  }, [selectedSymbol]);

  const getSignalIcon = () => {
    if (!currentSignal) return <Minus className="w-6 h-6 text-muted-foreground" />;
    switch (currentSignal.signal) {
      case 'BUY':
        return <TrendingUp className="w-6 h-6 text-green-500" />;
      case 'SELL':
        return <TrendingDown className="w-6 h-6 text-red-500" />;
      default:
        return <Minus className="w-6 h-6 text-muted-foreground" />;
    }
  };

  const totalValue = positions.reduce((sum, p) => sum + p.value_usdt, 0);
  const usdtBalance = balances.find((b) => b.asset === 'USDT')?.total || 0;

  return (
    <div className="p-4 md:p-6 space-y-6">
      {/* Header & Trading Mode Toggle */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h1 className="text-2xl font-bold">대시보드</h1>
        <div className="flex items-center gap-3">
          {/* Trading Mode Toggle */}
          <Tabs value={tradingMode} onValueChange={(value: any) => setTradingMode(value)}>
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="spot" className="data-[state=active]:bg-blue-500 data-[state=active]:text-white">
                현물 (Spot)
              </TabsTrigger>
              <TabsTrigger value="futures" className="data-[state=active]:bg-orange-500 data-[state=active]:text-white">
                선물 (Futures)
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <Button
            onClick={fetchData}
            disabled={loading}
            variant="outline"
            className="min-h-[44px] gap-2"
          >
            <RefreshCw className={cn('w-4 h-4', loading && 'animate-spin')} />
            새로고침
          </Button>
        </div>
      </div>

      {/* Symbol Selector */}
      <Tabs value={selectedSymbol} onValueChange={setSelectedSymbol} className="w-full">
        <TabsList className="w-full h-auto flex-wrap justify-start gap-1 bg-transparent p-0">
          {symbols.map((symbol) => (
            <TabsTrigger
              key={symbol}
              value={symbol}
              className="min-h-[44px] px-4 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              {symbol.replace('USDT', '')}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 md:gap-4">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <BarChart3 className="w-4 h-4" />
              현재가
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl md:text-2xl font-bold">
              ${currentPrice.toLocaleString()}
            </div>
            <Badge variant="outline" className="mt-1">{selectedSymbol}</Badge>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <Wallet className="w-4 h-4" />
              USDT 잔고
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl md:text-2xl font-bold">
              ${usdtBalance.toLocaleString()}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              포지션 가치
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl md:text-2xl font-bold">
              ${totalValue.toLocaleString()}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-medium text-muted-foreground flex items-center gap-2">
              <PiggyBank className="w-4 h-4" />
              총 자산
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-xl md:text-2xl font-bold text-primary">
              ${(usdtBalance + totalValue).toLocaleString()}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* AI Signal Card */}
      <Card className={cn(
        'border-2',
        currentSignal?.signal === 'BUY' && 'border-green-500/50 bg-green-500/5',
        currentSignal?.signal === 'SELL' && 'border-red-500/50 bg-red-500/5',
        (!currentSignal || currentSignal.signal === 'HOLD') && 'border-border'
      )}>
        <CardContent className="p-4 md:p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className={cn(
                'p-3 rounded-full',
                currentSignal?.signal === 'BUY' && 'bg-green-500/20',
                currentSignal?.signal === 'SELL' && 'bg-red-500/20',
                (!currentSignal || currentSignal.signal === 'HOLD') && 'bg-muted'
              )}>
                {getSignalIcon()}
              </div>
              <div>
                <div className="text-sm text-muted-foreground mb-1">AI 신호</div>
                <div className="flex items-center gap-2">
                  <span className={cn(
                    'text-2xl font-bold',
                    currentSignal?.signal === 'BUY' && 'text-green-500',
                    currentSignal?.signal === 'SELL' && 'text-red-500'
                  )}>
                    {currentSignal?.signal || 'HOLD'}
                  </span>
                  {currentSignal?.signal === 'BUY' && (
                    <Badge className="bg-green-500">매수</Badge>
                  )}
                  {currentSignal?.signal === 'SELL' && (
                    <Badge className="bg-red-500">매도</Badge>
                  )}
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="text-sm text-muted-foreground mb-1">신뢰도</div>
              <div className="text-xl font-bold">
                {currentSignal
                  ? `${(currentSignal.confidence * 100).toFixed(1)}%`
                  : '-'}
              </div>
              {currentSignal && (
                <Progress
                  value={currentSignal.confidence * 100}
                  className="w-24 h-2 mt-2"
                />
              )}
            </div>
          </div>
          {currentSignal?.analysis && (
            <div className="mt-4 p-3 bg-background/50 rounded-lg border">
              <div className="text-sm text-muted-foreground">{currentSignal.analysis}</div>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Positions */}
      <Card>
        <CardHeader>
          <CardTitle>보유 포지션</CardTitle>
        </CardHeader>
        <CardContent>
          {positions.length === 0 ? (
            <div className="text-muted-foreground text-center py-8">
              보유 포지션이 없습니다
            </div>
          ) : (
            <div className="overflow-x-auto">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>심볼</TableHead>
                    <TableHead>수량</TableHead>
                    <TableHead>현재가</TableHead>
                    <TableHead className="text-right">가치 (USDT)</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {positions.map((position) => (
                    <TableRow key={position.symbol}>
                      <TableCell className="font-medium">
                        <Badge variant="outline">{position.symbol}</Badge>
                      </TableCell>
                      <TableCell>{position.quantity.toFixed(6)}</TableCell>
                      <TableCell>${position.current_price.toLocaleString()}</TableCell>
                      <TableCell className="text-right font-medium">
                        ${position.value_usdt.toLocaleString()}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </div>
          )}
        </CardContent>
      </Card>

      {/* Trading Mode Specific Components */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2">
          {tradingMode === 'spot' ? <SpotTrading /> : <FuturesTrading />}
        </div>

        {/* Side Panel: Chart Preview & Info */}
        <div className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                {tradingMode === 'spot' ? (
                  <>
                    <Badge variant="outline" className="bg-blue-500/20 text-blue-600">현물</Badge>
                    기본 정보
                  </>
                ) : (
                  <>
                    <Badge variant="outline" className="bg-orange-500/20 text-orange-600">선물</Badge>
                    위험도 정보
                  </>
                )}
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {tradingMode === 'spot' ? (
                <>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">심볼</div>
                    <div className="font-semibold">{selectedSymbol}</div>
                  </div>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">현재가</div>
                    <div className="font-semibold text-lg">${currentPrice.toLocaleString()}</div>
                  </div>
                  <div className="p-3 bg-blue-500/10 border border-blue-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">거래 방식</div>
                    <div className="font-semibold">즉시 현물 거래</div>
                    <div className="text-xs text-muted-foreground mt-1">보유 자산 범위 내</div>
                  </div>
                </>
              ) : (
                <>
                  <div className="p-3 bg-orange-500/10 border border-orange-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">레버리지 주의</div>
                    <div className="font-semibold text-sm">1배 ~ 20배</div>
                    <div className="text-xs text-muted-foreground mt-1">높을수록 리스크 ↑</div>
                  </div>
                  <div className="p-3 bg-red-500/10 border border-red-500/20 rounded">
                    <div className="text-xs text-muted-foreground mb-1">청산 위험</div>
                    <div className="font-semibold text-sm">증증금 부족시</div>
                    <div className="text-xs text-muted-foreground mt-1">자동 포지션 청산</div>
                  </div>
                  <div className="p-3 bg-muted rounded">
                    <div className="text-xs text-muted-foreground mb-1">필수 설정</div>
                    <div className="font-semibold text-sm">스탑로스 & 익절</div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
