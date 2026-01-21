import { useState } from 'react';
import { useTradingStore } from '../store/tradingStore';
import { tradingApi } from '../api/client';
import { ArrowUpCircle, ArrowDownCircle, AlertCircle, CheckCircle2 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { cn } from '@/lib/utils';

export function TradingPanel() {
  const { selectedSymbol, settings, currentSignal } = useTradingStore();

  const [orderType, setOrderType] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [quantity, setQuantity] = useState<string>('');
  const [price, setPrice] = useState<string>('');
  const [stopLoss, setStopLoss] = useState<string>(
    (settings.default_stop_loss * 100).toString()
  );
  const [takeProfit, setTakeProfit] = useState<string>(
    (settings.default_take_profit * 100).toString()
  );
  const [useStopLoss, setUseStopLoss] = useState(true);
  const [useTakeProfit, setUseTakeProfit] = useState(true);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(
    null
  );

  const handleOrder = async (side: 'BUY' | 'SELL') => {
    if (!quantity || parseFloat(quantity) <= 0) {
      setMessage({ type: 'error', text: '수량을 입력해주세요' });
      return;
    }

    setLoading(true);
    setMessage(null);

    try {
      const order: Record<string, unknown> = {
        symbol: selectedSymbol,
        side,
        quantity: parseFloat(quantity),
        order_type: orderType,
      };

      if (orderType === 'LIMIT' && price) {
        order.price = parseFloat(price);
      }

      if (useStopLoss && stopLoss) {
        order.stop_loss = parseFloat(stopLoss) / 100;
      }

      if (useTakeProfit && takeProfit) {
        order.take_profit = parseFloat(takeProfit) / 100;
      }

      await tradingApi.createOrder(order as Parameters<typeof tradingApi.createOrder>[0]);
      setMessage({ type: 'success', text: `${side === 'BUY' ? '매수' : '매도'} 주문이 실행되었습니다` });
      setQuantity('');
      setPrice('');
    } catch (error: unknown) {
      const err = error as { response?: { data?: { detail?: string } } };
      setMessage({
        type: 'error',
        text: err.response?.data?.detail || '주문 실패',
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      <h1 className="text-2xl font-bold">매매</h1>

      {/* AI Recommendation */}
      {currentSignal && currentSignal.signal !== 'HOLD' && (
        <Alert className={cn(
          'border-2',
          currentSignal.signal === 'BUY' ? 'border-green-500/50 bg-green-500/10' : 'border-red-500/50 bg-red-500/10'
        )}>
          <AlertDescription className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {currentSignal.signal === 'BUY' ? (
                <ArrowUpCircle className="w-5 h-5 text-green-500" />
              ) : (
                <ArrowDownCircle className="w-5 h-5 text-red-500" />
              )}
              <span className="font-medium">
                AI 추천: {currentSignal.signal === 'BUY' ? '매수' : '매도'}
              </span>
            </div>
            <Badge variant="outline">
              신뢰도 {(currentSignal.confidence * 100).toFixed(1)}%
            </Badge>
          </AlertDescription>
        </Alert>
      )}

      {/* Order Type */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-medium text-muted-foreground">주문 유형</CardTitle>
        </CardHeader>
        <CardContent>
          <Tabs value={orderType} onValueChange={(v) => setOrderType(v as 'MARKET' | 'LIMIT')}>
            <TabsList className="w-full">
              <TabsTrigger value="MARKET" className="flex-1 min-h-[44px]">시장가</TabsTrigger>
              <TabsTrigger value="LIMIT" className="flex-1 min-h-[44px]">지정가</TabsTrigger>
            </TabsList>
          </Tabs>
        </CardContent>
      </Card>

      {/* Order Form */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2">
            <Badge variant="outline" className="text-base">{selectedSymbol}</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* Quantity */}
          <div className="space-y-2">
            <Label htmlFor="quantity">수량</Label>
            <Input
              id="quantity"
              type="number"
              value={quantity}
              onChange={(e) => setQuantity(e.target.value)}
              placeholder="0.00"
              className="h-12 text-lg"
            />
          </div>

          {/* Price (for LIMIT) */}
          {orderType === 'LIMIT' && (
            <div className="space-y-2">
              <Label htmlFor="price">가격 (USDT)</Label>
              <Input
                id="price"
                type="number"
                value={price}
                onChange={(e) => setPrice(e.target.value)}
                placeholder="0.00"
                className="h-12 text-lg"
              />
            </div>
          )}

          {/* Stop Loss */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/50">
            <div className="flex items-center gap-3">
              <Switch
                id="stopLoss"
                checked={useStopLoss}
                onCheckedChange={setUseStopLoss}
              />
              <Label htmlFor="stopLoss" className="cursor-pointer">스탑로스</Label>
            </div>
            {useStopLoss && (
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={stopLoss}
                  onChange={(e) => setStopLoss(e.target.value)}
                  className="w-20 h-10 text-center"
                />
                <span className="text-muted-foreground">%</span>
              </div>
            )}
          </div>

          {/* Take Profit */}
          <div className="flex items-center justify-between p-4 rounded-lg bg-secondary/50">
            <div className="flex items-center gap-3">
              <Switch
                id="takeProfit"
                checked={useTakeProfit}
                onCheckedChange={setUseTakeProfit}
              />
              <Label htmlFor="takeProfit" className="cursor-pointer">익절</Label>
            </div>
            {useTakeProfit && (
              <div className="flex items-center gap-2">
                <Input
                  type="number"
                  value={takeProfit}
                  onChange={(e) => setTakeProfit(e.target.value)}
                  className="w-20 h-10 text-center"
                />
                <span className="text-muted-foreground">%</span>
              </div>
            )}
          </div>

          {/* Message */}
          {message && (
            <Alert variant={message.type === 'error' ? 'destructive' : 'default'} className={cn(
              message.type === 'success' && 'border-green-500/50 bg-green-500/10 text-green-500'
            )}>
              {message.type === 'success' ? (
                <CheckCircle2 className="h-4 w-4" />
              ) : (
                <AlertCircle className="h-4 w-4" />
              )}
              <AlertDescription>{message.text}</AlertDescription>
            </Alert>
          )}

          {/* Order Buttons */}
          <div className="flex gap-4 pt-2">
            <Button
              onClick={() => handleOrder('BUY')}
              disabled={loading}
              className="flex-1 h-14 text-lg font-bold bg-green-600 hover:bg-green-700"
            >
              <ArrowUpCircle className="w-5 h-5 mr-2" />
              매수
            </Button>
            <Button
              onClick={() => handleOrder('SELL')}
              disabled={loading}
              variant="destructive"
              className="flex-1 h-14 text-lg font-bold"
            >
              <ArrowDownCircle className="w-5 h-5 mr-2" />
              매도
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
