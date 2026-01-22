import { useEffect, useState } from 'react';
import { useTradingStore } from '../../store/tradingStore';
import { marketApi } from '../../api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Slider } from '@/components/ui/slider';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

export function FuturesTrading() {
  const { selectedSymbol } = useTradingStore();
  
  const [positionMode, setPositionMode] = useState<'LONG' | 'SHORT'>('LONG');
  const [leverage, setLeverage] = useState([5]);
  const [quantity, setQuantity] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [currentPrice, setCurrentPrice] = useState(0);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const res = await marketApi.getTicker(selectedSymbol);
        setCurrentPrice(res.data.lastPrice);
      } catch (error) {
        console.error('가격 조회 실패:', error);
      }
    };
    fetchPrice();
  }, [selectedSymbol]);

  const handleOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    setMessage({ type: 'success', text: '선물 거래 기능은 준비 중입니다. (API 미구현)' });
    setQuantity('');
    setStopLoss('');
    setTakeProfit('');
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Badge variant="outline" className="bg-orange-500/20 text-orange-600">선물(Futures)</Badge>
            거래
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {message && (
            <Alert variant={message.type === 'success' ? 'default' : 'destructive'}>
              <AlertCircle className="h-4 w-4" />
              <AlertDescription>{message.text}</AlertDescription>
            </Alert>
          )}

          {/* Position Mode */}
          <Tabs value={positionMode} onValueChange={(value: any) => setPositionMode(value)}>
            <TabsList className="w-full grid w-full grid-cols-2">
              <TabsTrigger value="LONG" className="data-[state=active]:bg-green-500 data-[state=active]:text-white">
                롱 (상승)
              </TabsTrigger>
              <TabsTrigger value="SHORT" className="data-[state=active]:bg-red-500 data-[state=active]:text-white">
                숏 (하락)
              </TabsTrigger>
            </TabsList>
          </Tabs>

          {/* Current Price */}
          <div className="bg-muted p-3 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1">현재가</div>
            <div className="text-2xl font-bold">${currentPrice.toLocaleString()}</div>
          </div>

          {/* Balance */}
          <div className="bg-muted p-3 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1">USDT 증거금</div>
            <div className="text-lg font-semibold">${usdtBalance.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground mt-1">사용 증거금: ${(parseFloat(quantity) * currentPrice / leverage[0]).toLocaleString() || 0}</div>
          </div>

          <form onSubmit={handleOrder} className="space-y-4">
            {/* Leverage Slider */}
            <div>
              <div className="flex justify-between items-center mb-2">
                <Label>레버리지</Label>
                <Badge className="bg-orange-500">{leverage[0]}배</Badge>
              </div>
              <Slider
                value={leverage}
                onValueChange={setLeverage}
                min={1}
                max={20}
                step={1}
                className="w-full"
              />
              <div className="text-xs text-muted-foreground mt-2 space-y-1">
                <p>• 높은 레버리지 = 높은 리스크</p>
                <p>• 권장: 1~5배</p>
                <p>• 주의: 증증금 부족시 포지션 청산</p>
              </div>
            </div>

            {/* Quantity */}
            <div>
              <Label htmlFor="quantity">수량</Label>
              <Input
                id="quantity"
                type="number"
                placeholder="거래 수량"
                value={quantity}
                onChange={(e) => setQuantity(e.target.value)}
                step="0.00001"
                required
              />
            </div>

            {/* Risk Calculation */}
            <div className="grid grid-cols-2 gap-3">
              <div className="bg-red-500/10 p-3 rounded-lg border border-red-500/20">
                <div className="text-xs text-muted-foreground mb-1">최대 손실 (5%)</div>
                <div className="text-lg font-semibold text-red-600">${maxLoss.toLocaleString()}</div>
              </div>
              <div className="bg-green-500/10 p-3 rounded-lg border border-green-500/20">
                <div className="text-xs text-muted-foreground mb-1">예상 수익 (10%)</div>
                <div className="text-lg font-semibold text-green-600">${maxGain.toLocaleString()}</div>
              </div>
            </div>

            {/* Stop Loss */}
            <div>
              <Label htmlFor="stopLoss">스탑로스 가격 (필수)</Label>
              <Input
                id="stopLoss"
                type="number"
                placeholder="손절 가격"
                value={stopLoss}
                onChange={(e) => setStopLoss(e.target.value)}
                step="0.01"
                required
              />
            </div>

            {/* Take Profit */}
            <div>
              <Label htmlFor="takeProfit">익절 가격 (필수)</Label>
              <Input
                id="takeProfit"
                type="number"
                placeholder="익절 가격"
                value={takeProfit}
                onChange={(e) => setTakeProfit(e.target.value)}
                step="0.01"
                required
              />
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={loading || !quantity || !stopLoss || !takeProfit}
              className={`w-full h-12 text-lg font-semibold ${
                positionMode === 'LONG'
                  ? 'bg-green-500 hover:bg-green-600'
                  : 'bg-red-500 hover:bg-red-600'
              }`}
            >
              {loading 
                ? '처리중...' 
                : `${positionMode === 'LONG' ? '롱' : '숏'} ${leverage[0]}배 오픈`}
            </Button>
          </form>

          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              선물거래는 높은 리스크를 가집니다. 필히 스탑로스와 익절을 설정하세요.
            </AlertDescription>
          </Alert>
        </CardContent>
      </Card>
    </div>
  );
}
