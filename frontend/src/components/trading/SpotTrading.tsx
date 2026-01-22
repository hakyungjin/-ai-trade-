import { useEffect, useState } from 'react';
import { useTradingStore } from '../../store/tradingStore';
import { marketApi } from '../../api/client';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { AlertCircle } from 'lucide-react';
import { Alert, AlertDescription } from '@/components/ui/alert';

export function SpotTrading() {
  const { selectedSymbol } = useTradingStore();

  const [orderType, setOrderType] = useState<'BUY' | 'SELL'>('BUY');
  const [priceType, setPriceType] = useState<'MARKET' | 'LIMIT'>('MARKET');
  const [quantity, setQuantity] = useState('');
  const [price, setPrice] = useState('');
  const [stopLoss, setStopLoss] = useState('');
  const [takeProfit, setTakeProfit] = useState('');
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [currentPrice, setCurrentPrice] = useState(0);

  useEffect(() => {
    const fetchPrice = async () => {
      try {
        const res = await marketApi.getTicker(selectedSymbol);
        if (res.data.success) {
          setCurrentPrice(res.data.data.price);
        }
      } catch (error) {
        console.error('가격 조회 실패:', error);
      }
    };
    fetchPrice();
  }, [selectedSymbol]);

  const handleOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    try {
      // TODO: 실제 주문 API 연동 시 활성화
      setMessage({ type: 'success', text: '주문 기능은 API 키 설정 후 사용 가능합니다.' });
      setQuantity('');
      setPrice('');
      setStopLoss('');
      setTakeProfit('');
    } catch (error: any) {
      setMessage({ type: 'error', text: error.response?.data?.detail || '주문 생성 실패' });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Badge variant="outline" className="bg-blue-500/20 text-blue-600">현물(Spot)</Badge>
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

          {/* Order Type Tabs */}
          <Tabs value={orderType} onValueChange={(value: any) => setOrderType(value)}>
            <TabsList className="w-full grid w-full grid-cols-2">
              <TabsTrigger value="BUY" className="data-[state=active]:bg-green-500 data-[state=active]:text-white">
                매수
              </TabsTrigger>
              <TabsTrigger value="SELL" className="data-[state=active]:bg-red-500 data-[state=active]:text-white">
                매도
              </TabsTrigger>
            </TabsList>
          </Tabs>

          {/* Current Price */}
          <div className="bg-muted p-3 rounded-lg">
            <div className="text-sm text-muted-foreground mb-1">현재가</div>
            <div className="text-2xl font-bold">${currentPrice.toLocaleString()}</div>
          </div>

          <form onSubmit={handleOrder} className="space-y-4">
            {/* Price Type */}
            <div>
              <Label htmlFor="priceType">주문 종류</Label>
              <Select value={priceType} onValueChange={(value: any) => setPriceType(value)}>
                <SelectTrigger id="priceType">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="MARKET">시장가</SelectItem>
                  <SelectItem value="LIMIT">지정가</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Limit Price */}
            {priceType === 'LIMIT' && (
              <div>
                <Label htmlFor="price">가격</Label>
                <Input
                  id="price"
                  type="number"
                  placeholder="주문 가격"
                  value={price}
                  onChange={(e) => setPrice(e.target.value)}
                  step="0.01"
                  required
                />
              </div>
            )}

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

            {/* Stop Loss */}
            <div>
              <Label htmlFor="stopLoss">스탑로스 가격 (선택)</Label>
              <Input
                id="stopLoss"
                type="number"
                placeholder="손절 가격"
                value={stopLoss}
                onChange={(e) => setStopLoss(e.target.value)}
                step="0.01"
              />
            </div>

            {/* Take Profit */}
            <div>
              <Label htmlFor="takeProfit">익절 가격 (선택)</Label>
              <Input
                id="takeProfit"
                type="number"
                placeholder="익절 가격"
                value={takeProfit}
                onChange={(e) => setTakeProfit(e.target.value)}
                step="0.01"
              />
            </div>

            {/* Submit Button */}
            <Button
              type="submit"
              disabled={loading || !quantity}
              className={`w-full h-12 text-lg font-semibold ${
                orderType === 'BUY'
                  ? 'bg-green-500 hover:bg-green-600'
                  : 'bg-red-500 hover:bg-red-600'
              }`}
            >
              {loading ? '처리중...' : `${orderType === 'BUY' ? '매수' : '매도'} 주문`}
            </Button>
          </form>

          <div className="text-xs text-muted-foreground space-y-1">
            <p>• 시장가: 즉시 체결</p>
            <p>• 지정가: 설정 가격에서만 체결</p>
            <p>• 스탑로스/익절: 주문 후 설정</p>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
