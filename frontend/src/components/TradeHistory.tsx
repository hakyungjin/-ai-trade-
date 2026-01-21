import { useEffect, useState } from 'react';
import { tradingApi } from '../api/client';
import { RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';

interface Trade {
  symbol: string;
  id: number;
  order_id: number;
  price: number;
  quantity: number;
  commission: number;
  time: number;
  is_buyer: boolean;
}

export function TradeHistory() {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState<string>('전체');

  const symbols = ['전체', 'BTCUSDT', 'ETHUSDT', 'BNBUSDT'];

  const fetchTrades = async () => {
    setLoading(true);
    try {
      const res = await tradingApi.getHistory(
        selectedSymbol === '전체' ? undefined : selectedSymbol,
        50
      );
      setTrades(res.data);
    } catch (error) {
      console.error('거래 내역 조회 실패:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrades();
  }, [selectedSymbol]);

  const formatDate = (timestamp: number) => {
    return new Date(timestamp).toLocaleString('ko-KR');
  };

  return (
    <div className="p-4 md:p-6 space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <h1 className="text-2xl font-bold">거래 내역</h1>
        <Button
          onClick={fetchTrades}
          disabled={loading}
          variant="outline"
          className="min-h-[44px] gap-2"
        >
          <RefreshCw className={cn('w-4 h-4', loading && 'animate-spin')} />
          새로고침
        </Button>
      </div>

      {/* Symbol Filter */}
      <Tabs value={selectedSymbol} onValueChange={setSelectedSymbol} className="w-full">
        <TabsList className="w-full h-auto flex-wrap justify-start gap-1 bg-transparent p-0">
          {symbols.map((symbol) => (
            <TabsTrigger
              key={symbol}
              value={symbol}
              className="min-h-[44px] px-4 data-[state=active]:bg-primary data-[state=active]:text-primary-foreground"
            >
              {symbol === '전체' ? symbol : symbol.replace('USDT', '')}
            </TabsTrigger>
          ))}
        </TabsList>
      </Tabs>

      {/* Trade Table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">최근 거래</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>시간</TableHead>
                  <TableHead>심볼</TableHead>
                  <TableHead>유형</TableHead>
                  <TableHead>가격</TableHead>
                  <TableHead className="hidden sm:table-cell">수량</TableHead>
                  <TableHead className="text-right">총액</TableHead>
                  <TableHead className="hidden md:table-cell text-right">수수료</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {trades.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} className="h-32 text-center text-muted-foreground">
                      거래 내역이 없습니다
                    </TableCell>
                  </TableRow>
                ) : (
                  trades.map((trade) => (
                    <TableRow key={trade.id}>
                      <TableCell className="text-sm text-muted-foreground whitespace-nowrap">
                        {formatDate(trade.time)}
                      </TableCell>
                      <TableCell>
                        <Badge variant="outline">{trade.symbol.replace('USDT', '')}</Badge>
                      </TableCell>
                      <TableCell>
                        <Badge
                          className={cn(
                            trade.is_buyer
                              ? 'bg-green-500/20 text-green-500 hover:bg-green-500/30'
                              : 'bg-red-500/20 text-red-500 hover:bg-red-500/30'
                          )}
                        >
                          {trade.is_buyer ? '매수' : '매도'}
                        </Badge>
                      </TableCell>
                      <TableCell className="font-medium">${trade.price.toLocaleString()}</TableCell>
                      <TableCell className="hidden sm:table-cell">{trade.quantity.toFixed(6)}</TableCell>
                      <TableCell className="text-right font-medium">
                        ${(trade.price * trade.quantity).toLocaleString()}
                      </TableCell>
                      <TableCell className="hidden md:table-cell text-right text-sm text-muted-foreground">
                        {trade.commission.toFixed(8)}
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
