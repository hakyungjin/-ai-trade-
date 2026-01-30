import { useState } from 'react';
import {
  Database,
  Search,
  Loader,
  CheckCircle2,
  XCircle,
  Play,
  Coins,
  LineChart,
  BarChart3,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { cn } from '@/lib/utils';
import { apiClient, marketApi } from '@/api/client';
import { useQuery } from '@tanstack/react-query';

type MarketType = 'spot' | 'futures';

interface CollectionJob {
  id: string;
  symbol: string;
  timeframe: string;
  marketType: MarketType;
  targetCount: number;
  currentCount?: number;
  status: 'pending' | 'collecting' | 'completed' | 'error';
  message?: string;
  progress?: number;
}

const TIMEFRAMES = [
  { value: '1m', label: '1분' },
  { value: '3m', label: '3분' },
  { value: '5m', label: '5분' },
  { value: '15m', label: '15분' },
  { value: '30m', label: '30분' },
  { value: '1h', label: '1시간' },
  { value: '2h', label: '2시간' },
  { value: '4h', label: '4시간' },
  { value: '1d', label: '1일' },
];

const TARGET_COUNTS = [
  { value: 10000, label: '10,000개' },
  { value: 20000, label: '20,000개' },
  { value: 50000, label: '50,000개' },
  { value: 100000, label: '100,000개' },
];

export function ModelPreparation() {
  const [marketType, setMarketType] = useState<MarketType>('spot');
  const [selectedSymbol, setSelectedSymbol] = useState('');
  const [selectedTimeframe, setSelectedTimeframe] = useState('5m');
  const [selectedTargetCount, setSelectedTargetCount] = useState(50000);
  const [customTargetCount, setCustomTargetCount] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [jobs, setJobs] = useState<CollectionJob[]>([]);
  const [isCollecting, setIsCollecting] = useState(false);

  // 코인 목록 조회
  const { data: coinsData, isLoading: coinsLoading } = useQuery({
    queryKey: ['coins', 'list'],
    queryFn: async () => {
      const response = await apiClient.get('/v1/coins/list');
      return response.data;
    },
  });

  // 심볼 검색
  const { data: searchResults, isLoading: searchLoading } = useQuery({
    queryKey: ['market', 'search', searchQuery],
    queryFn: async () => {
      if (!searchQuery || searchQuery.length < 2) return { data: [] };
      const response = await marketApi.search(searchQuery);
      return response.data;
    },
    enabled: searchQuery.length >= 2,
  });

  const coins = coinsData?.coins || [];
  const searchCoins = searchResults?.data || [];

  // 수집 시작
  const handleStartCollection = async () => {
    if (!selectedSymbol) {
      alert('코인을 선택해주세요.');
      return;
    }

    const targetCount = customTargetCount
      ? parseInt(customTargetCount)
      : selectedTargetCount;

    if (!targetCount || targetCount < 1000) {
      alert('목표 개수는 최소 1,000개 이상이어야 합니다.');
      return;
    }

    const jobId = `${selectedSymbol}-${selectedTimeframe}-${Date.now()}`;
    const newJob: CollectionJob = {
      id: jobId,
      symbol: selectedSymbol,
      timeframe: selectedTimeframe,
      marketType: marketType,
      targetCount: targetCount,
      status: 'pending',
    };

    setJobs((prev) => [newJob, ...prev]);
    setIsCollecting(true);

    try {
      const response = await apiClient.post(
        `/v1/coins/collect/${selectedSymbol}`,
        null,
        {
          params: {
            timeframe: selectedTimeframe,
            target_count: targetCount,
            market_type: marketType,
          },
        }
      );

      const data = response.data;
      setJobs((prev) =>
        prev.map((job) =>
          job.id === jobId
            ? {
                ...job,
                status: data.success ? 'completed' : 'error',
                currentCount: data.current_count,
                message: data.message,
                progress: data.current_count
                  ? Math.min(100, (data.current_count / targetCount) * 100)
                  : 0,
              }
            : job
        )
      );
    } catch (error: any) {
      setJobs((prev) =>
        prev.map((job) =>
          job.id === jobId
            ? {
                ...job,
                status: 'error',
                message: error.response?.data?.detail || error.message || '수집 실패',
              }
            : job
        )
      );
    } finally {
      setIsCollecting(false);
    }
  };

  return (
    <div className="container mx-auto p-4 md:p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold flex items-center gap-2">
            <Database className="w-8 h-8" />
            모델 준비
          </h1>
          <p className="text-muted-foreground mt-2">
            AI 모델 학습을 위한 캔들 데이터 수집
          </p>
        </div>
      </div>

      <Tabs
        value={marketType}
        onValueChange={(v) => setMarketType(v as MarketType)}
        className="w-full"
      >
        <TabsList className="grid w-full max-w-md grid-cols-2">
          <TabsTrigger value="spot" className="flex items-center gap-2">
            <Coins className="w-4 h-4" />
            현물 (Spot)
          </TabsTrigger>
          <TabsTrigger value="futures" className="flex items-center gap-2">
            <LineChart className="w-4 h-4" />
            선물 (Futures)
          </TabsTrigger>
        </TabsList>
      </Tabs>

      <div className="grid gap-6 md:grid-cols-2">
        {/* 설정 카드 */}
        <Card>
          <CardHeader>
            <CardTitle>데이터 수집 설정</CardTitle>
            <CardDescription>
              코인, 타임프레임, 목표 개수를 선택하세요
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* 코인 선택 */}
            <div className="space-y-2">
              <Label>코인 선택</Label>
              <div className="space-y-2">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" />
                  <Input
                    placeholder="코인 검색 (예: BTCUSDT)"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value.toUpperCase())}
                    className="pl-9"
                  />
                </div>
                {searchQuery && searchQuery.length >= 2 && (
                  <div className="border rounded-md max-h-48 overflow-y-auto">
                    {searchLoading ? (
                      <div className="p-4 text-center text-sm text-muted-foreground">
                        검색 중...
                      </div>
                    ) : searchCoins.length > 0 ? (
                      <div className="p-2 space-y-1">
                        {searchCoins.slice(0, 10).map((coin: any) => (
                          <button
                            key={coin.symbol}
                            onClick={() => {
                              setSelectedSymbol(coin.symbol);
                              setSearchQuery('');
                            }}
                            className={cn(
                              'w-full text-left px-3 py-2 rounded-md hover:bg-accent transition-colors',
                              selectedSymbol === coin.symbol && 'bg-accent'
                            )}
                          >
                            <div className="flex items-center justify-between">
                              <span className="font-medium">{coin.symbol}</span>
                              <Badge variant="outline" className="text-xs">
                                {coin.quoteAsset}
                              </Badge>
                            </div>
                          </button>
                        ))}
                      </div>
                    ) : (
                      <div className="p-4 text-center text-sm text-muted-foreground">
                        검색 결과가 없습니다
                      </div>
                    )}
                  </div>
                )}
                {selectedSymbol && (
                  <div className="flex items-center gap-2">
                    <Badge variant="default" className="text-sm">
                      {selectedSymbol}
                    </Badge>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedSymbol('')}
                    >
                      <XCircle className="w-4 h-4" />
                    </Button>
                  </div>
                )}
              </div>
            </div>

            {/* 타임프레임 선택 */}
            <div className="space-y-2">
              <Label>타임프레임</Label>
              <Select
                value={selectedTimeframe}
                onValueChange={setSelectedTimeframe}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {TIMEFRAMES.map((tf) => (
                    <SelectItem key={tf.value} value={tf.value}>
                      {tf.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* 목표 개수 선택 */}
            <div className="space-y-2">
              <Label>목표 캔들 개수</Label>
              <div className="grid grid-cols-2 gap-2">
                {TARGET_COUNTS.map((tc) => (
                  <Button
                    key={tc.value}
                    variant={selectedTargetCount === tc.value ? 'default' : 'outline'}
                    onClick={() => {
                      setSelectedTargetCount(tc.value);
                      setCustomTargetCount('');
                    }}
                    className="justify-start"
                  >
                    {tc.label}
                  </Button>
                ))}
              </div>
              <div className="space-y-2">
                <Input
                  placeholder="또는 직접 입력 (예: 75000)"
                  value={customTargetCount}
                  onChange={(e) => {
                    setCustomTargetCount(e.target.value);
                    setSelectedTargetCount(0);
                  }}
                  type="number"
                  min={1000}
                />
                {customTargetCount && (
                  <p className="text-xs text-muted-foreground">
                    {parseInt(customTargetCount) >= 1000
                      ? `${parseInt(customTargetCount).toLocaleString()}개`
                      : '최소 1,000개 이상 입력해주세요'}
                  </p>
                )}
              </div>
            </div>

            {/* 수집 시작 버튼 */}
            <Button
              onClick={handleStartCollection}
              disabled={!selectedSymbol || isCollecting}
              className="w-full"
              size="lg"
            >
              {isCollecting ? (
                <>
                  <Loader className="w-4 h-4 mr-2 animate-spin" />
                  수집 중...
                </>
              ) : (
                <>
                  <Play className="w-4 h-4 mr-2" />
                  데이터 수집 시작
                </>
              )}
            </Button>
          </CardContent>
        </Card>

        {/* 수집 작업 목록 */}
        <Card>
          <CardHeader>
            <CardTitle>수집 작업 목록</CardTitle>
            <CardDescription>
              진행 중이거나 완료된 수집 작업을 확인하세요
            </CardDescription>
          </CardHeader>
          <CardContent>
            {jobs.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <BarChart3 className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>아직 수집 작업이 없습니다</p>
              </div>
            ) : (
              <div className="space-y-3">
                {jobs.map((job) => (
                  <div
                    key={job.id}
                    className="border rounded-lg p-4 space-y-2"
                  >
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            job.status === 'completed'
                              ? 'default'
                              : job.status === 'error'
                              ? 'destructive'
                              : 'secondary'
                          }
                        >
                          {job.marketType === 'spot' ? (
                            <Coins className="w-3 h-3 mr-1" />
                          ) : (
                            <LineChart className="w-3 h-3 mr-1" />
                          )}
                          {job.marketType}
                        </Badge>
                        <span className="font-medium">{job.symbol}</span>
                        <span className="text-sm text-muted-foreground">
                          {job.timeframe}
                        </span>
                      </div>
                      {job.status === 'completed' ? (
                        <CheckCircle2 className="w-5 h-5 text-green-600" />
                      ) : job.status === 'error' ? (
                        <XCircle className="w-5 h-5 text-red-600" />
                      ) : (
                        <Loader className="w-5 h-5 animate-spin text-blue-600" />
                      )}
                    </div>

                    {job.currentCount !== undefined && (
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-sm">
                          <span className="text-muted-foreground">
                            {job.currentCount.toLocaleString()} /{' '}
                            {job.targetCount.toLocaleString()}개
                          </span>
                          <span className="text-muted-foreground">
                            {job.progress?.toFixed(1)}%
                          </span>
                        </div>
                        <Progress value={job.progress || 0} className="h-2" />
                      </div>
                    )}

                    {job.message && (
                      <p
                        className={cn(
                          'text-sm',
                          job.status === 'error'
                            ? 'text-red-600'
                            : 'text-muted-foreground'
                        )}
                      >
                        {job.message}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* 코인 목록 (참고용) */}
      <Card>
        <CardHeader>
          <CardTitle>모니터링 중인 코인</CardTitle>
          <CardDescription>
            데이터베이스에 저장된 코인 목록
          </CardDescription>
        </CardHeader>
        <CardContent>
          {coinsLoading ? (
            <div className="text-center py-8">
              <Loader className="w-6 h-6 mx-auto animate-spin" />
            </div>
          ) : coins.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              코인이 없습니다
            </div>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
              {coins
                .filter((coin: any) => coin.market_type === marketType)
                .map((coin: any) => (
                  <button
                    key={coin.id}
                    onClick={() => setSelectedSymbol(coin.symbol)}
                    className={cn(
                      'p-3 border rounded-lg hover:bg-accent transition-colors text-left',
                      selectedSymbol === coin.symbol && 'bg-accent border-primary'
                    )}
                  >
                    <div className="font-medium">{coin.symbol}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {coin.candle_count?.toLocaleString() || 0}개
                    </div>
                  </button>
                ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

