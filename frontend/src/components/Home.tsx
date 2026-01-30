import { useNavigate } from 'react-router-dom';
import { Coins, TrendingUp } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';

export function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center p-4">
      <div className="w-full max-w-4xl">
        {/* 헤더 */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">AI Trader Pro</h1>
          <p className="text-lg text-slate-300">암호화폐와 주식을 한번에 관리하는 AI 자동매매 시스템</p>
        </div>

        {/* 선택 버튼 */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          {/* 코인 버튼 */}
          <Card 
            className="bg-slate-800 border-slate-700 hover:border-blue-500 hover:shadow-lg hover:shadow-blue-500/20 transition-all duration-300 cursor-pointer group overflow-hidden"
            onClick={() => navigate('/crypto')}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-blue-600/10 to-blue-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <Coins className="w-16 h-16 text-blue-400" />
                <TrendingUp className="w-8 h-8 text-blue-400/50 group-hover:text-blue-400 transition-colors" />
              </div>
              <CardTitle className="text-3xl text-white">암호화폐</CardTitle>
            </CardHeader>
            <CardContent className="relative z-10 space-y-4">
              <p className="text-slate-300 text-base">
                24/7 운영되는 가상자산 시장에서 AI 기반 자동매매
              </p>
              <ul className="space-y-2 text-sm text-slate-400">
                <li>✓ 실시간 신호 생성</li>
                <li>✓ 다중 타임프레임 분석</li>
                <li>✓ 자동 포지션 관리</li>
                <li>✓ 24/7 모니터링</li>
              </ul>
              <Button 
                className="w-full bg-blue-600 hover:bg-blue-700 text-white mt-6 group-hover:shadow-lg group-hover:shadow-blue-600/50"
                onClick={(e) => {
                  e.stopPropagation();
                  navigate('/crypto');
                }}
              >
                암호화폐 시작하기 →
              </Button>
            </CardContent>
          </Card>

          {/* 주식 버튼 */}
          <Card 
            className="bg-slate-800 border-slate-700 hover:border-emerald-500 hover:shadow-lg hover:shadow-emerald-500/20 transition-all duration-300 cursor-pointer group overflow-hidden"
            onClick={() => navigate('/stocks')}
          >
            <div className="absolute inset-0 bg-gradient-to-br from-emerald-600/10 to-emerald-900/10 opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
            <CardHeader className="relative z-10">
              <div className="flex items-center justify-between mb-4">
                <TrendingUp className="w-16 h-16 text-emerald-400" />
                <TrendingUp className="w-8 h-8 text-emerald-400/50 group-hover:text-emerald-400 transition-colors" />
              </div>
              <CardTitle className="text-3xl text-white">미국주식</CardTitle>
            </CardHeader>
            <CardContent className="relative z-10 space-y-4">
              <p className="text-slate-300 text-base">
                NYSE와 NASDAQ에 상장된 우량 종목 자동 분석
              </p>
              <ul className="space-y-2 text-sm text-slate-400">
                <li>✓ 실시간 신호 생성</li>
                <li>✓ 펀더멘탈 분석</li>
                <li>✓ 리스크 관리</li>
                <li>✓ 시간외 거래 알림</li>
              </ul>
              <Button 
                className="w-full bg-emerald-600 hover:bg-emerald-700 text-white mt-6 group-hover:shadow-lg group-hover:shadow-emerald-600/50"
                onClick={(e) => {
                  e.stopPropagation();
                  navigate('/stocks');
                }}
              >
                주식 시작하기 →
              </Button>
            </CardContent>
          </Card>
        </div>

        {/* 하단 정보 */}
        <div className="mt-16 text-center">
          <p className="text-slate-400 text-sm">
            이 시스템은 AI 기반 신호만 제공합니다. 실제 매매 결정은 신중하게 하세요.
          </p>
        </div>
      </div>
    </div>
  );
}
