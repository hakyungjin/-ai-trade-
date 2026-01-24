import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  Target,
  BarChart3,
  Menu,
  Bot,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';
import { usePaperTradingStore } from '@/store/paperTradingStore';

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const { positions = [], totalPnl = 0, balance = 10000 } = usePaperTradingStore();
  
  const openPositions = (positions || []).filter((p) => p.status === 'OPEN');
  const spotPositions = openPositions.filter((p) => p.marketType === 'spot');
  const futuresPositions = openPositions.filter((p) => p.marketType === 'futures' || !p.marketType);

  const navItems = [
    { 
      to: '/', 
      icon: Target, 
      label: '모의투자',
      badge: openPositions.length > 0 ? openPositions.length : undefined,
      subBadge: openPositions.length > 0 ? `현${spotPositions.length}/선${futuresPositions.length}` : undefined,
    },
    { 
      to: '/analysis', 
      icon: BarChart3, 
      label: '코인 분석',
    },
  ];

  const NavContent = () => (
    <>
      <div className="p-4 border-b border-border">
        <h1 className="text-xl font-bold text-primary flex items-center gap-2">
          <Bot className="w-6 h-6" />
          <span className="hidden sm:inline">Crypto AI Trader</span>
          <span className="sm:hidden">CAT</span>
        </h1>
      </div>
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                onClick={() => setMobileOpen(false)}
                className={({ isActive }) =>
                  cn(
                    'flex items-center gap-3 px-4 py-3 rounded-lg transition-colors min-h-[48px]',
                    isActive
                      ? 'bg-primary/10 text-primary'
                      : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                  )
                }
              >
                <item.icon className="w-5 h-5" />
                <span className="flex-1">{item.label}</span>
                <div className="flex items-center gap-1">
                  {item.badge && (
                    <Badge variant="secondary" className="text-xs">
                      {item.badge}
                    </Badge>
                  )}
                  {item.subBadge && (
                    <span className="text-[10px] text-muted-foreground">
                      {item.subBadge}
                    </span>
                  )}
                </div>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* 계정 요약 */}
      <div className="p-4 border-t border-border space-y-3">
        <div className="text-sm">
          <div className="text-muted-foreground mb-1">잔고</div>
          <div className="text-lg font-bold">
            ${balance.toLocaleString(undefined, { maximumFractionDigits: 2 })}
          </div>
        </div>
        <div className="text-sm">
          <div className="text-muted-foreground mb-1">총 PnL</div>
          <div className={cn(
            'text-base font-bold',
            totalPnl > 0 ? 'text-green-600' : totalPnl < 0 ? 'text-red-600' : ''
          )}>
            {totalPnl >= 0 ? '+' : ''}${totalPnl.toFixed(2)}
          </div>
        </div>
        <div className="flex items-center gap-2 text-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          <span className="text-muted-foreground">Binance</span>
          <Badge variant="outline" className="ml-auto text-xs">Live</Badge>
        </div>
      </div>
    </>
  );

  return (
    <div className="flex min-h-screen bg-background">
      {/* Desktop Sidebar - 고정 */}
      <aside className="hidden md:flex md:w-64 flex-col bg-card border-r border-border fixed top-0 left-0 h-screen z-40">
        <NavContent />
      </aside>

      {/* Mobile Header & Sheet */}
      <div className="flex-1 flex flex-col md:ml-64">
        <header className="md:hidden flex items-center justify-between p-4 border-b border-border bg-card sticky top-0 z-30">
          <h1 className="text-lg font-bold text-primary flex items-center gap-2">
            <Bot className="w-5 h-5" />
            Crypto AI Trader
          </h1>
          <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
            <SheetTrigger asChild>
              <Button variant="ghost" size="icon" className="h-10 w-10">
                <Menu className="h-5 w-5" />
              </Button>
            </SheetTrigger>
            <SheetContent side="left" className="w-64 p-0 flex flex-col">
              <NavContent />
            </SheetContent>
          </Sheet>
        </header>

        {/* Main Content - 브라우저 기본 스크롤 사용 */}
        <main className="flex-1">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
