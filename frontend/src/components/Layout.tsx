import { useState } from 'react';
import { NavLink, Outlet, useLocation, Link } from 'react-router-dom';
import {
  TrendingUp,
  Bot,
  Menu,
  BarChart3,
  Coins,
  Home as HomeIcon,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();
  
  // 경로에 따라 메뉴 변경
  const isCrypto = location.pathname.startsWith('/crypto');

  const cryptoNavItems = [
    { to: '/crypto', icon: Coins, label: '코인 분석', end: true },
    { to: '/crypto/paper-trading', icon: TrendingUp, label: '페이퍼 트레이딩', end: false },
    { to: '/crypto/model-prep', icon: Bot, label: '모델 준비', end: false },
  ];

  const stocksNavItems = [
    { to: '/stocks', icon: BarChart3, label: '주식 분석', end: true },
    { to: '/stocks/paper-trading', icon: TrendingUp, label: '페이퍼 트레이딩', end: false },
  ];

  const navItems = isCrypto ? cryptoNavItems : stocksNavItems;

  const NavContent = () => (
    <>
      <div className="p-4 border-b border-border">
        <div className="flex items-center gap-2 mb-4">
          <Button
            variant="ghost"
            size="sm"
            asChild
            className="text-muted-foreground hover:text-foreground"
          >
            <Link to="/" onClick={() => setMobileOpen(false)}>
              <HomeIcon className="w-4 h-4" />
              홈으로
            </Link>
          </Button>
        </div>
        <h1 className="text-lg font-bold text-primary flex items-center gap-2">
          {isCrypto ? (
            <>
              <Coins className="w-5 h-5" />
              <span className="hidden sm:inline">암호화폐</span>
              <span className="sm:hidden">Crypto</span>
            </>
          ) : (
            <>
              <BarChart3 className="w-5 h-5" />
              <span className="hidden sm:inline">미국주식</span>
              <span className="sm:hidden">Stocks</span>
            </>
          )}
        </h1>
      </div>
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map((item) => (
            <li key={item.to}>
              <NavLink
                to={item.to}
                end={item.end}
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
                {item.label}
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Connection Status */}
      <div className="p-4 border-t border-border">
        <div className="flex items-center gap-2 text-sm">
          <span className="relative flex h-2 w-2">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-green-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-green-500"></span>
          </span>
          <span className="text-muted-foreground">
            {isCrypto ? 'Binance' : 'Alpha Vantage'}
          </span>
          <Badge variant="outline" className="ml-auto text-xs">Live</Badge>
        </div>
      </div>
    </>
  );

  return (
    <div className="flex h-screen bg-background">
      {/* Desktop Sidebar */}
      <aside className="hidden md:flex md:w-64 flex-col bg-card border-r border-border">
        <NavContent />
      </aside>

      {/* Mobile Header & Sheet */}
      <div className="flex-1 flex flex-col overflow-hidden">
        <header className="md:hidden flex items-center justify-between p-4 border-b border-border bg-card">
          <h1 className="text-lg font-bold text-primary flex items-center gap-2">
            {isCrypto ? (
              <>
                <Coins className="w-5 h-5" />
                Crypto
              </>
            ) : (
              <>
                <BarChart3 className="w-5 h-5" />
                Stocks
              </>
            )}
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

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
