import { useState } from 'react';
import { NavLink, Outlet } from 'react-router-dom';
import {
  LayoutDashboard,
  TrendingUp,
  History,
  Bot,
  Menu,
  Activity,
  Settings,
  BarChart3,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Sheet, SheetContent, SheetTrigger } from '@/components/ui/sheet';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

export function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false);

  const navItems = [
    { to: '/', icon: LayoutDashboard, label: '대시보드' },
    { to: '/market', icon: BarChart3, label: '마켓' },
    { to: '/signals', icon: Activity, label: '모니터링' },
    { to: '/trading', icon: TrendingUp, label: '매매' },
    { to: '/admin', icon: Settings, label: 'Admin' },
    { to: '/ai-settings', icon: Bot, label: 'AI 설정' },
    { to: '/history', icon: History, label: '거래 내역' },
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
          <span className="text-muted-foreground">Binance Testnet</span>
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

        {/* Main Content */}
        <main className="flex-1 overflow-y-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
