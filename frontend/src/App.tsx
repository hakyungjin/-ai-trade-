import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Dashboard } from './components/Dashboard';
import { TradingPanel } from './components/TradingPanel';
import { AISettings } from './components/AISettings';
import { TradeHistory } from './components/TradeHistory';
import { SignalsPage } from './components/signals/SignalsPage';
import { AdminPage } from './components/admin/AdminPage';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="signals" element={<SignalsPage />} />
            <Route path="trading" element={<TradingPanel />} />
            <Route path="admin" element={<AdminPage />} />
            <Route path="ai-settings" element={<AISettings />} />
            <Route path="history" element={<TradeHistory />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
