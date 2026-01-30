import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Layout } from './components/Layout';
import { Home } from './components/Home';
import { PaperTrading } from './components/PaperTrading';
import { CoinAnalysis } from './components/CoinAnalysis';
import { StockAnalysis } from './components/StockAnalysis';
import { ModelPreparation } from './components/ModelPreparation';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/crypto" element={<Layout />}>
            <Route index element={<CoinAnalysis />} />
            <Route path="paper-trading" element={<PaperTrading />} />
            <Route path="model-prep" element={<ModelPreparation />} />
          </Route>
          <Route path="/stocks" element={<Layout />}>
            <Route index element={<StockAnalysis />} />
            <Route path="paper-trading" element={<PaperTrading />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

export default App;
