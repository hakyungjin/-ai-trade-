import { MarketOverview } from './MarketOverview';

export function MarketPage() {
  return (
    <div className="p-4 md:p-6">
      <h1 className="text-2xl font-bold mb-6">마켓</h1>
      <MarketOverview />
    </div>
  );
}
