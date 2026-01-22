import { useEffect, useState } from 'react';
import { Line, LineChart, ResponsiveContainer } from 'recharts';
import { marketApi } from '@/api/client';

interface MiniChartProps {
  symbol: string;
  trend?: 'up' | 'down' | 'neutral';
  height?: number;
}

export function MiniChart({ symbol, trend = 'neutral', height = 40 }: MiniChartProps) {
  const [data, setData] = useState<{ value: number }[]>([]);

  useEffect(() => {
    loadChartData();
  }, [symbol]);

  const loadChartData = async () => {
    try {
      const response = await marketApi.getMiniChart(symbol, '1h', 24);
      if (response.data.success && response.data.prices) {
        setData(response.data.prices.map((price: number) => ({ value: price })));
      }
    } catch (error) {
      console.error('Failed to load mini chart:', error);
    }
  };

  const strokeColor = trend === 'up' ? '#22c55e' : trend === 'down' ? '#ef4444' : '#6b7280';

  if (data.length === 0) {
    return <div style={{ height }} className="bg-muted/20 rounded animate-pulse" />;
  }

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data}>
        <Line
          type="monotone"
          dataKey="value"
          stroke={strokeColor}
          strokeWidth={1.5}
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
