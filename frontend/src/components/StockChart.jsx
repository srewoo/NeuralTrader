import { useMemo } from "react";
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip,
  ReferenceLine
} from "recharts";
import { format } from "date-fns";

/**
 * StockChart Component
 * Displays historical price data with area chart visualization
 * @param {Object} props - Component props
 * @param {Array} props.data - Price history data from API
 */
export default function StockChart({ data }) {
  // Transform API data to chart format
  const chartData = useMemo(() => {
    if (!data || !Array.isArray(data) || data.length === 0) {
      return [];
    }

    return data.map(item => ({
      date: item.date,
      price: parseFloat(item.close) || 0,
      volume: parseInt(item.volume) || 0,
      displayDate: format(new Date(item.date), "MMM dd")
    }));
  }, [data]);

  // Calculate price change for gradient color
  const priceChange = useMemo(() => {
    if (chartData.length < 2) return 0;
    return chartData[chartData.length - 1].price - chartData[0].price;
  }, [chartData]);

  const isPositive = priceChange >= 0;

  // Custom tooltip component
  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length > 0) {
      const data = payload[0].payload;
      return (
        <div className="bg-surface border border-[#1F1F1F] rounded-lg p-3 shadow-xl">
          <p className="text-xs text-text-secondary mb-1">
            {format(new Date(data.date), "MMM dd, yyyy")}
          </p>
          <p className="text-sm font-data font-medium text-text-primary">
            ₹{data.price.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </p>
          <p className="text-xs text-text-secondary mt-1">
            Vol: {(data.volume / 1000000).toFixed(2)}M
          </p>
        </div>
      );
    }
    return null;
  };

  if (!chartData || chartData.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <p className="text-text-secondary text-sm">No price data available</p>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart
        data={chartData}
        margin={{ top: 10, right: 10, left: 0, bottom: 0 }}
      >
        <defs>
          <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
            <stop 
              offset="5%" 
              stopColor={isPositive ? "#00FF94" : "#FF2E2E"} 
              stopOpacity={0.3}
            />
            <stop 
              offset="95%" 
              stopColor={isPositive ? "#00FF94" : "#FF2E2E"} 
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke="#1F1F1F" 
          vertical={false}
        />
        
        <XAxis
          dataKey="displayDate"
          stroke="#A1A1AA"
          tick={{ fill: '#A1A1AA', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={{ stroke: '#1F1F1F' }}
          interval="preserveStartEnd"
        />
        
        <YAxis
          stroke="#A1A1AA"
          tick={{ fill: '#A1A1AA', fontSize: 11, fontFamily: 'JetBrains Mono' }}
          tickLine={false}
          axisLine={{ stroke: '#1F1F1F' }}
          tickFormatter={(value) => `₹${value.toFixed(0)}`}
          domain={['auto', 'auto']}
        />
        
        <Tooltip content={<CustomTooltip />} />
        
        {/* Reference line for starting price */}
        <ReferenceLine 
          y={chartData[0]?.price} 
          stroke="#A1A1AA" 
          strokeDasharray="3 3"
          strokeOpacity={0.3}
        />
        
        <Area
          type="monotone"
          dataKey="price"
          stroke={isPositive ? "#00FF94" : "#FF2E2E"}
          strokeWidth={2}
          fill="url(#colorPrice)"
          animationDuration={800}
          animationEasing="ease-in-out"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

