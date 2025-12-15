import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import {
  TrendingUp,
  Play,
  Loader2,
  Download,
  BarChart3,
  Calendar,
  DollarSign,
  Target,
  AlertCircle,
  Search
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer
} from "recharts";
import { API_URL } from "@/config/api";

export default function Backtesting() {
  const [strategies, setStrategies] = useState([]);
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [symbol, setSymbol] = useState("RELIANCE.NS");
  const [startDate, setStartDate] = useState("2023-01-01");
  const [endDate, setEndDate] = useState("2024-12-31");
  const [initialCapital, setInitialCapital] = useState(100000);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);

  // Autocomplete state
  const [suggestions, setSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [searchLoading, setSearchLoading] = useState(false);
  const symbolInputRef = useRef(null);
  const suggestionsRef = useRef(null);

  useEffect(() => {
    fetchStrategies();
    fetchHistory();

    // Close suggestions when clicking outside
    const handleClickOutside = (event) => {
      if (
        symbolInputRef.current &&
        !symbolInputRef.current.contains(event.target) &&
        suggestionsRef.current &&
        !suggestionsRef.current.contains(event.target)
      ) {
        setShowSuggestions(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const fetchStrategies = async () => {
    try {
      const response = await axios.get(`${API_URL}/backtest/strategies`);
      setStrategies(response.data.strategies);
      if (response.data.strategies.length > 0) {
        setSelectedStrategy(response.data.strategies[0].name);
      }
    } catch (error) {
      console.error("Error fetching strategies:", error);
      toast.error("Failed to load strategies");
    }
  };

  const fetchHistory = async () => {
    try {
      const response = await axios.get(`${API_URL}/backtest/history?limit=10`);
      setHistory(response.data);
    } catch (error) {
      console.error("Error fetching history:", error);
    }
  };

  // Search for stocks as user types
  const searchStocks = async (query) => {
    if (!query || query.length < 1) {
      setSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    setSearchLoading(true);
    try {
      const response = await axios.get(`${API_URL}/stocks/search?q=${query}`);
      setSuggestions(response.data);
      setShowSuggestions(response.data.length > 0);
    } catch (error) {
      console.error("Error searching stocks:", error);
      setSuggestions([]);
    } finally {
      setSearchLoading(false);
    }
  };

  // Handle symbol input change with debounce
  const handleSymbolChange = (e) => {
    const value = e.target.value.toUpperCase();
    setSymbol(value);

    // Debounce search
    clearTimeout(window.symbolSearchTimeout);
    window.symbolSearchTimeout = setTimeout(() => {
      searchStocks(value.replace('.NS', '').replace('.BO', ''));
    }, 200);
  };

  // Handle stock selection from suggestions
  const handleSelectStock = (stock) => {
    setSymbol(`${stock.symbol}.NS`);
    setShowSuggestions(false);
    setSuggestions([]);
  };

  const runBacktest = async () => {
    if (!selectedStrategy) {
      toast.error("Please select a strategy");
      return;
    }

    setIsRunning(true);
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/backtest/run`, {
        symbol,
        strategy: selectedStrategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital
      });

      setResult(response.data);
      fetchHistory();
      toast.success("Backtest completed!");
    } catch (error) {
      console.error("Backtest error:", error);
      toast.error(error.response?.data?.detail || "Backtest failed");
    } finally {
      setIsRunning(false);
    }
  };

  const exportResults = () => {
    if (!result) return;
    
    const dataStr = JSON.stringify(result, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `backtest-${symbol}-${Date.now()}.json`;
    link.click();
    URL.revokeObjectURL(url);
    toast.success("Results exported");
  };

  const formatEquityCurve = () => {
    if (!result || !result.equity_curve) return [];
    
    return Object.entries(result.equity_curve).map(([date, value]) => ({
      date: new Date(date).toLocaleDateString(),
      value: value
    }));
  };

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
            <BarChart3 className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-heading font-bold text-text-primary">
              Strategy Backtesting
            </h1>
            <p className="text-text-secondary">
              Test trading strategies against historical data
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Configuration Panel */}
        <div className="lg:col-span-1">
          <Card className="card-surface">
            <CardHeader>
              <CardTitle className="text-sm font-heading">Backtest Configuration</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {/* Symbol with Autocomplete */}
              <div className="relative">
                <Label htmlFor="symbol">Stock Symbol</Label>
                <div className="relative">
                  <Input
                    ref={symbolInputRef}
                    id="symbol"
                    value={symbol}
                    onChange={handleSymbolChange}
                    onFocus={() => symbol && searchStocks(symbol.replace('.NS', '').replace('.BO', ''))}
                    placeholder="e.g., RELIANCE, TCS, INFY"
                    className="bg-surface-highlight border-[#1F1F1F]"
                    autoComplete="off"
                  />
                  {searchLoading && (
                    <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-text-secondary" />
                  )}
                </div>

                {/* Suggestions Dropdown */}
                {showSuggestions && suggestions.length > 0 && (
                  <div
                    ref={suggestionsRef}
                    className="absolute z-50 w-full mt-1 bg-surface border border-[#1F1F1F] rounded-lg shadow-lg overflow-hidden"
                  >
                    {suggestions.map((stock) => (
                      <div
                        key={stock.symbol}
                        onClick={() => handleSelectStock(stock)}
                        className="px-3 py-2 hover:bg-surface-highlight cursor-pointer transition-colors flex items-center justify-between"
                      >
                        <div>
                          <div className="font-data text-sm text-text-primary">
                            {stock.symbol}
                          </div>
                          <div className="text-xs text-text-secondary truncate max-w-[200px]">
                            {stock.name}
                          </div>
                        </div>
                        <Badge variant="outline" className="text-xs ml-2">
                          {stock.exchange}
                        </Badge>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Strategy */}
              <div>
                <Label htmlFor="strategy">Strategy</Label>
                <Select value={selectedStrategy} onValueChange={setSelectedStrategy}>
                  <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                    <SelectValue placeholder="Select strategy" />
                  </SelectTrigger>
                  <SelectContent>
                    {strategies.map((strategy) => (
                      <SelectItem key={strategy.name} value={strategy.name}>
                        {strategy.display_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedStrategy && (
                  <p className="text-xs text-text-secondary mt-1">
                    {strategies.find(s => s.name === selectedStrategy)?.description}
                  </p>
                )}
              </div>

              {/* Date Range */}
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <Label htmlFor="startDate">Start Date</Label>
                  <Input
                    id="startDate"
                    type="date"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="bg-surface-highlight border-[#1F1F1F]"
                  />
                </div>
                <div>
                  <Label htmlFor="endDate">End Date</Label>
                  <Input
                    id="endDate"
                    type="date"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="bg-surface-highlight border-[#1F1F1F]"
                  />
                </div>
              </div>

              {/* Initial Capital */}
              <div>
                <Label htmlFor="capital">Initial Capital (₹)</Label>
                <Input
                  id="capital"
                  type="number"
                  value={initialCapital}
                  onChange={(e) => setInitialCapital(Number(e.target.value))}
                  className="bg-surface-highlight border-[#1F1F1F]"
                />
              </div>

              {/* Run Button */}
              <Button
                onClick={runBacktest}
                disabled={isRunning}
                className="w-full btn-primary"
              >
                {isRunning ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Running...
                  </>
                ) : (
                  <>
                    <Play className="w-4 h-4 mr-2" />
                    Run Backtest
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          {/* Recent Backtests */}
          <Card className="card-surface mt-6">
            <CardHeader>
              <CardTitle className="text-sm font-heading">Recent Backtests</CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[300px]">
                {history.length === 0 ? (
                  <p className="text-text-secondary text-sm text-center py-4">
                    No backtests yet
                  </p>
                ) : (
                  <div className="space-y-2">
                    {history.map((item, idx) => (
                      <div
                        key={idx}
                        className="p-3 rounded-lg bg-surface-highlight hover:bg-[#1F1F1F] transition-colors cursor-pointer"
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="font-data text-sm text-text-primary">
                            {item.symbol}
                          </span>
                          <Badge variant="outline" className="text-xs">
                            {item.strategy}
                          </Badge>
                        </div>
                        <div className="flex items-center justify-between text-xs text-text-secondary">
                          <span>{item.metrics?.total_return_pct}%</span>
                          <span>{item.metrics?.total_trades} trades</span>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>

        {/* Results Panel */}
        <div className="lg:col-span-2">
          {!result ? (
            <Card className="card-surface">
              <CardContent className="py-24">
                <div className="text-center">
                  <BarChart3 className="w-16 h-16 mx-auto text-text-secondary/50 mb-4" />
                  <p className="text-text-secondary">
                    Configure parameters and run a backtest to see results
                  </p>
                </div>
              </CardContent>
            </Card>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-6"
            >
              {/* Performance Summary */}
              <Card className="card-surface">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-heading">Performance Summary</CardTitle>
                    <Button variant="outline" size="sm" onClick={exportResults}>
                      <Download className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 rounded-lg bg-surface-highlight">
                      <p className="text-xs text-text-secondary mb-1">Total Return</p>
                      <p className={`text-2xl font-data font-bold ${
                        result.metrics.total_return_pct >= 0 ? 'text-success' : 'text-danger'
                      }`}>
                        {result.metrics.total_return_pct}%
                      </p>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-highlight">
                      <p className="text-xs text-text-secondary mb-1">CAGR</p>
                      <p className="text-2xl font-data font-bold text-text-primary">
                        {result.metrics.cagr_pct}%
                      </p>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-highlight">
                      <p className="text-xs text-text-secondary mb-1">Sharpe Ratio</p>
                      <p className="text-2xl font-data font-bold text-text-primary">
                        {result.metrics.sharpe_ratio}
                      </p>
                    </div>
                    <div className="p-4 rounded-lg bg-surface-highlight">
                      <p className="text-xs text-text-secondary mb-1">Max Drawdown</p>
                      <p className="text-2xl font-data font-bold text-danger">
                        {result.metrics.max_drawdown_pct}%
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Equity Curve */}
              <Card className="card-surface">
                <CardHeader>
                  <CardTitle className="text-sm font-heading">Equity Curve</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={formatEquityCurve()}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#1F1F1F" />
                        <XAxis 
                          dataKey="date" 
                          stroke="#A1A1AA"
                          tick={{ fill: '#A1A1AA', fontSize: 11 }}
                        />
                        <YAxis 
                          stroke="#A1A1AA"
                          tick={{ fill: '#A1A1AA', fontSize: 11 }}
                          tickFormatter={(value) => `₹${(value/1000).toFixed(0)}K`}
                        />
                        <RechartsTooltip
                          contentStyle={{
                            backgroundColor: '#0A0A0A',
                            border: '1px solid #1F1F1F',
                            borderRadius: '8px'
                          }}
                          formatter={(value) => [`₹${value.toLocaleString()}`, 'Portfolio Value']}
                        />
                        <Line
                          type="monotone"
                          dataKey="value"
                          stroke="#3B82F6"
                          strokeWidth={2}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>

              {/* Detailed Metrics */}
              <Card className="card-surface">
                <CardHeader>
                  <CardTitle className="text-sm font-heading">Detailed Metrics</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                    <MetricItem label="Final Value" value={`₹${result.metrics.final_value.toLocaleString()}`} />
                    <MetricItem label="Total Trades" value={result.metrics.total_trades} />
                    <MetricItem label="Win Rate" value={`${result.metrics.win_rate_pct}%`} />
                    <MetricItem label="Profit Factor" value={result.metrics.profit_factor} />
                    <MetricItem label="Avg Win" value={`₹${result.metrics.avg_win.toLocaleString()}`} />
                    <MetricItem label="Avg Loss" value={`₹${result.metrics.avg_loss.toLocaleString()}`} />
                    <MetricItem label="Sortino Ratio" value={result.metrics.sortino_ratio} />
                    <MetricItem label="Calmar Ratio" value={result.metrics.calmar_ratio} />
                    <MetricItem label="Volatility" value={`${result.metrics.volatility_pct}%`} />
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          )}
        </div>
      </div>
    </div>
  );
}

function MetricItem({ label, value }) {
  return (
    <div className="p-3 rounded-lg bg-surface-highlight">
      <p className="text-xs text-text-secondary mb-1">{label}</p>
      <p className="font-data text-text-primary">{value}</p>
    </div>
  );
}

