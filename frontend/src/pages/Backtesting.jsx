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
  Search,
  Brain,
  Shuffle,
  RefreshCw,
  Settings2,
  Sparkles
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
  const [comparisonResults, setComparisonResults] = useState(null);
  const [aiInsights, setAiInsights] = useState(null);
  const [loadingInsights, setLoadingInsights] = useState(false);
  const [history, setHistory] = useState([]);

  // Advanced features state
  const [activeTab, setActiveTab] = useState("basic");
  const [monteCarloResult, setMonteCarloResult] = useState(null);
  const [walkForwardResult, setWalkForwardResult] = useState(null);
  const [isRunningMonteCarlo, setIsRunningMonteCarlo] = useState(false);
  const [isRunningWalkForward, setIsRunningWalkForward] = useState(false);
  const [monteCarloSimulations, setMonteCarloSimulations] = useState(1000);
  const [walkForwardWindows, setWalkForwardWindows] = useState(5);

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
    setComparisonResults(null);

    try {
      if (selectedStrategy === 'ALL') {
        // Run all strategies in parallel
        toast.info("Running all strategies in parallel...");

        const strategyNames = strategies
          .filter(s => s.name !== 'ALL')
          .map(s => s.name);

        const promises = strategyNames.map(strategyName =>
          axios.post(`${API_URL}/backtest/run`, {
            symbol,
            strategy: strategyName,
            start_date: startDate,
            end_date: endDate,
            initial_capital: initialCapital
          }).catch(error => ({
            error: true,
            strategy: strategyName,
            message: error.response?.data?.detail || error.message
          }))
        );

        const results = await Promise.all(promises);

        // Process results
        const comparisonData = results.map((result, index) => {
          if (result.error) {
            return {
              strategy: result.strategy,
              error: result.message
            };
          }

          // Extract metrics from the response
          // Axios wraps response in .data property
          const metrics = result.data?.metrics || result.metrics || {};
          const strategyName = result.data?.strategy || strategyNames[index];

          console.log('Processing result for:', strategyName, metrics);

          return {
            strategy: strategyName,
            display_name: strategies.find(s => s.name === strategyName)?.display_name,
            total_return_pct: metrics.total_return_pct ?? null,
            cagr_pct: metrics.cagr_pct ?? null,
            sharpe_ratio: metrics.sharpe_ratio ?? null,
            max_drawdown_pct: metrics.max_drawdown_pct ?? null,
            win_rate_pct: metrics.win_rate_pct ?? null,
            total_trades: metrics.total_trades ?? null,
            final_value: metrics.final_value ?? null
          };
        });

        console.log('Comparison data:', comparisonData);

        setComparisonResults(comparisonData);

        // Generate AI insights
        fetchAiInsights(comparisonData);

        fetchHistory();
        toast.success(`Completed backtesting ${strategyNames.length} strategies!`);
      } else {
        // Run single strategy
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
      }
    } catch (error) {
      console.error("Backtest error:", error);
      toast.error(error.response?.data?.detail || "Backtest failed");
    } finally {
      setIsRunning(false);
    }
  };

  const fetchAiInsights = async (comparisonData) => {
    setLoadingInsights(true);
    setAiInsights(null);

    try {
      const response = await axios.post(`${API_URL}/backtest/insights`, {
        symbol: symbol.replace('.NS', '').replace('.BO', ''),
        strategies: comparisonData
      });

      setAiInsights(response.data);
    } catch (error) {
      console.error("Error fetching AI insights:", error);
      // Set fallback insights
      setAiInsights({
        insights: [
          "Review the comparison table above for detailed performance metrics",
          "Consider both absolute returns and risk-adjusted metrics (Sharpe ratio)",
          "Lower maximum drawdown indicates better risk management",
          "Higher win rate doesn't always mean better overall performance",
          "Past performance does not guarantee future results"
        ]
      });
    } finally {
      setLoadingInsights(false);
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

  // Monte Carlo Simulation
  const runMonteCarlo = async () => {
    if (!selectedStrategy) {
      toast.error("Please select a strategy first");
      return;
    }

    setIsRunningMonteCarlo(true);
    setMonteCarloResult(null);

    try {
      const response = await axios.post(`${API_URL}/backtest/monte-carlo`, {
        symbol,
        strategy: selectedStrategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        simulations: monteCarloSimulations
      });

      setMonteCarloResult(response.data);
      toast.success(`Completed ${monteCarloSimulations} Monte Carlo simulations!`);
    } catch (error) {
      console.error("Monte Carlo error:", error);
      toast.error(error.response?.data?.detail || "Monte Carlo simulation failed");
    } finally {
      setIsRunningMonteCarlo(false);
    }
  };

  // Walk-Forward Analysis
  const runWalkForward = async () => {
    if (!selectedStrategy) {
      toast.error("Please select a strategy first");
      return;
    }

    setIsRunningWalkForward(true);
    setWalkForwardResult(null);

    try {
      const response = await axios.post(`${API_URL}/backtest/walk-forward`, {
        symbol,
        strategy: selectedStrategy,
        start_date: startDate,
        end_date: endDate,
        initial_capital: initialCapital,
        windows: walkForwardWindows
      });

      setWalkForwardResult(response.data);
      toast.success(`Completed walk-forward analysis with ${walkForwardWindows} windows!`);
    } catch (error) {
      console.error("Walk-forward error:", error);
      toast.error(error.response?.data?.detail || "Walk-forward analysis failed");
    } finally {
      setIsRunningWalkForward(false);
    }
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
                    <SelectItem value="ALL">
                      🔥 ALL (Compare All Strategies)
                    </SelectItem>
                    <div className="border-t border-[#1F1F1F] my-1"></div>
                    {strategies.map((strategy) => (
                      <SelectItem key={strategy.name} value={strategy.name}>
                        {strategy.display_name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                {selectedStrategy === 'ALL' ? (
                  <p className="text-xs text-text-secondary mt-1">
                    Run all strategies in parallel and compare results
                  </p>
                ) : selectedStrategy && (
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
          {!result && !comparisonResults ? (
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
          ) : comparisonResults ? (
            /* Strategy Comparison Table */
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Card className="card-surface">
                <CardHeader>
                  <CardTitle className="text-sm font-heading">Strategy Comparison</CardTitle>
                  <p className="text-xs text-text-secondary mt-1">
                    Comparison of all strategies for {symbol}
                  </p>
                </CardHeader>
                <CardContent>
                  <div className="overflow-x-auto">
                    <table className="w-full">
                      <thead>
                        <tr className="border-b border-[#1F1F1F]">
                          <th className="text-left py-3 px-4 text-xs font-medium text-text-secondary">Strategy</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Return %</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">CAGR %</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Sharpe</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Max DD %</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Win Rate %</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Trades</th>
                          <th className="text-right py-3 px-4 text-xs font-medium text-text-secondary">Final Value</th>
                        </tr>
                      </thead>
                      <tbody>
                        {comparisonResults
                          .filter(r => !r.error)
                          .sort((a, b) => (b.total_return_pct || 0) - (a.total_return_pct || 0))
                          .map((row, index) => (
                            <tr
                              key={row.strategy}
                              className={`border-b border-[#1F1F1F] hover:bg-surface-highlight transition-colors ${
                                index === 0 ? 'bg-success/5' : ''
                              }`}
                            >
                              <td className="py-3 px-4">
                                <div className="flex items-center gap-2">
                                  {index === 0 && <span className="text-success">🏆</span>}
                                  <span className="text-sm text-text-primary font-medium">
                                    {row.display_name || strategies.find(s => s.name === row.strategy)?.display_name || row.strategy}
                                  </span>
                                </div>
                              </td>
                              <td className={`py-3 px-4 text-right font-data ${
                                row.total_return_pct >= 0 ? 'text-success' : 'text-danger'
                              }`}>
                                {row.total_return_pct?.toFixed(2) || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-text-primary">
                                {row.cagr_pct?.toFixed(2) || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-text-primary">
                                {row.sharpe_ratio?.toFixed(2) || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-danger">
                                {row.max_drawdown_pct?.toFixed(2) || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-text-primary">
                                {row.win_rate_pct?.toFixed(1) || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-text-primary">
                                {row.total_trades || 'N/A'}
                              </td>
                              <td className="py-3 px-4 text-right font-data text-text-primary">
                                ₹{row.final_value?.toLocaleString() || 'N/A'}
                              </td>
                            </tr>
                          ))}
                        {comparisonResults.filter(r => r.error).map((row) => (
                          <tr key={row.strategy} className="border-b border-[#1F1F1F]">
                            <td className="py-3 px-4 text-sm text-text-primary">
                              {strategies.find(s => s.name === row.strategy)?.display_name || row.strategy}
                            </td>
                            <td colSpan="7" className="py-3 px-4 text-sm text-danger">
                              Error: {row.error}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* AI-Powered Insights */}
                  <div className="mt-6 p-4 rounded-lg bg-gradient-to-br from-ai-accent/5 to-primary/5 border border-ai-accent/20">
                    <h4 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
                      <Brain className="w-4 h-4 text-ai-accent" />
                      AI Analysis & Insights
                    </h4>

                    {loadingInsights ? (
                      <div className="flex items-center gap-2 text-sm text-text-secondary py-4">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        <span>AI is analyzing the results...</span>
                      </div>
                    ) : aiInsights ? (
                      <div className="space-y-2 text-sm text-text-secondary">
                        <ul className="space-y-2.5">
                          {aiInsights.insights.map((insight, index) => (
                            <li key={index} className="flex items-start gap-2">
                              <span className="text-ai-accent mt-0.5 flex-shrink-0">✨</span>
                              <span className="text-text-primary leading-relaxed">{insight}</span>
                            </li>
                          ))}
                        </ul>
                        <div className="mt-4 pt-3 border-t border-[#1F1F1F] flex items-center gap-2 text-xs opacity-60">
                          <Brain className="w-3 h-3" />
                          <span>AI-powered analysis by NeuralTrader</span>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </CardContent>
              </Card>
            </motion.div>
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

