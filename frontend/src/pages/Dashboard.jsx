import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import { 
  Search, 
  TrendingUp, 
  TrendingDown, 
  Minus,
  BarChart3,
  Activity,
  Loader2,
  ArrowRight,
  Brain,
  Star,
  Plus
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ScrollArea } from "@/components/ui/scroll-area";
import StockChart from "@/components/StockChart";
import AgentWorkflow from "@/components/AgentWorkflow";
import ReasoningLog from "@/components/ReasoningLog";
import CandlestickPatterns from "@/components/CandlestickPatterns";
import MarketIndices from "@/components/MarketIndices";
import LivePriceWidget from "@/components/LivePriceWidget";
import NewsWidget from "@/components/NewsWidget";
import InstitutionalActivity from "@/components/InstitutionalActivity";
import AIAssistant from "@/components/AIAssistant";
import { API_URL } from "@/config/api";

export default function Dashboard() {
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [stockData, setStockData] = useState(null);
  const [technicalIndicators, setTechnicalIndicators] = useState(null);
  const [priceHistory, setPriceHistory] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [watchlist, setWatchlist] = useState([]);
  const [recentAnalyses, setRecentAnalyses] = useState([]);

  // Fetch watchlist and recent analyses on mount
  useEffect(() => {
    fetchWatchlist();
    fetchRecentAnalyses();
  }, []);

  const fetchWatchlist = async () => {
    try {
      const response = await axios.get(`${API_URL}/watchlist`);
      setWatchlist(response.data);
    } catch (error) {
      console.error("Error fetching watchlist:", error);
    }
  };

  const fetchRecentAnalyses = async () => {
    try {
      const response = await axios.get(`${API_URL}/analysis/history?limit=5`);
      setRecentAnalyses(response.data);
    } catch (error) {
      console.error("Error fetching recent analyses:", error);
    }
  };

  const handleSearch = useCallback(async (query) => {
    if (!query || query.length < 1) {
      setSearchResults([]);
      return;
    }
    
    try {
      const response = await axios.get(`${API_URL}/stocks/search?q=${query}`);
      setSearchResults(response.data);
    } catch (error) {
      console.error("Search error:", error);
    }
  }, []);

  useEffect(() => {
    const debounce = setTimeout(() => {
      handleSearch(searchQuery);
    }, 300);
    return () => clearTimeout(debounce);
  }, [searchQuery, handleSearch]);

  const selectStock = async (symbol) => {
    setSelectedStock(symbol);
    setSearchQuery("");
    setSearchResults([]);
    setIsLoading(true);
    setAnalysisResult(null);

    try {
      const [stockResponse, indicatorsResponse, historyResponse] = await Promise.all([
        axios.get(`${API_URL}/stocks/${symbol}`),
        axios.get(`${API_URL}/stocks/${symbol}/indicators`),
        axios.get(`${API_URL}/stocks/${symbol}/history?period=6mo`)
      ]);

      setStockData(stockResponse.data);
      setTechnicalIndicators(indicatorsResponse.data);
      setPriceHistory(historyResponse.data);
    } catch (error) {
      toast.error("Failed to fetch stock data");
      console.error(error);
    } finally {
      setIsLoading(false);
    }
  };

  const runAnalysis = async () => {
    if (!selectedStock) {
      toast.error("Please select a stock first");
      return;
    }

    setIsAnalyzing(true);
    setAnalysisResult(null);

    try {
      toast.info("Running ensemble analysis with multiple AI models...");
      const response = await axios.post(`${API_URL}/analyze/ensemble`, {
        symbol: selectedStock
      });

      setAnalysisResult(response.data);
      fetchRecentAnalyses();
      toast.success("Ensemble analysis complete!");
    } catch (error) {
      const errorMsg = error.response?.data?.detail || "Analysis failed";
      toast.error(errorMsg);
      console.error(error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const addToWatchlist = async (symbol) => {
    try {
      await axios.post(`${API_URL}/watchlist/${symbol}`);
      fetchWatchlist();
      toast.success(`${symbol} added to watchlist`);
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to add to watchlist");
    }
  };

  const getSignalBadge = (recommendation) => {
    switch (recommendation) {
      case "BUY":
        return <Badge className="bg-success-dim text-success border-0">BUY</Badge>;
      case "SELL":
        return <Badge className="bg-danger-dim text-danger border-0">SELL</Badge>;
      default:
        return <Badge className="bg-[#1F1F1F] text-text-secondary border-0">HOLD</Badge>;
    }
  };

  const getIndicatorSignal = (label, value, currentPrice) => {
    if (!value) return "neutral";

    switch (label) {
      case "RSI":
        if (value < 30) return "good";
        if (value > 70) return "bad";
        return "neutral";

      case "MACD":
        if (value > 5) return "good";
        if (value < -5) return "bad";
        return "neutral";

      case "SMA 20":
      case "SMA 50":
        if (!currentPrice) return "neutral";
        if (currentPrice > value * 1.02) return "good";
        if (currentPrice < value * 0.98) return "bad";
        return "neutral";

      case "BB Upper":
        if (!currentPrice) return "neutral";
        if (currentPrice > value * 0.99) return "bad";
        return "neutral";

      case "BB Lower":
        if (!currentPrice) return "neutral";
        if (currentPrice < value * 1.01) return "good";
        return "neutral";

      case "ATR":
        // ATR indicates volatility - high ATR is risky (amber/bad), low ATR is stable (neutral/good)
        if (!currentPrice) return "neutral";
        const atrPercent = (value / currentPrice) * 100;
        if (atrPercent > 3) return "bad"; // High volatility
        if (atrPercent < 1) return "good"; // Low volatility
        return "neutral";

      default:
        return "neutral";
    }
  };

  const getIndicatorColors = (signal) => {
    switch (signal) {
      case "good":
        return "bg-success/10 border border-success/20";
      case "bad":
        return "bg-danger/10 border border-danger/20";
      default:
        return "bg-amber-500/10 border border-amber-500/20";
    }
  };

  const getIndicatorTextColor = (signal) => {
    switch (signal) {
      case "good":
        return "text-success";
      case "bad":
        return "text-danger";
      default:
        return "text-amber-500";
    }
  };

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6" data-testid="dashboard">
      {/* Market Indices Overview */}
      <div className="mb-8">
        <MarketIndices onStockSelect={selectStock} />
      </div>

      {/* Search Section */}
      <div className="mb-8">
        <div className="relative max-w-2xl">
          <div className="relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
            <Input
              data-testid="stock-search-input"
              placeholder="Search NSE/BSE stocks (e.g., RELIANCE, TCS, INFY)"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="pl-12 h-14 bg-surface border-[#1F1F1F] text-lg text-text-primary placeholder:text-text-secondary focus:border-primary"
            />
          </div>
          
          {/* Search Results Dropdown */}
          <AnimatePresence>
            {searchResults.length > 0 && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="absolute z-50 w-full mt-2 bg-surface border border-[#1F1F1F] rounded-lg shadow-xl overflow-hidden"
              >
                {searchResults.map((stock) => (
                  <button
                    key={stock.symbol}
                    data-testid={`search-result-${stock.symbol}`}
                    onClick={() => selectStock(stock.symbol)}
                    className="w-full flex items-center justify-between px-4 py-3 hover:bg-surface-highlight transition-colors text-left"
                  >
                    <div>
                      <span className="font-data font-medium text-text-primary">{stock.symbol}</span>
                      <span className="text-text-secondary ml-2 text-sm">{stock.name}</span>
                    </div>
                    <Badge variant="outline" className="text-xs">{stock.exchange}</Badge>
                  </button>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Sidebar - Live Prices, Watchlist, News, FII/DII */}
        <div className="lg:col-span-3 space-y-6">
          {/* Live Prices Widget */}
          <LivePriceWidget onStockSelect={selectStock} />

          {/* Watchlist */}
          <Card className="card-surface" data-testid="watchlist-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-heading flex items-center gap-2">
                <Star className="w-4 h-4 text-yellow-500" />
                Watchlist
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[150px]">
                {watchlist.length === 0 ? (
                  <p className="text-text-secondary text-sm text-center py-4">
                    No stocks in watchlist
                  </p>
                ) : (
                  <div className="space-y-2">
                    {watchlist.map((item) => (
                      <button
                        key={item.symbol}
                        data-testid={`watchlist-${item.symbol}`}
                        onClick={() => navigate(`/stock/${item.symbol}`)}
                        className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-surface-highlight transition-colors"
                      >
                        <span className="font-data text-text-primary">{item.symbol}</span>
                        <ArrowRight className="w-4 h-4 text-text-secondary" />
                      </button>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Recent Analyses */}
          <Card className="card-surface" data-testid="recent-analyses-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-heading flex items-center gap-2">
                <History className="w-4 h-4 text-ai-accent" />
                Recent Analyses
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[150px]">
                {recentAnalyses.length === 0 ? (
                  <p className="text-text-secondary text-sm text-center py-4">
                    No recent analyses
                  </p>
                ) : (
                  <div className="space-y-2">
                    {recentAnalyses.map((analysis) => (
                      <button
                        key={analysis.id}
                        data-testid={`recent-analysis-${analysis.id}`}
                        onClick={() => navigate(`/analysis/${analysis.id}`)}
                        className="w-full flex items-center justify-between p-2 rounded-lg hover:bg-surface-highlight transition-colors"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-data text-text-primary">{analysis.symbol}</span>
                          {getSignalBadge(analysis.recommendation)}
                        </div>
                        <span className="text-xs text-text-secondary font-data">
                          {analysis.confidence}%
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>

          {/* Market News */}
          <NewsWidget />

          {/* FII/DII Institutional Activity */}
          <InstitutionalActivity />
        </div>

        {/* Main Content */}
        <div className="lg:col-span-9 space-y-6">
          {/* Stock Overview & Chart */}
          {stockData ? (
            <>
              {/* Stock Header */}
              <Card className="card-surface" data-testid="stock-overview-card">
                <CardContent className="pt-6">
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
                        <BarChart3 className="w-6 h-6 text-primary" />
                      </div>
                      <div>
                        <div className="flex items-center gap-3">
                          <h2 className="text-2xl font-heading font-bold text-text-primary">
                            {stockData.symbol}
                          </h2>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => addToWatchlist(stockData.symbol)}
                            data-testid="add-watchlist-btn"
                          >
                            <Plus className="w-4 h-4" />
                          </Button>
                        </div>
                        <p className="text-text-secondary">{stockData.name}</p>
                      </div>
                    </div>
                    
                    <div className="flex items-end gap-6">
                      <div className="text-right">
                        <p className="text-3xl font-data font-bold text-text-primary">
                          ₹{stockData.current_price?.toLocaleString()}
                        </p>
                        <div className={`flex items-center justify-end gap-1 ${
                          stockData.change >= 0 ? "text-success" : "text-danger"
                        }`}>
                          {stockData.change >= 0 ? (
                            <TrendingUp className="w-4 h-4" />
                          ) : (
                            <TrendingDown className="w-4 h-4" />
                          )}
                          <span className="font-data">
                            {stockData.change >= 0 ? "+" : ""}{stockData.change} ({stockData.change_percent}%)
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Quick Stats */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6 pt-6 border-t border-[#1F1F1F]">
                    <div>
                      <p className="text-xs text-text-secondary">Volume</p>
                      <p className="font-data text-text-primary">{stockData.volume?.toLocaleString()}</p>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">Market Cap</p>
                      <p className="font-data text-text-primary">
                        {stockData.market_cap ? `₹${(stockData.market_cap / 10000000).toFixed(0)}Cr` : "N/A"}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">P/E Ratio</p>
                      <p className="font-data text-text-primary">{stockData.pe_ratio?.toFixed(2) || "N/A"}</p>
                    </div>
                    <div>
                      <p className="text-xs text-text-secondary">52W Range</p>
                      <p className="font-data text-text-primary">
                        {stockData.week_52_low && stockData.week_52_high
                          ? `₹${stockData.week_52_low?.toFixed(2)} - ₹${stockData.week_52_high?.toFixed(2)}`
                          : "N/A"}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Chart */}
              <Card className="card-surface" data-testid="stock-chart-card">
                <CardHeader>
                  <CardTitle className="text-sm font-heading flex items-center gap-2">
                    <Activity className="w-4 h-4 text-primary" />
                    Price Chart (6 Months)
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="h-[300px]">
                    {priceHistory ? (
                      <StockChart data={priceHistory} />
                    ) : (
                      <div className="h-full flex items-center justify-center">
                        <Loader2 className="w-8 h-8 animate-spin text-text-secondary" />
                      </div>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Technical Indicators */}
              {technicalIndicators && Object.keys(technicalIndicators).length > 0 && (
                <Card className="card-surface" data-testid="technical-indicators-card">
                  <CardHeader>
                    <CardTitle className="text-sm font-heading">Technical Indicators</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
                      {[
                        { label: "RSI", value: technicalIndicators.rsi },
                        { label: "MACD", value: technicalIndicators.macd },
                        { label: "SMA 20", value: technicalIndicators.sma_20, prefix: "₹" },
                        { label: "SMA 50", value: technicalIndicators.sma_50, prefix: "₹" },
                        { label: "BB Upper", value: technicalIndicators.bb_upper, prefix: "₹" },
                        { label: "BB Lower", value: technicalIndicators.bb_lower, prefix: "₹" },
                        { label: "ATR", value: technicalIndicators.atr },
                      ].map((indicator) => {
                        const signal = getIndicatorSignal(indicator.label, indicator.value, stockData?.current_price);
                        const colorClass = getIndicatorColors(signal);
                        const textColorClass = getIndicatorTextColor(signal);

                        return (
                          <div key={indicator.label} className={`p-3 rounded-lg ${colorClass}`}>
                            <p className="text-xs text-text-secondary">{indicator.label}</p>
                            <p className={`font-data ${textColorClass} font-semibold`}>
                              {indicator.prefix || ""}{indicator.value?.toFixed(2) || "N/A"}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* Candlestick Patterns */}
              <CandlestickPatterns symbol={selectedStock} />

              {/* Analysis Section */}
              <Card className="card-surface border-ai-accent/20" data-testid="analysis-section">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-heading flex items-center gap-2">
                      <Brain className="w-4 h-4 text-ai-accent" />
                      AI Ensemble Analysis
                    </CardTitle>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-text-secondary">
                        Multi-model consensus (OpenAI + Gemini + Claude)
                      </span>
                      <Button
                        onClick={runAnalysis}
                        disabled={isAnalyzing}
                        className="btn-primary"
                        data-testid="analyze-btn"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Analyzing...
                          </>
                        ) : (
                          <>
                            <Brain className="w-4 h-4 mr-2" />
                            Run Analysis
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {isAnalyzing && (
                    <AgentWorkflow isRunning={true} />
                  )}

                  {analysisResult && (
                    <motion.div
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-6"
                    >
                      {/* Signal Banner */}
                      <div className={`p-6 rounded-lg ${
                        analysisResult.recommendation === "BUY" 
                          ? "bg-success-dim border border-success/20" 
                          : analysisResult.recommendation === "SELL"
                          ? "bg-danger-dim border border-danger/20"
                          : "bg-[#1F1F1F] border border-[#2F2F2F]"
                      }`}>
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            {analysisResult.recommendation === "BUY" ? (
                              <TrendingUp className="w-8 h-8 text-success" />
                            ) : analysisResult.recommendation === "SELL" ? (
                              <TrendingDown className="w-8 h-8 text-danger" />
                            ) : (
                              <Minus className="w-8 h-8 text-text-secondary" />
                            )}
                            <div>
                              <p className={`text-2xl font-heading font-bold ${
                                analysisResult.recommendation === "BUY" 
                                  ? "text-success" 
                                  : analysisResult.recommendation === "SELL"
                                  ? "text-danger"
                                  : "text-text-secondary"
                              }`}>
                                {analysisResult.recommendation} Signal
                              </p>
                              <p className="text-text-secondary">
                                Confidence: {analysisResult.confidence}% • Model: {analysisResult.model_used}
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Price Targets */}
                        <div className="grid grid-cols-3 gap-4 mt-6">
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Entry Price</p>
                            <p className="font-data text-lg text-text-primary">
                              ₹{analysisResult.entry_price?.toFixed(2) || "N/A"}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Target Price</p>
                            <p className="font-data text-lg text-success">
                              ₹{analysisResult.target_price?.toFixed(2) || "N/A"}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Stop Loss</p>
                            <p className="font-data text-lg text-danger">
                              ₹{analysisResult.stop_loss?.toFixed(2) || "N/A"}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Agent Steps */}
                      {analysisResult.agent_steps && (
                        <AgentWorkflow steps={analysisResult.agent_steps} />
                      )}

                      {/* Reasoning */}
                      <ReasoningLog reasoning={analysisResult.reasoning} />

                      {/* Key Risks */}
                      {analysisResult.key_risks && analysisResult.key_risks.length > 0 && (
                        <div className="p-4 rounded-lg bg-danger-dim border border-danger/20">
                          <p className="text-sm font-medium text-danger mb-2">Key Risks</p>
                          <ul className="space-y-1">
                            {analysisResult.key_risks.map((risk, index) => (
                              <li key={index} className="text-sm text-text-secondary flex items-start gap-2">
                                <span className="text-danger">•</span>
                                {risk}
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                    </motion.div>
                  )}

                  {!isAnalyzing && !analysisResult && (
                    <div className="text-center py-12">
                      <Brain className="w-12 h-12 mx-auto text-ai-accent/50 mb-4" />
                      <p className="text-text-secondary">
                        Click "Run Analysis" to get AI ensemble recommendations
                      </p>
                      <p className="text-xs text-text-secondary/70 mt-2">
                        Powered by OpenAI, Gemini & Claude
                      </p>
                    </div>
                  )}
                </CardContent>
              </Card>
            </>
          ) : (
            /* Empty State */
            <Card className="card-surface" data-testid="empty-state">
              <CardContent className="py-24">
                <div className="text-center">
                  <div className="w-20 h-20 mx-auto mb-6 rounded-2xl bg-primary/10 flex items-center justify-center">
                    <Search className="w-10 h-10 text-primary" />
                  </div>
                  <h3 className="text-xl font-heading font-bold text-text-primary mb-2">
                    Search for a Stock
                  </h3>
                  <p className="text-text-secondary max-w-md mx-auto">
                    Enter a stock symbol or name to view real-time data, technical indicators, 
                    and get AI-powered trading recommendations.
                  </p>
                  <div className="mt-6 flex flex-wrap justify-center gap-2">
                    {["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"].map((symbol) => (
                      <Button
                        key={symbol}
                        variant="outline"
                        size="sm"
                        onClick={() => selectStock(symbol)}
                        data-testid={`quick-select-${symbol}`}
                        className="border-[#1F1F1F] hover:bg-surface-highlight"
                      >
                        {symbol}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {isLoading && (
            <Card className="card-surface">
              <CardContent className="py-12">
                <div className="flex items-center justify-center gap-3">
                  <Loader2 className="w-6 h-6 animate-spin text-primary" />
                  <span className="text-text-secondary">Loading stock data...</span>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      </div>

      {/* AI Assistant Floating Chat */}
      <AIAssistant />
    </div>
  );
}

// Import History icon for the component
import { History } from "lucide-react";
