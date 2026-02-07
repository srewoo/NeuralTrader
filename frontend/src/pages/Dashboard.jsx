import { useState, useEffect, useCallback } from "react";
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
  Brain,
  Plus,
  History,
  Clock
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import StockChart from "@/components/StockChart";
import ReasoningLog from "@/components/ReasoningLog";
import MarketIndices from "@/components/MarketIndices";
import NewsWidget from "@/components/NewsWidget";
import { API_URL } from "@/config/api";

export default function Dashboard() {
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [selectedStock, setSelectedStock] = useState(null);
  const [stockData, setStockData] = useState(null);
  const [technicalIndicators, setTechnicalIndicators] = useState(null);
  const [priceHistory, setPriceHistory] = useState(null);
  const [patterns, setPatterns] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [recentAnalyses, setRecentAnalyses] = useState([]);
  const [quickSelectStocks, setQuickSelectStocks] = useState([]);

  // Fetch recent analyses on mount
  useEffect(() => {
    fetchRecentAnalyses();
    fetchQuickSelectStocks();
  }, []);

  const fetchRecentAnalyses = async () => {
    try {
      const response = await axios.get(`${API_URL}/analysis/history?limit=5`);
      setRecentAnalyses(response.data);
    } catch (error) {
      console.error("Error fetching recent analyses:", error);
    }
  };

  const fetchQuickSelectStocks = async () => {
    try {
      const response = await axios.get(`${API_URL}/market/top-movers?limit=5`);
      if (response.data?.gainers) {
        const symbols = response.data.gainers.slice(0, 5).map(s => s.symbol);
        setQuickSelectStocks(symbols);
      }
    } catch (error) {
      try {
        const popularRes = await axios.get(`${API_URL}/stocks/popular?limit=5`);
        if (popularRes.data?.length > 0) {
          setQuickSelectStocks(popularRes.data.map(s => s.symbol));
        }
      } catch {
        setQuickSelectStocks([]);
      }
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
    setPatterns(null);

    try {
      const [stockResponse, indicatorsResponse, historyResponse, patternsResponse] = await Promise.all([
        axios.get(`${API_URL}/stocks/${symbol}`),
        axios.get(`${API_URL}/stocks/${symbol}/indicators`),
        axios.get(`${API_URL}/stocks/${symbol}/history?period=6mo`),
        axios.get(`${API_URL}/patterns/${symbol}`).catch(() => ({ data: null }))
      ]);

      setStockData(stockResponse.data);
      setTechnicalIndicators(indicatorsResponse.data);
      setPriceHistory(historyResponse.data);
      if (patternsResponse.data) setPatterns(patternsResponse.data);
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
      toast.info("Running AI analysis with multi-agent pipeline...");
      const response = await axios.post(`${API_URL}/analyze`, {
        symbol: selectedStock
      });

      setAnalysisResult(response.data);
      fetchRecentAnalyses();
      toast.success("AI analysis complete!");
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
        if (!currentPrice) return "neutral";
        const atrPercent = (value / currentPrice) * 100;
        if (atrPercent > 3) return "bad";
        if (atrPercent < 1) return "good";
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
        {/* Left Sidebar */}
        <div className="lg:col-span-3 space-y-6">
          {/* Recent Analyses */}
          <Card className="card-surface" data-testid="recent-analyses-card">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-heading flex items-center gap-2">
                <History className="w-4 h-4 text-ai-accent" />
                Recent Analyses
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[200px]">
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
                        onClick={() => selectStock(analysis.symbol)}
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
                          {stockData.current_price != null ? `\u20B9${stockData.current_price.toLocaleString()}` : "N/A"}
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
                        {stockData.market_cap ? `\u20B9${(stockData.market_cap / 10000000).toFixed(0)}Cr` : "N/A"}
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
                          ? `\u20B9${stockData.week_52_low?.toFixed(2)} - \u20B9${stockData.week_52_high?.toFixed(2)}`
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
                        { label: "SMA 20", value: technicalIndicators.sma_20, prefix: "\u20B9" },
                        { label: "SMA 50", value: technicalIndicators.sma_50, prefix: "\u20B9" },
                        { label: "BB Upper", value: technicalIndicators.bb_upper, prefix: "\u20B9" },
                        { label: "BB Lower", value: technicalIndicators.bb_lower, prefix: "\u20B9" },
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
              {patterns && patterns.patterns && patterns.patterns.length > 0 && (
                <Card className="card-surface">
                  <CardHeader>
                    <CardTitle className="text-sm font-heading flex items-center gap-2">
                      <Activity className="w-4 h-4 text-ai-accent" />
                      Candlestick Patterns Detected
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex flex-wrap gap-2">
                      {patterns.patterns.map((pattern, idx) => (
                        <Badge
                          key={idx}
                          className={`${
                            pattern.sentiment === "bullish"
                              ? "bg-success/10 text-success border-success/30"
                              : pattern.sentiment === "bearish"
                              ? "bg-danger/10 text-danger border-danger/30"
                              : "bg-amber-500/10 text-amber-500 border-amber-500/30"
                          }`}
                        >
                          {pattern.name || pattern.pattern} ({pattern.confidence || "N/A"}%)
                        </Badge>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              )}

              {/* AI Analysis Section */}
              <Card className="card-surface border-ai-accent/20" data-testid="analysis-section">
                <CardHeader>
                  <div className="flex items-center justify-between">
                    <CardTitle className="text-sm font-heading flex items-center gap-2">
                      <Brain className="w-4 h-4 text-ai-accent" />
                      AI Analysis
                    </CardTitle>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-text-secondary">
                        Multi-agent pipeline (Data + Technical + Reasoning + Validation)
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
                            Analyze with AI
                          </>
                        )}
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  {isAnalyzing && (
                    <div className="flex flex-col items-center justify-center py-12 gap-4">
                      <Brain className="w-12 h-12 text-ai-accent animate-pulse" />
                      <p className="text-text-secondary">AI agents are analyzing {selectedStock}...</p>
                      <div className="flex gap-2 text-xs text-text-secondary">
                        <span className="px-2 py-1 rounded bg-surface-highlight">Collecting Data</span>
                        <span className="px-2 py-1 rounded bg-surface-highlight">Technical Analysis</span>
                        <span className="px-2 py-1 rounded bg-surface-highlight">Deep Reasoning</span>
                        <span className="px-2 py-1 rounded bg-surface-highlight">Validation</span>
                      </div>
                    </div>
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
                                Confidence: {analysisResult.confidence}% {analysisResult.model_used ? `| Model: ${analysisResult.model_used}` : ''}
                              </p>
                            </div>
                          </div>
                        </div>

                        {/* Price Targets */}
                        <div className="grid grid-cols-3 gap-4 mt-6">
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Entry Price</p>
                            <p className="font-data text-lg text-text-primary">
                              {analysisResult.entry_price != null ? `\u20B9${analysisResult.entry_price.toFixed(2)}` : "N/A"}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Target Price</p>
                            <p className="font-data text-lg text-success">
                              {analysisResult.target_price != null ? `\u20B9${analysisResult.target_price.toFixed(2)}` : "N/A"}
                            </p>
                          </div>
                          <div className="p-3 rounded-lg bg-black/20">
                            <p className="text-xs text-text-secondary">Stop Loss</p>
                            <p className="font-data text-lg text-danger">
                              {analysisResult.stop_loss != null ? `\u20B9${analysisResult.stop_loss.toFixed(2)}` : "N/A"}
                            </p>
                          </div>
                        </div>
                      </div>

                      {/* Reasoning */}
                      <ReasoningLog reasoning={analysisResult.reasoning} />

                      {/* Key Risks */}
                      {analysisResult.key_risks && analysisResult.key_risks.length > 0 && (
                        <div className="p-4 rounded-lg bg-danger-dim border border-danger/20">
                          <p className="text-sm font-medium text-danger mb-2">Key Risks</p>
                          <ul className="space-y-1">
                            {analysisResult.key_risks.map((risk, index) => (
                              <li key={index} className="text-sm text-text-secondary flex items-start gap-2">
                                <span className="text-danger">&#8226;</span>
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
                        Click "Analyze with AI" to get recommendations
                      </p>
                      <p className="text-xs text-text-secondary/70 mt-2">
                        Powered by LangGraph multi-agent pipeline
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
                    {quickSelectStocks.map((symbol) => (
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
    </div>
  );
}
