import { useState, useEffect } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Brain,
  Activity,
  BarChart3,
  Loader2,
  Star,
  Download,
  Share2,
  RefreshCw
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import StockChart from "@/components/StockChart";
import AgentWorkflow from "@/components/AgentWorkflow";
import ReasoningLog from "@/components/ReasoningLog";
import { format } from "date-fns";
import { API_URL } from "@/config/api";

export default function StockDetail() {
  const { symbol, analysisId } = useParams();
  const navigate = useNavigate();
  
  const [analysis, setAnalysis] = useState(null);
  const [stockData, setStockData] = useState(null);
  const [priceHistory, setPriceHistory] = useState(null);
  const [technicalIndicators, setTechnicalIndicators] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isReloading, setIsReloading] = useState(false);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    if (analysisId) {
      fetchAnalysisById(analysisId);
    } else if (symbol) {
      fetchStockDetail(symbol);
    }
  }, [symbol, analysisId]);

  // Auto-refresh stock data every 30 seconds when enabled
  useEffect(() => {
    if (!autoRefresh || (!symbol && !analysis?.symbol)) return;

    const interval = setInterval(() => {
      const targetSymbol = analysis?.symbol || symbol;
      if (targetSymbol) {
        refreshStockData(targetSymbol);
      }
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [autoRefresh, symbol, analysis?.symbol]);

  const refreshStockData = async (stockSymbol) => {
    try {
      const stockResponse = await axios.get(`${API_URL}/stocks/${stockSymbol}`);
      setStockData(stockResponse.data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error refreshing stock data:", error);
    }
  };

  const fetchAnalysisById = async (id) => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_URL}/analysis/${id}`);
      setAnalysis(response.data);
      
      // Fetch additional stock data
      if (response.data.symbol) {
        fetchStockData(response.data.symbol);
      }
    } catch (error) {
      console.error("Error fetching analysis:", error);
      toast.error("Failed to load analysis");
      navigate("/history");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStockDetail = async (stockSymbol) => {
    setIsLoading(true);
    try {
      await fetchStockData(stockSymbol);
    } catch (error) {
      console.error("Error fetching stock detail:", error);
      toast.error("Failed to load stock data");
    } finally {
      setIsLoading(false);
    }
  };

  const fetchStockData = async (stockSymbol) => {
    try {
      const [stockResponse, historyResponse, indicatorsResponse] = await Promise.all([
        axios.get(`${API_URL}/stocks/${stockSymbol}`),
        axios.get(`${API_URL}/stocks/${stockSymbol}/history?period=6mo`),
        axios.get(`${API_URL}/stocks/${stockSymbol}/indicators`)
      ]);

      setStockData(stockResponse.data);
      setPriceHistory(historyResponse.data);
      setTechnicalIndicators(indicatorsResponse.data);
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Error fetching stock data:", error);
      throw error;
    }
  };

  const getSignalIcon = (recommendation) => {
    switch (recommendation?.toUpperCase()) {
      case "BUY":
        return <TrendingUp className="w-8 h-8 text-success" />;
      case "SELL":
        return <TrendingDown className="w-8 h-8 text-danger" />;
      default:
        return <Minus className="w-8 h-8 text-text-secondary" />;
    }
  };

  const getSignalBadge = (recommendation) => {
    switch (recommendation?.toUpperCase()) {
      case "BUY":
        return <Badge className="bg-success-dim text-success border-0 text-lg px-4 py-1">BUY</Badge>;
      case "SELL":
        return <Badge className="bg-danger-dim text-danger border-0 text-lg px-4 py-1">SELL</Badge>;
      default:
        return <Badge className="bg-[#1F1F1F] text-text-secondary border-0 text-lg px-4 py-1">HOLD</Badge>;
    }
  };

  const getIndicatorSignal = (label, value, currentPrice) => {
    if (!value) return "neutral";

    switch (label) {
      case "RSI":
        // RSI: <30 oversold (bullish), >70 overbought (bearish), 30-70 neutral
        if (value < 30) return "good"; // Oversold - potential buy
        if (value > 70) return "bad"; // Overbought - potential sell
        return "neutral";

      case "MACD":
        // MACD: positive is bullish, negative is bearish
        if (value > 5) return "good";
        if (value < -5) return "bad";
        return "neutral";

      case "SMA 20":
      case "SMA 50":
        // Price above SMA is bullish, below is bearish
        if (!currentPrice) return "neutral";
        if (currentPrice > value * 1.02) return "good"; // 2% above SMA
        if (currentPrice < value * 0.98) return "bad"; // 2% below SMA
        return "neutral";

      case "BB Upper":
        // Near upper band can indicate overbought
        if (!currentPrice) return "neutral";
        if (currentPrice > value * 0.99) return "bad"; // Near upper band
        return "neutral";

      case "BB Lower":
        // Near lower band can indicate oversold
        if (!currentPrice) return "neutral";
        if (currentPrice < value * 1.01) return "good"; // Near lower band
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

  const exportAnalysis = () => {
    if (!analysis) return;
    
    const data = JSON.stringify(analysis, null, 2);
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis-${analysis.symbol}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast.success("Analysis exported");
  };

  const shareAnalysis = async () => {
    if (!analysis) return;

    const shareText = `AI Analysis for ${analysis.symbol}: ${analysis.recommendation} Signal with ${analysis.confidence}% confidence`;

    if (navigator.share) {
      try {
        await navigator.share({
          title: `Stock Analysis - ${analysis.symbol}`,
          text: shareText,
          url: window.location.href
        });
      } catch (error) {
        console.error("Error sharing:", error);
      }
    } else {
      navigator.clipboard.writeText(shareText);
      toast.success("Analysis copied to clipboard");
    }
  };

  const reloadAnalysis = async () => {
    const targetSymbol = displaySymbol;
    if (!targetSymbol) {
      toast.error("No stock symbol available");
      return;
    }

    setIsReloading(true);
    try {
      // Request a new analysis
      toast.info("Generating new analysis...");
      const analyzeResponse = await axios.post(`${API_URL}/analyze`, {
        symbol: targetSymbol
      });

      // Fetch the new analysis and stock data
      if (analyzeResponse.data.analysis_id) {
        const [analysisResponse, stockResponse, historyResponse, indicatorsResponse] = await Promise.all([
          axios.get(`${API_URL}/analysis/${analyzeResponse.data.analysis_id}`),
          axios.get(`${API_URL}/stocks/${targetSymbol}`),
          axios.get(`${API_URL}/stocks/${targetSymbol}/history?period=6mo`),
          axios.get(`${API_URL}/stocks/${targetSymbol}/indicators`)
        ]);

        setAnalysis(analysisResponse.data);
        setStockData(stockResponse.data);
        setPriceHistory(historyResponse.data);
        setTechnicalIndicators(indicatorsResponse.data);

        toast.success("Analysis updated successfully");
      }
    } catch (error) {
      console.error("Error reloading analysis:", error);
      toast.error("Failed to reload analysis");
    } finally {
      setIsReloading(false);
    }
  };

  if (isLoading) {
    return (
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="flex items-center justify-center h-[600px]">
          <div className="text-center">
            <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
            <p className="text-text-secondary">Loading analysis...</p>
          </div>
        </div>
      </div>
    );
  }

  const displaySymbol = analysis?.symbol || symbol;
  const displayData = stockData;

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6" data-testid="stock-detail">
      {/* Back Button */}
      <Button
        variant="ghost"
        onClick={() => navigate(-1)}
        className="mb-6 hover:bg-surface-highlight"
      >
        <ArrowLeft className="w-4 h-4 mr-2" />
        Back
      </Button>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Stock Header */}
          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-start justify-between mb-6">
                <div className="flex items-start gap-4">
                  <div className="w-16 h-16 rounded-lg bg-primary/10 flex items-center justify-center">
                    <BarChart3 className="w-8 h-8 text-primary" />
                  </div>
                  <div>
                    <div className="flex items-center gap-3 mb-1">
                      <h1 className="text-3xl font-heading font-bold text-text-primary">
                        {displaySymbol}
                      </h1>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={reloadAnalysis}
                        disabled={isReloading}
                        className="h-8 w-8 p-0 hover:bg-primary/10"
                        title="Reload analysis"
                      >
                        <RefreshCw className={`w-5 h-5 text-primary ${isReloading ? 'animate-spin' : ''}`} />
                      </Button>
                    </div>
                    {displayData && (
                      <>
                        <p className="text-text-secondary">{displayData.name}</p>
                        {displayData.last_updated && (
                          <div className="flex items-center gap-2 mt-1 text-xs text-text-secondary">
                            <span>
                              Updated: {new Date(displayData.last_updated).toLocaleTimeString()}
                            </span>
                            {displayData.is_realtime && (
                              <Badge className="bg-success/10 text-success border-success/20 text-[10px] px-1 py-0">LIVE</Badge>
                            )}
                            {displayData.data_age_minutes > 15 && (
                              <Badge className="bg-amber-500/10 text-amber-500 border-amber-500/20 text-[10px] px-1 py-0">
                                {displayData.data_age_minutes}m ago
                              </Badge>
                            )}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>

                {displayData && (
                  <div className="text-right">
                    <p className="text-3xl font-data font-bold text-text-primary">
                      ₹{displayData.current_price?.toLocaleString()}
                    </p>
                    <div className={`flex items-center justify-end gap-1 ${
                      displayData.change >= 0 ? "text-success" : "text-danger"
                    }`}>
                      {displayData.change >= 0 ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      <span className="font-data text-sm">
                        {displayData.change >= 0 ? "+" : ""}{displayData.change} ({displayData.change_percent}%)
                      </span>
                    </div>
                    <div className="flex items-center justify-end gap-2 mt-2">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setAutoRefresh(!autoRefresh)}
                        className={`text-xs h-7 ${autoRefresh ? 'text-success' : 'text-text-secondary'}`}
                      >
                        <RefreshCw className={`w-3 h-3 mr-1 ${autoRefresh ? 'animate-spin' : ''}`} />
                        Auto-refresh {autoRefresh ? 'ON' : 'OFF'}
                      </Button>
                    </div>
                  </div>
                )}
              </div>

              {/* Quick Stats */}
              {displayData && (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-[#1F1F1F]">
                  <div>
                    <p className="text-xs text-text-secondary">Volume</p>
                    <p className="font-data text-sm text-text-primary">
                      {displayData.volume?.toLocaleString()}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-secondary">Market Cap</p>
                    <p className="font-data text-sm text-text-primary">
                      {displayData.market_cap ? `₹${(displayData.market_cap / 10000000).toFixed(0)}Cr` : "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-secondary">P/E Ratio</p>
                    <p className="font-data text-sm text-text-primary">
                      {displayData.pe_ratio?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-secondary">52W Range</p>
                    <p className="font-data text-sm text-text-primary">
                      ₹{displayData.week_52_low} - ₹{displayData.week_52_high}
                    </p>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Price Chart */}
          {priceHistory && (
            <Card className="card-surface">
              <CardHeader>
                <CardTitle className="text-sm font-heading flex items-center gap-2">
                  <Activity className="w-4 h-4 text-primary" />
                  Price Chart (6 Months)
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="h-[300px]">
                  <StockChart data={priceHistory} />
                </div>
              </CardContent>
            </Card>
          )}

          {/* Technical Indicators */}
          {technicalIndicators && Object.keys(technicalIndicators).length > 0 && (
            <Card className="card-surface">
              <CardHeader>
                <CardTitle className="text-sm font-heading">Technical Indicators</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  {[
                    { label: "RSI", value: technicalIndicators.rsi },
                    { label: "MACD", value: technicalIndicators.macd },
                    { label: "SMA 20", value: technicalIndicators.sma_20, prefix: "₹" },
                    { label: "SMA 50", value: technicalIndicators.sma_50, prefix: "₹" },
                    { label: "BB Upper", value: technicalIndicators.bb_upper, prefix: "₹" },
                    { label: "BB Lower", value: technicalIndicators.bb_lower, prefix: "₹" },
                  ].map((indicator) => {
                    const signal = getIndicatorSignal(indicator.label, indicator.value, displayData?.current_price);
                    const colorClass = getIndicatorColors(signal);
                    const textColorClass = getIndicatorTextColor(signal);

                    return (
                      <div key={indicator.label} className={`p-3 rounded-lg ${colorClass}`}>
                        <p className="text-xs text-text-secondary mb-1">{indicator.label}</p>
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

          {/* AI Analysis Section */}
          {analysis && (
            <>
              {/* Agent Workflow */}
              {analysis.agent_steps && analysis.agent_steps.length > 0 && (
                <Card className="card-surface border-ai-accent/20">
                  <CardHeader>
                    <CardTitle className="text-sm font-heading flex items-center gap-2">
                      <Brain className="w-4 h-4 text-ai-accent" />
                      Agent Workflow
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <AgentWorkflow steps={analysis.agent_steps} />
                  </CardContent>
                </Card>
              )}

              {/* Reasoning Log */}
              {analysis.reasoning && (
                <Card className="card-surface border-ai-accent/20">
                  <CardHeader>
                    <CardTitle className="text-sm font-heading flex items-center gap-2">
                      <Brain className="w-4 h-4 text-ai-accent" />
                      AI Reasoning
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ReasoningLog reasoning={analysis.reasoning} />
                  </CardContent>
                </Card>
              )}
            </>
          )}
        </div>

        {/* Sidebar */}
        <div className="lg:col-span-1 space-y-6">
          {/* Analysis Summary */}
          {analysis ? (
            <Card className="card-surface border-ai-accent/20">
              <CardHeader>
                <CardTitle className="text-sm font-heading flex items-center gap-2">
                  <Brain className="w-4 h-4 text-ai-accent" />
                  AI Analysis Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Signal */}
                <div className={`p-6 rounded-lg text-center ${
                  analysis.recommendation?.toUpperCase() === "BUY" 
                    ? "bg-success-dim border border-success/20" 
                    : analysis.recommendation?.toUpperCase() === "SELL"
                    ? "bg-danger-dim border border-danger/20"
                    : "bg-[#1F1F1F] border border-[#2F2F2F]"
                }`}>
                  {getSignalIcon(analysis.recommendation)}
                  <div className="mt-4">
                    {getSignalBadge(analysis.recommendation)}
                  </div>
                  <p className="text-text-secondary text-sm mt-3">
                    Confidence: {analysis.confidence}%
                  </p>
                </div>

                <Separator className="bg-[#1F1F1F]" />

                {/* Price Targets */}
                <div className="space-y-4">
                  <div>
                    <p className="text-xs text-text-secondary mb-1">Entry Price</p>
                    <p className="font-data text-xl text-text-primary">
                      ₹{analysis.entry_price?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-secondary mb-1">Target Price</p>
                    <p className="font-data text-xl text-success">
                      ₹{analysis.target_price?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-text-secondary mb-1">Stop Loss</p>
                    <p className="font-data text-xl text-danger">
                      ₹{analysis.stop_loss?.toFixed(2) || "N/A"}
                    </p>
                  </div>
                </div>

                <Separator className="bg-[#1F1F1F]" />

                {/* Metadata */}
                <div className="space-y-3 text-sm">
                  <div className="flex items-center justify-between">
                    <span className="text-text-secondary">Model Used</span>
                    <Badge variant="outline" className="text-xs">
                      {analysis.model_used || "Unknown"}
                    </Badge>
                  </div>
                  {analysis.created_at && (
                    <div className="flex items-center gap-2 text-xs text-text-secondary">
                      <Calendar className="w-3.5 h-3.5" />
                      <span>
                        {format(new Date(analysis.created_at), "MMM dd, yyyy 'at' hh:mm a")}
                      </span>
                    </div>
                  )}
                </div>

                <Separator className="bg-[#1F1F1F]" />

                {/* Actions */}
                <div className="space-y-2">
                  <Button
                    onClick={exportAnalysis}
                    variant="outline"
                    className="w-full border-[#1F1F1F] hover:bg-surface-highlight"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Export Analysis
                  </Button>
                  <Button
                    onClick={shareAnalysis}
                    variant="outline"
                    className="w-full border-[#1F1F1F] hover:bg-surface-highlight"
                  >
                    <Share2 className="w-4 h-4 mr-2" />
                    Share
                  </Button>
                </div>
              </CardContent>
            </Card>
          ) : (
            <Card className="card-surface border-ai-accent/20">
              <CardHeader>
                <CardTitle className="text-sm font-heading flex items-center gap-2">
                  <Brain className="w-4 h-4 text-ai-accent" />
                  AI Analysis
                </CardTitle>
              </CardHeader>
              <CardContent className="text-center py-12">
                <Brain className="w-12 h-12 mx-auto text-ai-accent/50 mb-4" />
                <p className="text-text-secondary mb-4">No AI analysis available yet</p>
                <Button
                  onClick={reloadAnalysis}
                  disabled={isReloading}
                  className="btn-primary"
                >
                  {isReloading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Analyzing...
                    </>
                  ) : (
                    <>
                      <Brain className="w-4 h-4 mr-2" />
                      Run AI Analysis
                    </>
                  )}
                </Button>
              </CardContent>
            </Card>
          )}

          {/* Key Risks */}
          {analysis && analysis.key_risks && analysis.key_risks.length > 0 && (
            <Card className="card-surface border-danger/20">
              <CardHeader>
                <CardTitle className="text-sm font-heading text-danger">Key Risks</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-auto max-h-[200px]">
                  <ul className="space-y-2">
                    {analysis.key_risks.map((risk, index) => (
                      <li key={index} className="text-sm text-text-secondary flex items-start gap-2">
                        <span className="text-danger mt-0.5">•</span>
                        <span>{risk}</span>
                      </li>
                    ))}
                  </ul>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

