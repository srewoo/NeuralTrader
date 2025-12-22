import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import {
  Brain,
  TrendingUp,
  TrendingDown,
  RefreshCw,
  Loader2,
  AlertCircle,
  ArrowUpRight,
  ArrowDownRight,
  BarChart3,
  Clock,
  Filter,
  ChevronRight,
  Sparkles,
  Target,
  Activity,
  Zap,
  Play
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { API_URL } from "@/config/api";

export default function AIRecommends() {
  const navigate = useNavigate();
  const [recommendations, setRecommendations] = useState(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState(null);
  const [selectedSector, setSelectedSector] = useState("all");

  useEffect(() => {
    loadCachedRecommendations();
  }, []);

  // Load cached recommendations from database
  const loadCachedRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_URL}/recommendations`);
      setRecommendations(response.data);
    } catch (err) {
      console.error("Error loading recommendations:", err);
      setError(err.response?.data?.detail || "Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  };

  // Generate fresh ENHANCED recommendations
  const generateRecommendations = async () => {
    setGenerating(true);
    setError(null);

    try {
      toast.info("🔍 Analyzing NIFTY stocks with enhanced AI...\n✅ Sentiment enabled\n✅ Backtest enabled\n✅ 65%+ confidence\n\nTakes 30-60 seconds...");

      const response = await axios.post(`${API_URL}/recommendations/generate/enhanced`, null, {
        params: {
          limit: 30,
          min_confidence: 65.0,
          enable_sentiment: true,
          enable_backtest: true
        }
      });

      setRecommendations(response.data);

      const summary = response.data.summary;
      toast.success(
        `✅ Enhanced analysis complete!\n📈 ${summary.total_buy_signals} BUY (${summary.avg_buy_confidence}% avg)\n📉 ${summary.total_sell_signals} SELL (${summary.avg_sell_confidence}% avg)\n💹 Market: ${summary.market_sentiment}`,
        { duration: 5000 }
      );
    } catch (err) {
      console.error("Error generating recommendations:", err);
      setError(err.response?.data?.detail || "Failed to generate recommendations");
      toast.error("Failed to generate recommendations");
    } finally {
      setGenerating(false);
    }
  };

  // Format the generated date
  const formatGeneratedDate = (dateStr) => {
    if (!dateStr) return null;
    const date = new Date(dateStr);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);
    const diffDays = Math.floor(diffMs / 86400000);

    if (diffMins < 1) return "Just now";
    if (diffMins < 60) return `${diffMins} minute${diffMins > 1 ? 's' : ''} ago`;
    if (diffHours < 24) return `${diffHours} hour${diffHours > 1 ? 's' : ''} ago`;
    if (diffDays < 7) return `${diffDays} day${diffDays > 1 ? 's' : ''} ago`;
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
  };

  // Available sectors in Indian markets
  const INDIAN_MARKET_SECTORS = [
    "Banking",
    "IT",
    "Energy",
    "Automotive",
    "Pharma",
    "FMCG",
    "Telecom",
    "Metals",
    "Infrastructure",
    "Cement",
    "Financial Services",
    "Insurance",
    "Chemicals",
    "Consumer Durables",
    "Media",
    "Real Estate",
    "Healthcare",
    "Power",
    "Capital Goods",
    "Retail"
  ];

  const getSectors = () => {
    // Get sectors from recommendations
    const recommendedSectors = new Set();
    [...(recommendations?.buy_recommendations || []), ...(recommendations?.sell_recommendations || [])].forEach(r => {
      if (r.sector && r.sector !== "N/A") recommendedSectors.add(r.sector);
    });

    // If recommendations have sectors, use those; otherwise show all available sectors
    if (recommendedSectors.size > 0) {
      return Array.from(recommendedSectors).sort();
    }

    // Return default sectors if no recommendations yet
    return INDIAN_MARKET_SECTORS;
  };

  const filterBySector = (items) => {
    if (selectedSector === "all") return items;
    return items.filter(item => item.sector === selectedSector);
  };

  const handleStockClick = (symbol) => {
    navigate(`/stock/${symbol}`);
  };

  if (loading) {
    return (
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-lg bg-ai-accent/10 flex items-center justify-center">
              <Brain className="w-6 h-6 text-ai-accent" />
            </div>
            <div>
              <h1 className="text-3xl font-heading font-bold text-text-primary">
                AI Recommendations
              </h1>
              <p className="text-text-secondary">
                Loading recommendations...
              </p>
            </div>
          </div>
        </div>

        <Card className="card-surface">
          <CardContent className="py-24">
            <div className="flex flex-col items-center justify-center gap-4">
              <Loader2 className="w-12 h-12 animate-spin text-ai-accent" />
              <p className="text-text-secondary">Loading cached recommendations...</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Show generating state overlay
  if (generating) {
    return (
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-12 h-12 rounded-lg bg-ai-accent/10 flex items-center justify-center">
              <Brain className="w-6 h-6 text-ai-accent" />
            </div>
            <div>
              <h1 className="text-3xl font-heading font-bold text-text-primary">
                AI Recommendations
              </h1>
              <p className="text-text-secondary">
                Generating fresh analysis...
              </p>
            </div>
          </div>
        </div>

        <Card className="card-surface">
          <CardContent className="py-24">
            <div className="flex flex-col items-center justify-center gap-4">
              <div className="relative">
                <Brain className="w-16 h-16 text-ai-accent animate-pulse" />
                <Sparkles className="w-6 h-6 text-yellow-500 absolute -top-1 -right-1 animate-bounce" />
              </div>
              <div className="text-center">
                <p className="text-lg font-medium text-text-primary mb-2">
                  AI is analyzing 100 stocks...
                </p>
                <p className="text-sm text-text-secondary">
                  Scanning RSI, MACD, Moving Averages, Bollinger Bands & more
                </p>
              </div>
              <Loader2 className="w-8 h-8 animate-spin text-ai-accent mt-4" />
              <p className="text-xs text-text-secondary mt-2">This may take 30-60 seconds</p>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
        <Card className="card-surface">
          <CardContent className="py-12">
            <div className="text-center">
              <AlertCircle className="w-12 h-12 mx-auto text-danger mb-4" />
              <p className="text-text-primary mb-4">{error}</p>
              <Button onClick={loadCachedRecommendations}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Retry
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  const buyRecs = filterBySector(recommendations?.buy_recommendations || []);
  const sellRecs = filterBySector(recommendations?.sell_recommendations || []);
  const hasRecommendations = recommendations?.generated_at && (buyRecs.length > 0 || sellRecs.length > 0);

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-lg bg-ai-accent/10 flex items-center justify-center">
              <Brain className="w-6 h-6 text-ai-accent" />
            </div>
            <div>
              <h1 className="text-3xl font-heading font-bold text-text-primary">
                AI Recommendations
              </h1>
              <p className="text-text-secondary">
                Technical analysis of top 100 NSE/BSE stocks
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Select value={selectedSector} onValueChange={setSelectedSector}>
              <SelectTrigger className="w-[180px] bg-surface-highlight border-[#1F1F1F]">
                <Filter className="w-4 h-4 mr-2" />
                <SelectValue placeholder="Filter by sector" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All Sectors</SelectItem>
                {getSectors().map(sector => (
                  <SelectItem key={sector} value={sector}>{sector}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button
              onClick={generateRecommendations}
              disabled={generating}
              className="bg-ai-accent hover:bg-ai-accent/90"
            >
              {generating ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-4 h-4 mr-2" />
                  Generate New
                </>
              )}
            </Button>
          </div>
        </div>

        {recommendations?.generated_at && (
          <div className="flex items-center gap-2 mt-4 text-sm text-text-secondary">
            <Clock className="w-4 h-4" />
            Last analyzed: {formatGeneratedDate(recommendations.generated_at)}
          </div>
        )}
      </div>

      {/* Show empty state if no recommendations */}
      {!hasRecommendations && !recommendations?.message && (
        <Card className="card-surface mb-8">
          <CardContent className="py-16">
            <div className="text-center">
              <div className="relative inline-block mb-4">
                <Brain className="w-16 h-16 text-ai-accent/50" />
                <Sparkles className="w-6 h-6 text-yellow-500/50 absolute -top-1 -right-1" />
              </div>
              <h3 className="text-lg font-medium text-text-primary mb-2">
                No Recommendations Yet
              </h3>
              <p className="text-text-secondary mb-6 max-w-md mx-auto">
                Click the "Generate New" button to analyze 100 top NSE/BSE stocks
                and get AI-powered buy/sell recommendations based on technical indicators.
              </p>
              <Button
                onClick={generateRecommendations}
                disabled={generating}
                className="bg-ai-accent hover:bg-ai-accent/90"
                size="lg"
              >
                <Zap className="w-5 h-5 mr-2" />
                Generate Recommendations
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {recommendations?.message && !hasRecommendations && (
        <Card className="card-surface mb-8">
          <CardContent className="py-16">
            <div className="text-center">
              <div className="relative inline-block mb-4">
                <Brain className="w-16 h-16 text-ai-accent/50" />
                <Sparkles className="w-6 h-6 text-yellow-500/50 absolute -top-1 -right-1" />
              </div>
              <h3 className="text-lg font-medium text-text-primary mb-2">
                {recommendations.message}
              </h3>
              <Button
                onClick={generateRecommendations}
                disabled={generating}
                className="bg-ai-accent hover:bg-ai-accent/90 mt-4"
                size="lg"
              >
                <Zap className="w-5 h-5 mr-2" />
                Generate Recommendations
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {hasRecommendations && (
        <>
      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
        <Card className="card-surface">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Stocks Analyzed</p>
                <p className="text-2xl font-data font-bold text-text-primary">
                  {recommendations?.total_stocks_analyzed || 0}
                </p>
              </div>
              <BarChart3 className="w-8 h-8 text-primary" />
            </div>
          </CardContent>
        </Card>

        <Card className="card-surface border-success/30">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Buy Signals</p>
                <p className="text-2xl font-data font-bold text-success">
                  {recommendations?.summary?.total_buy_signals || 0}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-success" />
            </div>
          </CardContent>
        </Card>

        <Card className="card-surface border-danger/30">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Sell Signals</p>
                <p className="text-2xl font-data font-bold text-danger">
                  {recommendations?.summary?.total_sell_signals || 0}
                </p>
              </div>
              <TrendingDown className="w-8 h-8 text-danger" />
            </div>
          </CardContent>
        </Card>

        <Card className="card-surface">
          <CardContent className="py-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Market Sentiment</p>
                <p className={`text-2xl font-data font-bold ${
                  recommendations?.summary?.market_sentiment === 'Bullish' ? 'text-success' :
                  recommendations?.summary?.market_sentiment === 'Bearish' ? 'text-danger' :
                  'text-text-primary'
                }`}>
                  {recommendations?.summary?.market_sentiment || 'Neutral'}
                </p>
              </div>
              <Activity className="w-8 h-8 text-ai-accent" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recommendations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Buy Recommendations */}
        <Card className="card-surface border-success/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-lg bg-success/10 flex items-center justify-center">
                  <TrendingUp className="w-5 h-5 text-success" />
                </div>
                <div>
                  <CardTitle className="text-lg text-success">Buy Recommendations</CardTitle>
                  <CardDescription>Stocks showing bullish signals</CardDescription>
                </div>
              </div>
              <Badge className="bg-success/20 text-success border-0">
                {buyRecs.length} stocks
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] pr-4">
              {buyRecs.length === 0 ? (
                <div className="text-center py-12 text-text-secondary">
                  <TrendingUp className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>No strong buy signals found</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <AnimatePresence>
                    {buyRecs.map((rec, index) => (
                      <motion.div
                        key={rec.symbol}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.03 }}
                        onClick={() => handleStockClick(rec.symbol)}
                        className="p-4 rounded-lg bg-success/5 border border-success/20 hover:border-success/40 cursor-pointer transition-all hover:bg-success/10"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-data font-bold text-text-primary">
                                {rec.symbol}
                              </span>
                              <Badge variant="outline" className="text-xs">
                                {rec.sector}
                              </Badge>
                            </div>
                            <p className="text-sm text-text-secondary truncate max-w-[200px]">
                              {rec.name}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="font-data font-bold text-text-primary">
                              ₹{rec.current_price?.toLocaleString()}
                            </p>
                            <p className={`text-sm ${rec.change_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                              {rec.change_pct >= 0 ? '+' : ''}{rec.change_pct}%
                            </p>
                          </div>
                        </div>

                        <div className="mb-3">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-text-secondary">Confidence</span>
                            <span className="text-success font-medium">{rec.confidence}%</span>
                          </div>
                          <Progress value={rec.confidence} className="h-2 bg-success/20" />
                        </div>

                        <div className="flex flex-wrap gap-1.5">
                          {rec.signals?.slice(0, 3).map((signal, idx) => (
                            <Badge
                              key={idx}
                              variant="outline"
                              className="text-xs bg-success/10 border-success/30 text-success"
                            >
                              {signal}
                            </Badge>
                          ))}
                          {rec.signals?.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{rec.signals.length - 3} more
                            </Badge>
                          )}
                        </div>

                        <div className="flex items-center justify-between mt-3 pt-3 border-t border-success/20">
                          <div className="flex gap-3 text-xs text-text-secondary">
                            <span>RSI: {rec.indicators?.rsi}</span>
                            <span>ADX: {rec.indicators?.adx}</span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-success" />
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>

        {/* Sell Recommendations */}
        <Card className="card-surface border-danger/20">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <div className="w-10 h-10 rounded-lg bg-danger/10 flex items-center justify-center">
                  <TrendingDown className="w-5 h-5 text-danger" />
                </div>
                <div>
                  <CardTitle className="text-lg text-danger">Sell Recommendations</CardTitle>
                  <CardDescription>Stocks showing bearish signals</CardDescription>
                </div>
              </div>
              <Badge className="bg-danger/20 text-danger border-0">
                {sellRecs.length} stocks
              </Badge>
            </div>
          </CardHeader>
          <CardContent>
            <ScrollArea className="h-[500px] pr-4">
              {sellRecs.length === 0 ? (
                <div className="text-center py-12 text-text-secondary">
                  <TrendingDown className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>No strong sell signals found</p>
                </div>
              ) : (
                <div className="space-y-3">
                  <AnimatePresence>
                    {sellRecs.map((rec, index) => (
                      <motion.div
                        key={rec.symbol}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: index * 0.03 }}
                        onClick={() => handleStockClick(rec.symbol)}
                        className="p-4 rounded-lg bg-danger/5 border border-danger/20 hover:border-danger/40 cursor-pointer transition-all hover:bg-danger/10"
                      >
                        <div className="flex items-start justify-between mb-3">
                          <div>
                            <div className="flex items-center gap-2">
                              <span className="font-data font-bold text-text-primary">
                                {rec.symbol}
                              </span>
                              <Badge variant="outline" className="text-xs">
                                {rec.sector}
                              </Badge>
                            </div>
                            <p className="text-sm text-text-secondary truncate max-w-[200px]">
                              {rec.name}
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="font-data font-bold text-text-primary">
                              ₹{rec.current_price?.toLocaleString()}
                            </p>
                            <p className={`text-sm ${rec.change_pct >= 0 ? 'text-success' : 'text-danger'}`}>
                              {rec.change_pct >= 0 ? '+' : ''}{rec.change_pct}%
                            </p>
                          </div>
                        </div>

                        <div className="mb-3">
                          <div className="flex items-center justify-between text-xs mb-1">
                            <span className="text-text-secondary">Confidence</span>
                            <span className="text-danger font-medium">{rec.confidence}%</span>
                          </div>
                          <Progress value={rec.confidence} className="h-2 bg-danger/20" />
                        </div>

                        <div className="flex flex-wrap gap-1.5">
                          {rec.signals?.slice(0, 3).map((signal, idx) => (
                            <Badge
                              key={idx}
                              variant="outline"
                              className="text-xs bg-danger/10 border-danger/30 text-danger"
                            >
                              {signal}
                            </Badge>
                          ))}
                          {rec.signals?.length > 3 && (
                            <Badge variant="outline" className="text-xs">
                              +{rec.signals.length - 3} more
                            </Badge>
                          )}
                        </div>

                        <div className="flex items-center justify-between mt-3 pt-3 border-t border-danger/20">
                          <div className="flex gap-3 text-xs text-text-secondary">
                            <span>RSI: {rec.indicators?.rsi}</span>
                            <span>ADX: {rec.indicators?.adx}</span>
                          </div>
                          <ChevronRight className="w-4 h-4 text-danger" />
                        </div>
                      </motion.div>
                    ))}
                  </AnimatePresence>
                </div>
              )}
            </ScrollArea>
          </CardContent>
        </Card>
      </div>

      {/* Disclaimer */}
      <Card className="card-surface mt-6">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-text-secondary">
              <p className="font-medium text-text-primary mb-1">Disclaimer</p>
              <p>
                These recommendations are generated based on technical indicators only and should not be considered as financial advice.
                Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
                Past performance does not guarantee future results.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
        </>
      )}
    </div>
  );
}
