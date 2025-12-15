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
  Activity
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
  const [error, setError] = useState(null);
  const [selectedSector, setSelectedSector] = useState("all");
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    fetchRecommendations();
  }, []);

  const fetchRecommendations = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await axios.get(`${API_URL}/recommendations`);
      setRecommendations(response.data);
      setLastUpdated(new Date());
      toast.success(`Analyzed ${response.data.total_stocks_analyzed} stocks`);
    } catch (err) {
      console.error("Error fetching recommendations:", err);
      setError(err.response?.data?.detail || "Failed to fetch recommendations");
      toast.error("Failed to load recommendations");
    } finally {
      setLoading(false);
    }
  };

  const getSectors = () => {
    if (!recommendations) return [];
    const sectors = new Set();
    [...(recommendations.buy_recommendations || []), ...(recommendations.sell_recommendations || [])].forEach(r => {
      if (r.sector) sectors.add(r.sector);
    });
    return Array.from(sectors).sort();
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
                Analyzing top 100 NSE/BSE stocks...
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
                  AI is analyzing stocks...
                </p>
                <p className="text-sm text-text-secondary">
                  Scanning technical indicators across 100 stocks
                </p>
              </div>
              <Loader2 className="w-8 h-8 animate-spin text-ai-accent mt-4" />
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
              <Button onClick={fetchRecommendations}>
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
              variant="outline"
              onClick={fetchRecommendations}
              disabled={loading}
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>
        </div>

        {lastUpdated && (
          <div className="flex items-center gap-2 mt-4 text-sm text-text-secondary">
            <Clock className="w-4 h-4" />
            Last updated: {lastUpdated.toLocaleTimeString()}
          </div>
        )}
      </div>

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
    </div>
  );
}
