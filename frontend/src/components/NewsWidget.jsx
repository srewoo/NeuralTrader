import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { Newspaper, TrendingUp, TrendingDown, Minus, ExternalLink, RefreshCw, Clock } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { API_URL } from "@/config/api";

export default function NewsWidget({ symbol = null }) {
  const [news, setNews] = useState([]);
  const [sentiment, setSentiment] = useState(null);
  const [trending, setTrending] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [activeTab, setActiveTab] = useState("news");

  useEffect(() => {
    fetchNews();
    if (!symbol) {
      fetchTrending();
    }
  }, [symbol]);

  const fetchNews = async () => {
    setIsLoading(true);
    try {
      if (symbol) {
        // Fetch symbol-specific news with sentiment
        const [newsRes, sentimentRes] = await Promise.all([
          axios.get(`${API_URL}/news/comprehensive/${symbol}`),
          axios.get(`${API_URL}/news/sentiment/${symbol}`)
        ]);
        setNews(newsRes.data.articles || []);
        setSentiment(sentimentRes.data);
      } else {
        // Fetch general market news
        const response = await axios.get(`${API_URL}/news/market`);
        setNews(response.data.articles || response.data || []);
      }
    } catch (error) {
      console.error("Error fetching news:", error);
      setNews([]);
    } finally {
      setIsLoading(false);
    }
  };

  const fetchTrending = async () => {
    try {
      const response = await axios.get(`${API_URL}/news/trending`);
      setTrending(response.data.trending || response.data.topics || []);
    } catch (error) {
      console.error("Error fetching trending:", error);
      setTrending([]);
    }
  };

  const getSentimentIcon = (sentimentScore) => {
    if (sentimentScore > 0.2) return <TrendingUp className="w-4 h-4 text-success" />;
    if (sentimentScore < -0.2) return <TrendingDown className="w-4 h-4 text-danger" />;
    return <Minus className="w-4 h-4 text-amber-500" />;
  };

  const getSentimentColor = (sentimentScore) => {
    if (sentimentScore > 0.2) return "text-success";
    if (sentimentScore < -0.2) return "text-danger";
    return "text-amber-500";
  };

  const getSentimentBadge = (sentimentScore) => {
    if (sentimentScore > 0.2) return <Badge className="bg-success/10 text-success border-0">Bullish</Badge>;
    if (sentimentScore < -0.2) return <Badge className="bg-danger/10 text-danger border-0">Bearish</Badge>;
    return <Badge className="bg-amber-500/10 text-amber-500 border-0">Neutral</Badge>;
  };

  const formatTime = (timestamp) => {
    if (!timestamp) return "";
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now - date;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMs / 3600000);

    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    return date.toLocaleDateString();
  };

  return (
    <Card className="card-surface">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-heading flex items-center gap-2">
            <Newspaper className="w-4 h-4 text-primary" />
            {symbol ? `${symbol} News` : "Market News"}
          </CardTitle>
          <div className="flex items-center gap-2">
            {!symbol && (
              <div className="flex rounded-lg bg-surface-highlight p-0.5">
                <button
                  onClick={() => setActiveTab("news")}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    activeTab === "news" ? "bg-primary text-white" : "text-text-secondary hover:text-text-primary"
                  }`}
                >
                  Latest
                </button>
                <button
                  onClick={() => setActiveTab("trending")}
                  className={`px-3 py-1 text-xs rounded-md transition-colors ${
                    activeTab === "trending" ? "bg-primary text-white" : "text-text-secondary hover:text-text-primary"
                  }`}
                >
                  Trending
                </button>
              </div>
            )}
            <Button variant="ghost" size="sm" onClick={fetchNews} className="h-7 w-7 p-0">
              <RefreshCw className={`w-3 h-3 ${isLoading ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Sentiment Summary for Symbol */}
        {symbol && sentiment && (
          <div className="mb-4 p-3 rounded-lg bg-surface-highlight">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getSentimentIcon(sentiment.overall_sentiment || 0)}
                <span className="text-sm text-text-primary">Overall Sentiment</span>
              </div>
              {getSentimentBadge(sentiment.overall_sentiment || 0)}
            </div>
            <div className="mt-2 grid grid-cols-3 gap-2 text-xs">
              <div className="text-center">
                <p className="text-text-secondary">Positive</p>
                <p className="text-success font-medium">{sentiment.positive_count || 0}</p>
              </div>
              <div className="text-center">
                <p className="text-text-secondary">Neutral</p>
                <p className="text-amber-500 font-medium">{sentiment.neutral_count || 0}</p>
              </div>
              <div className="text-center">
                <p className="text-text-secondary">Negative</p>
                <p className="text-danger font-medium">{sentiment.negative_count || 0}</p>
              </div>
            </div>
          </div>
        )}

        {/* News List */}
        <ScrollArea className="h-[300px]">
          {isLoading ? (
            <div className="flex items-center justify-center h-full">
              <RefreshCw className="w-6 h-6 animate-spin text-text-secondary" />
            </div>
          ) : activeTab === "news" ? (
            <div className="space-y-3">
              {news.length === 0 ? (
                <p className="text-text-secondary text-sm text-center py-8">No news available</p>
              ) : (
                news.slice(0, 10).map((article, idx) => (
                  <motion.a
                    key={idx}
                    href={article.url || article.link}
                    target="_blank"
                    rel="noopener noreferrer"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: idx * 0.05 }}
                    className="block p-3 rounded-lg hover:bg-surface-highlight transition-colors group"
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1">
                        <h4 className="text-sm font-medium text-text-primary group-hover:text-primary transition-colors line-clamp-2">
                          {article.title}
                        </h4>
                        <div className="flex items-center gap-2 mt-1">
                          <span className="text-xs text-text-secondary">{article.source}</span>
                          {article.published && (
                            <>
                              <span className="text-text-secondary">•</span>
                              <span className="text-xs text-text-secondary flex items-center gap-1">
                                <Clock className="w-3 h-3" />
                                {formatTime(article.published)}
                              </span>
                            </>
                          )}
                          {article.sentiment !== undefined && (
                            <>
                              <span className="text-text-secondary">•</span>
                              <span className={`text-xs ${getSentimentColor(article.sentiment)}`}>
                                {article.sentiment > 0.2 ? "Positive" : article.sentiment < -0.2 ? "Negative" : "Neutral"}
                              </span>
                            </>
                          )}
                        </div>
                      </div>
                      <ExternalLink className="w-4 h-4 text-text-secondary opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0" />
                    </div>
                  </motion.a>
                ))
              )}
            </div>
          ) : (
            /* Trending Topics */
            <div className="space-y-2">
              {trending.length === 0 ? (
                <p className="text-text-secondary text-sm text-center py-8">No trending topics</p>
              ) : (
                trending.map((topic, idx) => (
                  <div
                    key={idx}
                    className="p-3 rounded-lg bg-surface-highlight flex items-center justify-between"
                  >
                    <div className="flex items-center gap-2">
                      <span className="text-lg">#{idx + 1}</span>
                      <span className="text-sm text-text-primary">{topic.keyword || topic}</span>
                    </div>
                    {topic.count && (
                      <Badge variant="outline" className="text-xs">
                        {topic.count} mentions
                      </Badge>
                    )}
                  </div>
                ))
              )}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
