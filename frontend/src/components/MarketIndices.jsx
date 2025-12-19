import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import {
  TrendingUp,
  TrendingDown,
  ArrowUp,
  ArrowDown,
  Minus,
  RefreshCw,
  Activity
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { API_URL } from "@/config/api";

const MarketIndices = ({ onStockSelect }) => {
  const [marketData, setMarketData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [selectedIndex, setSelectedIndex] = useState("NIFTY_50");
  const [lastUpdate, setLastUpdate] = useState(null);

  const fetchMarketData = async (isInitialLoad = false) => {
    try {
      if (isInitialLoad) {
        setLoading(true);
      } else {
        setIsRefreshing(true);
      }
      const response = await axios.get(`${API_URL}/market/overview`);
      setMarketData(response.data);
      setLastUpdate(new Date());
    } catch (error) {
      console.error("Error fetching market data:", error);
    } finally {
      setLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    fetchMarketData(true); // Initial load
    // Refresh every 30 seconds for real-time market data
    const interval = setInterval(() => fetchMarketData(false), 30000);
    return () => clearInterval(interval);
  }, []);

  const getChangeIcon = (change) => {
    if (change > 0) return <ArrowUp className="w-4 h-4 text-success" />;
    if (change < 0) return <ArrowDown className="w-4 h-4 text-error" />;
    return <Minus className="w-4 h-4 text-text-secondary" />;
  };

  const getChangeClass = (change) => {
    if (change > 0) return "text-success";
    if (change < 0) return "text-error";
    return "text-text-secondary";
  };

  const formatNumber = (num) => {
    if (!num) return "0";
    return new Intl.NumberFormat('en-IN').format(num);
  };

  const majorIndices = marketData?.indices?.filter(idx =>
    ["NIFTY_50", "SENSEX", "NIFTY_BANK", "NIFTY_METAL"].includes(idx.name)
  ) || [];

  const selectedMovers = marketData?.market_movers?.[selectedIndex] || { gainers: [], losers: [] };

  return (
    <div className="space-y-4">
      {/* Header with Refresh */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-xl font-heading font-bold text-text-primary flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            Indian Market Overview
          </h2>
          {isRefreshing && (
            <Badge variant="outline" className="gap-1.5 text-xs animate-pulse">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Updating...
            </Badge>
          )}
        </div>
        <Button
          onClick={() => fetchMarketData(false)}
          disabled={isRefreshing}
          size="sm"
          variant="outline"
          className="gap-2"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin' : ''}`} />
          Refresh
        </Button>
      </div>

      {/* Major Indices Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {majorIndices.map((index) => (
          <motion.div
            key={index.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            whileHover={{ scale: 1.02 }}
            className="cursor-pointer"
            onClick={() => setSelectedIndex(index.name)}
          >
            <Card className={`card-surface ${selectedIndex === index.name ? 'border-primary' : ''}`}>
              <CardContent className="p-4">
                <div className="space-y-2">
                  {/* Index Name */}
                  <div className="flex items-center justify-between">
                    <h3 className="text-sm font-medium text-text-secondary">
                      {index.name.replace(/_/g, ' ')}
                    </h3>
                    {getChangeIcon(index.change)}
                  </div>

                  {/* Current Value */}
                  <div className="text-2xl font-bold text-text-primary">
                    {formatNumber(index.current_value)}
                  </div>

                  {/* Change */}
                  <div className={`flex items-center gap-2 text-sm ${getChangeClass(index.change)}`}>
                    <span className="font-medium">
                      {index.change > 0 ? '+' : ''}{index.change?.toFixed(2)}
                    </span>
                    <span className="font-medium">
                      ({index.change_percent > 0 ? '+' : ''}{index.change_percent?.toFixed(2)}%)
                    </span>
                  </div>

                  {/* High/Low */}
                  <div className="flex items-center justify-between text-xs text-text-secondary">
                    <span>H: {formatNumber(index.high)}</span>
                    <span>L: {formatNumber(index.low)}</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        ))}
      </div>

      {/* Gainers and Losers */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Top Gainers */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-success" />
              Top Gainers - {selectedIndex.replace(/_/g, ' ')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {selectedMovers.gainers?.length > 0 ? (
                selectedMovers.gainers.map((stock, index) => (
                  <motion.div
                    key={stock.symbol}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => onStockSelect && onStockSelect(stock.symbol.replace('.NS', '').replace('.BO', ''))}
                    className="flex items-center justify-between p-3 rounded-lg bg-surface-highlight hover:bg-surface-hover transition-colors cursor-pointer"
                  >
                    <div className="flex-1">
                      <div className="font-medium text-text-primary">
                        {stock.symbol.replace('.NS', '').replace('.BO', '')}
                      </div>
                      <div className="text-xs text-text-secondary truncate">
                        {stock.name}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium text-text-primary">
                        ₹{stock.current_price}
                      </div>
                      <div className="flex items-center gap-1 text-sm text-success">
                        <ArrowUp className="w-3 h-3" />
                        <span className="font-medium">{stock.change_percent?.toFixed(2)}%</span>
                      </div>
                    </div>
                  </motion.div>
                ))
              ) : loading && !marketData ? (
                <div className="text-center text-text-secondary py-8">
                  Loading...
                </div>
              ) : (
                <div className="text-center text-text-secondary py-8">
                  No data available
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Top Losers */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <TrendingDown className="w-5 h-5 text-error" />
              Top Losers - {selectedIndex.replace(/_/g, ' ')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {selectedMovers.losers?.length > 0 ? (
                selectedMovers.losers.map((stock, index) => (
                  <motion.div
                    key={stock.symbol}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    onClick={() => onStockSelect && onStockSelect(stock.symbol.replace('.NS', '').replace('.BO', ''))}
                    className="flex items-center justify-between p-3 rounded-lg bg-surface-highlight hover:bg-surface-hover transition-colors cursor-pointer"
                  >
                    <div className="flex-1">
                      <div className="font-medium text-text-primary">
                        {stock.symbol.replace('.NS', '').replace('.BO', '')}
                      </div>
                      <div className="text-xs text-text-secondary truncate">
                        {stock.name}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="font-medium text-text-primary">
                        ₹{stock.current_price}
                      </div>
                      <div className="flex items-center gap-1 text-sm text-error">
                        <ArrowDown className="w-3 h-3" />
                        <span className="font-medium">{stock.change_percent?.toFixed(2)}%</span>
                      </div>
                    </div>
                  </motion.div>
                ))
              ) : loading && !marketData ? (
                <div className="text-center text-text-secondary py-8">
                  Loading...
                </div>
              ) : (
                <div className="text-center text-text-secondary py-8">
                  No data available
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Last Update Time */}
      {lastUpdate && (
        <div className="text-xs text-text-secondary text-center">
          Last updated: {lastUpdate.toLocaleTimeString('en-IN')}
        </div>
      )}
    </div>
  );
};

export default MarketIndices;
