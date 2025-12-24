import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, TrendingUp, TrendingDown, Wifi, WifiOff, Zap, RefreshCw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import axios from "axios";

const WS_URL = process.env.REACT_APP_WS_URL || "ws://localhost:8005/ws";
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8005";

export default function LivePriceWidget({ onStockSelect }) {
  const [prices, setPrices] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const [watchedSymbols, setWatchedSymbols] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const wsRef = useRef(null);
  const clientId = useRef(`client_${Date.now()}`);

  // Fetch watched symbols dynamically
  useEffect(() => {
    fetchWatchedSymbols();
  }, []);

  const fetchWatchedSymbols = async () => {
    setIsLoading(true);
    try {
      // First try to get user's watchlist
      const watchlistRes = await axios.get(`${API_URL}/api/watchlist`);

      if (watchlistRes.data && watchlistRes.data.length > 0) {
        // Use watchlist symbols
        const symbols = watchlistRes.data.map(item => item.symbol).slice(0, 10);
        setWatchedSymbols(symbols);
      } else {
        // Fallback: Get top gainers/trending from market
        try {
          const trendingRes = await axios.get(`${API_URL}/api/market/top-movers?limit=10`);
          if (trendingRes.data?.gainers) {
            const symbols = trendingRes.data.gainers.slice(0, 5).map(s => s.symbol);
            const losers = trendingRes.data.losers?.slice(0, 3).map(s => s.symbol) || [];
            setWatchedSymbols([...symbols, ...losers]);
          } else {
            // Final fallback: Get from stock search (top stocks)
            const stocksRes = await axios.get(`${API_URL}/api/stocks/admin/cache-status`);
            if (stocksRes.data?.cached_symbols) {
              setWatchedSymbols(stocksRes.data.cached_symbols.slice(0, 8));
            } else {
              // Last resort defaults
              setWatchedSymbols(["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]);
            }
          }
        } catch {
          // Use defaults if market API fails
          setWatchedSymbols(["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]);
        }
      }
    } catch (error) {
      console.error("Failed to fetch watchlist:", error);
      // Fallback to defaults
      setWatchedSymbols(["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]);
    } finally {
      setIsLoading(false);
    }
  };

  // Connect WebSocket when symbols are loaded
  useEffect(() => {
    if (watchedSymbols.length > 0) {
      connectWebSocket();
    }
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [watchedSymbols]);

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`${WS_URL}/${clientId.current}`);

      ws.onopen = () => {
        setIsConnected(true);
        // Subscribe to symbols
        ws.send(JSON.stringify({
          action: "subscribe",
          symbols: watchedSymbols
        }));
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "ticker") {
            setPrices(prev => ({
              ...prev,
              [data.symbol]: {
                price: data.price,
                change_pct: data.change_pct,
                volume: data.volume,
                timestamp: data.timestamp,
                flash: true
              }
            }));

            // Remove flash after animation
            setTimeout(() => {
              setPrices(prev => ({
                ...prev,
                [data.symbol]: { ...prev[data.symbol], flash: false }
              }));
            }, 500);
          }

          if (data.type === "analysis_alert") {
            setAlerts(prev => [data, ...prev].slice(0, 5));
          }
        } catch (e) {
          console.error("WebSocket parse error:", e);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        // Reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000);
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        setIsConnected(false);
      };

      wsRef.current = ws;
    } catch (error) {
      console.error("WebSocket connection failed:", error);
      setIsConnected(false);
    }
  };

  return (
    <Card className="card-surface">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-heading flex items-center gap-2">
            <Zap className="w-4 h-4 text-ai-accent" />
            Watchlist & Live Prices
            <span className="text-xs text-text-secondary font-normal">
              ({watchedSymbols.length} stocks)
            </span>
          </CardTitle>
          <div className="flex items-center gap-2">
            <button
              onClick={fetchWatchedSymbols}
              className="p-1 hover:bg-surface-highlight rounded transition-colors"
              title="Refresh stocks"
            >
              <RefreshCw className={`w-3 h-3 text-text-secondary ${isLoading ? 'animate-spin' : ''}`} />
            </button>
            <Badge
              variant="outline"
              className={`text-xs ${isConnected ? 'text-success border-success/30' : 'text-danger border-danger/30'}`}
            >
              {isConnected ? (
                <><Wifi className="w-3 h-3 mr-1" /> Live</>
              ) : (
                <><WifiOff className="w-3 h-3 mr-1" /> Offline</>
              )}
            </Badge>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <RefreshCw className="w-6 h-6 text-text-secondary animate-spin" />
          </div>
        ) : watchedSymbols.length === 0 ? (
          <div className="text-center py-8 text-text-secondary text-sm">
            <p>No stocks in watchlist</p>
            <p className="text-xs mt-1">Add stocks to your watchlist to see live prices</p>
          </div>
        ) : (
        <div className="space-y-2">
          {watchedSymbols.map(symbol => {
            const data = prices[symbol];
            const isUp = data?.change_pct >= 0;

            return (
              <motion.button
                key={symbol}
                onClick={() => onStockSelect?.(symbol)}
                className={`w-full flex items-center justify-between p-3 rounded-lg transition-all ${
                  data?.flash
                    ? isUp ? 'bg-success/20' : 'bg-danger/20'
                    : 'hover:bg-surface-highlight'
                }`}
                animate={data?.flash ? { scale: [1, 1.02, 1] } : {}}
                transition={{ duration: 0.3 }}
              >
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${
                    isUp ? 'bg-success/10' : 'bg-danger/10'
                  }`}>
                    {isUp ? (
                      <TrendingUp className="w-4 h-4 text-success" />
                    ) : (
                      <TrendingDown className="w-4 h-4 text-danger" />
                    )}
                  </div>
                  <span className="font-data font-medium text-text-primary">{symbol}</span>
                </div>
                <div className="text-right">
                  <p className="font-data text-text-primary">
                    {data?.price ? `₹${data.price.toLocaleString()}` : '---'}
                  </p>
                  <p className={`text-xs font-data ${isUp ? 'text-success' : 'text-danger'}`}>
                    {data?.change_pct ? `${isUp ? '+' : ''}${data.change_pct.toFixed(2)}%` : '---'}
                  </p>
                </div>
              </motion.button>
            );
          })}
        </div>
        )}

        {/* Live Alerts */}
        {alerts.length > 0 && (
          <div className="mt-4 pt-4 border-t border-[#1F1F1F]">
            <p className="text-xs text-text-secondary mb-2 flex items-center gap-1">
              <Activity className="w-3 h-3" /> Live Analysis Alerts
            </p>
            <AnimatePresence>
              {alerts.slice(0, 3).map((alert, idx) => (
                <motion.div
                  key={idx}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0 }}
                  className="text-xs p-2 rounded bg-ai-accent/10 border border-ai-accent/20 mb-1"
                >
                  <span className="text-ai-accent font-medium">{alert.symbol}</span>
                  <span className="text-text-secondary ml-2">{alert.message}</span>
                </motion.div>
              ))}
            </AnimatePresence>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
