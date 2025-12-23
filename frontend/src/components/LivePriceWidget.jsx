import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Activity, TrendingUp, TrendingDown, Wifi, WifiOff, Zap } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const WS_URL = import.meta.env.VITE_WS_URL || "ws://localhost:8005/ws";

export default function LivePriceWidget({ onStockSelect }) {
  const [prices, setPrices] = useState({});
  const [isConnected, setIsConnected] = useState(false);
  const [alerts, setAlerts] = useState([]);
  const wsRef = useRef(null);
  const clientId = useRef(`client_${Date.now()}`);

  const watchedSymbols = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK"];

  useEffect(() => {
    connectWebSocket();
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

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
            Live Prices
          </CardTitle>
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
      </CardHeader>
      <CardContent>
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
