import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import { Brain, TrendingUp, TrendingDown, Target, AlertTriangle, RefreshCw, Sparkles } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { API_URL } from "@/config/api";

export default function MLPredictions({ symbol }) {
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (symbol) {
      fetchPrediction();
    }
  }, [symbol]);

  const fetchPrediction = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await axios.get(`${API_URL}/ml/predict/${symbol}`);
      setPrediction(response.data);
    } catch (err) {
      console.error("Error fetching ML prediction:", err);
      setError(err.response?.data?.detail || "Failed to fetch prediction");
    } finally {
      setIsLoading(false);
    }
  };

  const getDirectionColor = (direction) => {
    if (direction === "UP" || direction === "BULLISH") return "text-success";
    if (direction === "DOWN" || direction === "BEARISH") return "text-danger";
    return "text-amber-500";
  };

  const getDirectionBg = (direction) => {
    if (direction === "UP" || direction === "BULLISH") return "bg-success/10 border-success/20";
    if (direction === "DOWN" || direction === "BEARISH") return "bg-danger/10 border-danger/20";
    return "bg-amber-500/10 border-amber-500/20";
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 70) return "text-success";
    if (confidence >= 50) return "text-amber-500";
    return "text-danger";
  };

  if (isLoading) {
    return (
      <Card className="card-surface">
        <CardContent className="py-8">
          <div className="flex items-center justify-center gap-3">
            <RefreshCw className="w-5 h-5 animate-spin text-ai-accent" />
            <span className="text-text-secondary">Loading ML predictions...</span>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="card-surface">
        <CardContent className="py-8">
          <div className="text-center">
            <AlertTriangle className="w-8 h-8 text-amber-500 mx-auto mb-2" />
            <p className="text-text-secondary">{error}</p>
            <Button variant="outline" size="sm" onClick={fetchPrediction} className="mt-3">
              <RefreshCw className="w-3 h-3 mr-2" />
              Retry
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (!prediction) return null;

  return (
    <Card className="card-surface border-ai-accent/20">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-heading flex items-center gap-2">
            <Brain className="w-4 h-4 text-ai-accent" />
            ML Price Prediction
          </CardTitle>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs">
              <Sparkles className="w-3 h-3 mr-1" />
              AI Model
            </Badge>
            <Button variant="ghost" size="sm" onClick={fetchPrediction} className="h-7 w-7 p-0">
              <RefreshCw className="w-3 h-3" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="space-y-4"
        >
          {/* Main Prediction */}
          <div className={`p-4 rounded-lg border ${getDirectionBg(prediction.direction || prediction.prediction)}`}>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {(prediction.direction === "UP" || prediction.prediction === "BULLISH") ? (
                  <TrendingUp className="w-8 h-8 text-success" />
                ) : (prediction.direction === "DOWN" || prediction.prediction === "BEARISH") ? (
                  <TrendingDown className="w-8 h-8 text-danger" />
                ) : (
                  <Target className="w-8 h-8 text-amber-500" />
                )}
                <div>
                  <p className={`text-xl font-bold ${getDirectionColor(prediction.direction || prediction.prediction)}`}>
                    {prediction.direction || prediction.prediction || "NEUTRAL"}
                  </p>
                  <p className="text-sm text-text-secondary">
                    {prediction.timeframe || "5-day"} forecast
                  </p>
                </div>
              </div>
              <div className="text-right">
                <p className="text-sm text-text-secondary">Confidence</p>
                <p className={`text-2xl font-data font-bold ${getConfidenceColor(prediction.confidence || 0)}`}>
                  {(prediction.confidence || 0).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>

          {/* Price Targets */}
          <div className="grid grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-surface-highlight text-center">
              <p className="text-xs text-text-secondary">Current</p>
              <p className="font-data text-text-primary">
                ₹{prediction.current_price?.toLocaleString() || 'N/A'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-success/10 text-center">
              <p className="text-xs text-text-secondary">Target</p>
              <p className="font-data text-success">
                ₹{prediction.predicted_price?.toLocaleString() || prediction.target_price?.toLocaleString() || 'N/A'}
              </p>
            </div>
            <div className="p-3 rounded-lg bg-primary/10 text-center">
              <p className="text-xs text-text-secondary">Change</p>
              <p className={`font-data ${(prediction.expected_change || 0) >= 0 ? 'text-success' : 'text-danger'}`}>
                {(prediction.expected_change || 0) >= 0 ? '+' : ''}{(prediction.expected_change || 0).toFixed(2)}%
              </p>
            </div>
          </div>

          {/* Confidence Breakdown */}
          {prediction.signals && (
            <div className="space-y-2">
              <p className="text-xs text-text-secondary">Signal Breakdown</p>
              {Object.entries(prediction.signals).map(([key, value]) => (
                <div key={key} className="flex items-center justify-between text-sm">
                  <span className="text-text-secondary capitalize">{key.replace(/_/g, ' ')}</span>
                  <div className="flex items-center gap-2">
                    <Progress value={typeof value === 'number' ? value : 50} className="w-24 h-2" />
                    <span className="font-data text-text-primary w-12 text-right">
                      {typeof value === 'number' ? `${value}%` : value}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Model Info */}
          <div className="pt-3 border-t border-[#1F1F1F] flex items-center justify-between text-xs text-text-secondary">
            <span>Model: {prediction.model || 'Ensemble ML'}</span>
            <span>Updated: {prediction.timestamp ? new Date(prediction.timestamp).toLocaleTimeString() : 'Just now'}</span>
          </div>

          {/* Disclaimer */}
          <p className="text-[10px] text-text-secondary/60 italic">
            ML predictions are for informational purposes only. Past performance does not guarantee future results.
          </p>
        </motion.div>
      </CardContent>
    </Card>
  );
}
