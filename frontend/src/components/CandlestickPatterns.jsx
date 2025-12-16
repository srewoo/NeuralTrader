import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "./ui/card";
import { Badge } from "./ui/badge";
import { Alert, AlertDescription } from "./ui/alert";
import { Button } from "./ui/button";
import { 
  TrendingUp, 
  TrendingDown, 
  Minus, 
  AlertCircle,
  Info,
  RefreshCw
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { BACKEND_URL } from "@/config/api";

/**
 * CandlestickPatterns Component
 * Displays detected candlestick patterns with real pattern detection
 */
export default function CandlestickPatterns({ symbol }) {
  const [patterns, setPatterns] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedPeriod, setSelectedPeriod] = useState(30);

  useEffect(() => {
    if (symbol) {
      fetchPatterns();
    }
  }, [symbol, selectedPeriod]);

  const fetchPatterns = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${BACKEND_URL}/api/patterns/${symbol}?days=${selectedPeriod}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch patterns');
      }

      const data = await response.json();
      setPatterns(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const getPatternIcon = (type) => {
    if (type.includes('bullish')) return <TrendingUp className="w-4 h-4" />;
    if (type.includes('bearish')) return <TrendingDown className="w-4 h-4" />;
    return <Minus className="w-4 h-4" />;
  };

  const getPatternColor = (type) => {
    if (type.includes('bullish')) return 'bg-green-500/10 text-green-600 border-green-500/20';
    if (type.includes('bearish')) return 'bg-red-500/10 text-red-600 border-red-500/20';
    return 'bg-yellow-500/10 text-yellow-600 border-yellow-500/20';
  };

  const getStrengthBadge = (strength) => {
    const colors = {
      strong: 'bg-purple-500/20 text-purple-700 border-purple-500/30',
      medium: 'bg-blue-500/20 text-blue-700 border-blue-500/30',
      weak: 'bg-gray-500/20 text-gray-700 border-gray-500/30'
    };
    return colors[strength] || colors.weak;
  };

  const formatPatternName = (name) => {
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (loading) {
    return (
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle>Candlestick Patterns</CardTitle>
          <CardDescription>Analyzing price patterns...</CardDescription>
        </CardHeader>
        <CardContent className="flex items-center justify-center py-12">
          <RefreshCw className="w-8 h-8 animate-spin text-primary" />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle>Candlestick Patterns</CardTitle>
        </CardHeader>
        <CardContent>
          <Alert variant="destructive">
            <AlertCircle className="w-4 h-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
          <Button onClick={fetchPatterns} className="mt-4">
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        </CardContent>
      </Card>
    );
  }

  if (!patterns) {
    return null;
  }

  return (
    <div className="space-y-6">
      {/* Summary Card */}
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle>Candlestick Patterns</CardTitle>
              <CardDescription>
                Real-time pattern detection for {symbol}
              </CardDescription>
            </div>
            <div className="flex gap-1.5">
              {[1, 7, 15, 30, 60].map((days) => (
                <Button
                  key={days}
                  variant={selectedPeriod === days ? "default" : "outline"}
                  size="sm"
                  onClick={() => setSelectedPeriod(days)}
                  className="px-2.5"
                >
                  {days}d
                </Button>
              ))}
              <Button
                variant="outline"
                size="sm"
                onClick={fetchPatterns}
              >
                <RefreshCw className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent>
          {/* Pattern Counts */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-green-500/10 p-4 rounded-lg border border-green-500/20">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-5 h-5 text-green-600" />
                <span className="text-sm font-medium text-green-600">Bullish</span>
              </div>
              <div className="text-2xl font-bold text-green-700">
                {patterns.pattern_counts.bullish}
              </div>
            </div>

            <div className="bg-red-500/10 p-4 rounded-lg border border-red-500/20">
              <div className="flex items-center gap-2 mb-2">
                <TrendingDown className="w-5 h-5 text-red-600" />
                <span className="text-sm font-medium text-red-600">Bearish</span>
              </div>
              <div className="text-2xl font-bold text-red-700">
                {patterns.pattern_counts.bearish}
              </div>
            </div>

            <div className="bg-yellow-500/10 p-4 rounded-lg border border-yellow-500/20">
              <div className="flex items-center gap-2 mb-2">
                <Minus className="w-5 h-5 text-yellow-600" />
                <span className="text-sm font-medium text-yellow-600">Indecision</span>
              </div>
              <div className="text-2xl font-bold text-yellow-700">
                {patterns.pattern_counts.indecision}
              </div>
            </div>
          </div>

          {/* Latest Signal */}
          {patterns.latest_signal && (
            <Alert className={`${getPatternColor(patterns.latest_signal.type)} border`}>
              <Info className="w-4 h-4" />
              <AlertDescription>
                <div className="font-semibold mb-1">
                  Latest Signal: {formatPatternName(patterns.latest_signal.pattern)}
                </div>
                <div className="text-sm opacity-90">
                  {patterns.latest_signal.description}
                </div>
                {/* Trading Implication for Latest Signal */}
                {patterns.latest_signal.implication && (
                  <div className={`mt-2 p-2 rounded ${
                    patterns.latest_signal.implication.direction === 'bullish'
                      ? 'bg-green-500/20 border border-green-500/30'
                      : patterns.latest_signal.implication.direction === 'bearish'
                      ? 'bg-red-500/20 border border-red-500/30'
                      : 'bg-yellow-500/20 border border-yellow-500/30'
                  }`}>
                    <div className="font-semibold text-sm">{patterns.latest_signal.implication.signal}</div>
                    <div className="text-xs opacity-90 mt-0.5">{patterns.latest_signal.implication.meaning}</div>
                  </div>
                )}
                <div className="text-xs opacity-75 mt-2">
                  Detected on {patterns.latest_signal.date_formatted || new Date(patterns.latest_signal.date).toLocaleDateString()}
                  at {patterns.latest_signal.price_formatted || `₹${patterns.latest_signal.price.toFixed(2)}`}
                </div>
              </AlertDescription>
            </Alert>
          )}
        </CardContent>
      </Card>

      {/* Recent Patterns */}
      {patterns.recent_patterns && patterns.recent_patterns.length > 0 && (
        <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
          <CardHeader>
            <CardTitle className="text-lg">Recent Patterns (Last 5 Days)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              <AnimatePresence>
                {patterns.recent_patterns.map((pattern, index) => (
                  <motion.div
                    key={`${pattern.date}-${pattern.pattern}`}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={`p-4 rounded-lg border ${getPatternColor(pattern.type)}`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-3">
                        <div className="mt-1">
                          {getPatternIcon(pattern.type)}
                        </div>
                        <div>
                          <div className="font-semibold">
                            {formatPatternName(pattern.pattern)}
                          </div>
                          <div className="text-sm opacity-90 mt-1">
                            {pattern.description}
                          </div>
                          {/* Trading Implication */}
                          {pattern.implication && (
                            <div className={`mt-2 p-2 rounded text-sm ${
                              pattern.implication.direction === 'bullish'
                                ? 'bg-green-500/10 border border-green-500/20'
                                : pattern.implication.direction === 'bearish'
                                ? 'bg-red-500/10 border border-red-500/20'
                                : 'bg-yellow-500/10 border border-yellow-500/20'
                            }`}>
                              <div className="font-medium">{pattern.implication.signal}</div>
                              <div className="text-xs opacity-80 mt-0.5">{pattern.implication.meaning}</div>
                            </div>
                          )}
                          <div className="flex gap-4 mt-2 text-xs opacity-75">
                            <span>Date: {pattern.date_formatted || new Date(pattern.date).toLocaleDateString()}</span>
                            <span>Price: {pattern.price_formatted || `₹${pattern.price.toFixed(2)}`}</span>
                          </div>
                        </div>
                      </div>
                      <Badge
                        variant="outline"
                        className={getStrengthBadge(pattern.strength)}
                      >
                        {pattern.strength}
                      </Badge>
                    </div>
                  </motion.div>
                ))}
              </AnimatePresence>
            </div>
          </CardContent>
        </Card>
      )}

      {/* All Patterns */}
      <Card className="border-border/40 bg-card/50 backdrop-blur-sm">
        <CardHeader>
          <CardTitle className="text-lg">
            All Detected Patterns ({patterns.total_patterns})
          </CardTitle>
          <CardDescription>
            Patterns detected over the last {selectedPeriod} days
          </CardDescription>
        </CardHeader>
        <CardContent>
          {patterns.all_patterns.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              No patterns detected in this period
            </div>
          ) : (
            <div className="space-y-2 max-h-96 overflow-y-auto">
              {patterns.all_patterns.map((pattern, index) => (
                <div
                  key={`all-${pattern.date}-${pattern.pattern}-${index}`}
                  className={`p-3 rounded-lg border ${getPatternColor(pattern.type)}`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      {getPatternIcon(pattern.type)}
                      <div>
                        <div className="font-medium text-sm">
                          {formatPatternName(pattern.pattern)}
                        </div>
                        <div className="text-xs opacity-75">
                          {pattern.date_formatted || new Date(pattern.date).toLocaleDateString()} - {pattern.price_formatted || `₹${pattern.price.toFixed(2)}`}
                        </div>
                      </div>
                    </div>
                    <Badge
                      variant="outline"
                      className={`${getStrengthBadge(pattern.strength)} text-xs`}
                    >
                      {pattern.strength}
                    </Badge>
                  </div>
                  {/* Implication for all patterns */}
                  {pattern.implication && (
                    <div className={`mt-2 p-2 rounded text-xs ${
                      pattern.implication.direction === 'bullish'
                        ? 'bg-green-500/5'
                        : pattern.implication.direction === 'bearish'
                        ? 'bg-red-500/5'
                        : 'bg-yellow-500/5'
                    }`}>
                      <span className="font-medium">{pattern.implication.signal}</span>
                      <span className="opacity-75 ml-2">{pattern.implication.meaning}</span>
                    </div>
                  )}
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

