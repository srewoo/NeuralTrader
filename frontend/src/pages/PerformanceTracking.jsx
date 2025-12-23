import { useState, useEffect } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import {
  Target,
  TrendingUp,
  TrendingDown,
  Award,
  BarChart3,
  Clock,
  CheckCircle,
  XCircle,
  RefreshCw,
  Loader2,
  Trophy,
  Percent
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { API_URL } from "@/config/api";

export default function PerformanceTracking() {
  const [accuracy, setAccuracy] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const [accuracyRes, predictionsRes, leaderboardRes] = await Promise.all([
        axios.get(`${API_URL}/tracking/accuracy`),
        axios.get(`${API_URL}/tracking/predictions?limit=50`),
        axios.get(`${API_URL}/tracking/leaderboard`)
      ]);
      setAccuracy(accuracyRes.data);
      setPredictions(predictionsRes.data.predictions || predictionsRes.data || []);
      setLeaderboard(leaderboardRes.data.strategies || leaderboardRes.data || []);
    } catch (error) {
      console.error("Error fetching tracking data:", error);
    } finally {
      setIsLoading(false);
    }
  };

  const getAccuracyColor = (value) => {
    if (value >= 70) return "text-success";
    if (value >= 50) return "text-amber-500";
    return "text-danger";
  };

  const getResultBadge = (result) => {
    if (result === "CORRECT" || result === "WIN") {
      return <Badge className="bg-success/10 text-success border-0">Correct</Badge>;
    }
    if (result === "INCORRECT" || result === "LOSS") {
      return <Badge className="bg-danger/10 text-danger border-0">Incorrect</Badge>;
    }
    return <Badge className="bg-amber-500/10 text-amber-500 border-0">Pending</Badge>;
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <Loader2 className="w-8 h-8 animate-spin text-primary" />
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
              <Target className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-heading font-bold text-text-primary">Performance Tracking</h1>
              <p className="text-text-secondary">Track prediction accuracy and strategy performance</p>
            </div>
          </div>
          <Button onClick={fetchData} variant="outline" className="flex items-center gap-2">
            <RefreshCw className="w-4 h-4" />
            Refresh
          </Button>
        </div>

        {/* Accuracy Overview Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Overall Accuracy</p>
                  <p className={`text-3xl font-bold ${getAccuracyColor(accuracy?.overall || 0)}`}>
                    {(accuracy?.overall || 0).toFixed(1)}%
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Percent className="w-6 h-6 text-primary" />
                </div>
              </div>
              <Progress value={accuracy?.overall || 0} className="mt-3 h-2" />
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Total Predictions</p>
                  <p className="text-3xl font-bold text-text-primary">
                    {accuracy?.total_predictions || predictions.length || 0}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-ai-accent/10 flex items-center justify-center">
                  <BarChart3 className="w-6 h-6 text-ai-accent" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Correct</p>
                  <p className="text-3xl font-bold text-success">
                    {accuracy?.correct || 0}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-success/10 flex items-center justify-center">
                  <CheckCircle className="w-6 h-6 text-success" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Incorrect</p>
                  <p className="text-3xl font-bold text-danger">
                    {accuracy?.incorrect || 0}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-danger/10 flex items-center justify-center">
                  <XCircle className="w-6 h-6 text-danger" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Breakdown by Type */}
        {accuracy?.by_type && (
          <Card className="card-surface">
            <CardHeader>
              <CardTitle className="text-lg font-heading flex items-center gap-2">
                <Award className="w-5 h-5 text-ai-accent" />
                Accuracy by Signal Type
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {Object.entries(accuracy.by_type).map(([type, data]) => (
                  <div key={type} className="p-4 rounded-lg bg-surface-highlight">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-text-primary capitalize">{type}</span>
                      <Badge className={`${
                        type === 'BUY' ? 'bg-success/10 text-success' :
                        type === 'SELL' ? 'bg-danger/10 text-danger' :
                        'bg-amber-500/10 text-amber-500'
                      } border-0`}>
                        {type}
                      </Badge>
                    </div>
                    <p className={`text-2xl font-bold ${getAccuracyColor(data.accuracy || 0)}`}>
                      {(data.accuracy || 0).toFixed(1)}%
                    </p>
                    <p className="text-xs text-text-secondary mt-1">
                      {data.correct || 0} / {data.total || 0} correct
                    </p>
                    <Progress value={data.accuracy || 0} className="mt-2 h-1.5" />
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        <Tabs defaultValue="predictions" className="w-full">
          <TabsList className="grid w-full grid-cols-2 mb-4">
            <TabsTrigger value="predictions">Recent Predictions</TabsTrigger>
            <TabsTrigger value="leaderboard">Strategy Leaderboard</TabsTrigger>
          </TabsList>

          <TabsContent value="predictions">
            <Card className="card-surface">
              <CardHeader>
                <CardTitle className="text-lg font-heading flex items-center gap-2">
                  <Clock className="w-5 h-5 text-primary" />
                  Recent Predictions
                </CardTitle>
                <CardDescription>Track how our predictions performed over time</CardDescription>
              </CardHeader>
              <CardContent>
                {predictions.length === 0 ? (
                  <p className="text-text-secondary text-center py-8">No predictions tracked yet</p>
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead>Symbol</TableHead>
                        <TableHead>Signal</TableHead>
                        <TableHead>Entry</TableHead>
                        <TableHead>Target</TableHead>
                        <TableHead>Actual</TableHead>
                        <TableHead>Return</TableHead>
                        <TableHead>Result</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {predictions.slice(0, 20).map((pred, idx) => (
                        <TableRow key={idx}>
                          <TableCell className="text-xs">
                            {pred.date ? new Date(pred.date).toLocaleDateString() : 'N/A'}
                          </TableCell>
                          <TableCell className="font-medium">{pred.symbol}</TableCell>
                          <TableCell>
                            <Badge className={`${
                              pred.signal === 'BUY' ? 'bg-success/10 text-success' :
                              pred.signal === 'SELL' ? 'bg-danger/10 text-danger' :
                              'bg-amber-500/10 text-amber-500'
                            } border-0`}>
                              {pred.signal}
                            </Badge>
                          </TableCell>
                          <TableCell className="font-data">₹{pred.entry_price?.toFixed(2) || 'N/A'}</TableCell>
                          <TableCell className="font-data">₹{pred.target_price?.toFixed(2) || 'N/A'}</TableCell>
                          <TableCell className="font-data">₹{pred.actual_price?.toFixed(2) || 'Pending'}</TableCell>
                          <TableCell className={`font-data ${
                            (pred.return_pct || 0) >= 0 ? 'text-success' : 'text-danger'
                          }`}>
                            {pred.return_pct ? `${pred.return_pct >= 0 ? '+' : ''}${pred.return_pct.toFixed(2)}%` : '-'}
                          </TableCell>
                          <TableCell>{getResultBadge(pred.result)}</TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="leaderboard">
            <Card className="card-surface">
              <CardHeader>
                <CardTitle className="text-lg font-heading flex items-center gap-2">
                  <Trophy className="w-5 h-5 text-yellow-500" />
                  Strategy Leaderboard
                </CardTitle>
                <CardDescription>Compare performance across different strategies</CardDescription>
              </CardHeader>
              <CardContent>
                {leaderboard.length === 0 ? (
                  <p className="text-text-secondary text-center py-8">No strategy data available</p>
                ) : (
                  <div className="space-y-3">
                    {leaderboard.map((strategy, idx) => (
                      <motion.div
                        key={idx}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: idx * 0.1 }}
                        className="p-4 rounded-lg bg-surface-highlight flex items-center justify-between"
                      >
                        <div className="flex items-center gap-4">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                            idx === 0 ? 'bg-yellow-500/20 text-yellow-500' :
                            idx === 1 ? 'bg-gray-400/20 text-gray-400' :
                            idx === 2 ? 'bg-amber-600/20 text-amber-600' :
                            'bg-surface text-text-secondary'
                          }`}>
                            {idx < 3 ? <Trophy className="w-5 h-5" /> : <span className="font-bold">#{idx + 1}</span>}
                          </div>
                          <div>
                            <p className="font-medium text-text-primary">{strategy.name}</p>
                            <p className="text-xs text-text-secondary">
                              {strategy.trades || 0} trades • Win rate: {(strategy.win_rate || 0).toFixed(1)}%
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className={`text-xl font-bold ${
                            (strategy.total_return || 0) >= 0 ? 'text-success' : 'text-danger'
                          }`}>
                            {(strategy.total_return || 0) >= 0 ? '+' : ''}{(strategy.total_return || 0).toFixed(2)}%
                          </p>
                          <p className="text-xs text-text-secondary">Total Return</p>
                        </div>
                      </motion.div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </motion.div>
    </div>
  );
}
