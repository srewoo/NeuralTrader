import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import axios from "axios";
import { toast } from "sonner";
import {
  History,
  TrendingUp,
  TrendingDown,
  Minus,
  Calendar,
  Brain,
  Trash2,
  Eye,
  Loader2,
  RefreshCw,
  Search
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { format } from "date-fns";
import { API_URL } from "@/config/api";

export default function AnalysisHistory() {
  const navigate = useNavigate();
  const [analyses, setAnalyses] = useState([]);
  const [filteredAnalyses, setFilteredAnalyses] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const [deleteId, setDeleteId] = useState(null);

  useEffect(() => {
    fetchAnalyses();
  }, []);

  useEffect(() => {
    if (!searchQuery) {
      setFilteredAnalyses(analyses);
    } else {
      const filtered = analyses.filter(analysis =>
        analysis.symbol?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        analysis.model_used?.toLowerCase().includes(searchQuery.toLowerCase())
      );
      setFilteredAnalyses(filtered);
    }
  }, [searchQuery, analyses]);

  const fetchAnalyses = async () => {
    setIsLoading(true);
    try {
      const response = await axios.get(`${API_URL}/analysis/history`);
      setAnalyses(response.data || []);
      setFilteredAnalyses(response.data || []);
    } catch (error) {
      console.error("Error fetching analyses:", error);
      toast.error("Failed to fetch analysis history");
    } finally {
      setIsLoading(false);
    }
  };

  const deleteAnalysis = async (id) => {
    try {
      await axios.delete(`${API_URL}/analysis/${id}`);
      setAnalyses(analyses.filter(a => a.id !== id));
      toast.success("Analysis deleted");
    } catch (error) {
      console.error("Error deleting analysis:", error);
      toast.error("Failed to delete analysis");
    } finally {
      setDeleteId(null);
    }
  };

  const getSignalIcon = (recommendation) => {
    switch (recommendation?.toUpperCase()) {
      case "BUY":
        return <TrendingUp className="w-5 h-5 text-success" />;
      case "SELL":
        return <TrendingDown className="w-5 h-5 text-danger" />;
      default:
        return <Minus className="w-5 h-5 text-text-secondary" />;
    }
  };

  const getSignalBadge = (recommendation) => {
    switch (recommendation?.toUpperCase()) {
      case "BUY":
        return <Badge className="bg-success-dim text-success border-0">BUY</Badge>;
      case "SELL":
        return <Badge className="bg-danger-dim text-danger border-0">SELL</Badge>;
      default:
        return <Badge className="bg-[#1F1F1F] text-text-secondary border-0">HOLD</Badge>;
    }
  };

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return "text-success";
    if (confidence >= 60) return "text-yellow-500";
    return "text-danger";
  };

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6" data-testid="analysis-history">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-lg bg-ai-accent/10 flex items-center justify-center">
              <History className="w-6 h-6 text-ai-accent" />
            </div>
            <div>
              <h1 className="text-3xl font-heading font-bold text-text-primary">
                Analysis History
              </h1>
              <p className="text-text-secondary">
                View and manage your past AI-generated stock analyses
              </p>
            </div>
          </div>
          <Button
            onClick={fetchAnalyses}
            variant="outline"
            className="border-[#1F1F1F] hover:bg-surface-highlight"
            disabled={isLoading}
          >
            <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Search Bar */}
        <div className="relative max-w-xl">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-text-secondary" />
          <Input
            placeholder="Search by stock symbol or model..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-12 bg-surface border-[#1F1F1F] text-text-primary"
            data-testid="search-input"
          />
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card className="card-surface">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Total Analyses</p>
                <p className="text-2xl font-data font-bold text-text-primary">
                  {analyses.length}
                </p>
              </div>
              <Brain className="w-8 h-8 text-ai-accent" />
            </div>
          </CardContent>
        </Card>

        <Card className="card-surface">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Buy Signals</p>
                <p className="text-2xl font-data font-bold text-success">
                  {analyses.filter(a => a.recommendation?.toUpperCase() === "BUY").length}
                </p>
              </div>
              <TrendingUp className="w-8 h-8 text-success" />
            </div>
          </CardContent>
        </Card>

        <Card className="card-surface">
          <CardContent className="pt-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs text-text-secondary mb-1">Sell Signals</p>
                <p className="text-2xl font-data font-bold text-danger">
                  {analyses.filter(a => a.recommendation?.toUpperCase() === "SELL").length}
                </p>
              </div>
              <TrendingDown className="w-8 h-8 text-danger" />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Analysis List */}
      <Card className="card-surface">
        <CardHeader>
          <CardTitle className="text-sm font-heading">
            All Analyses ({filteredAnalyses.length})
          </CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <span className="ml-3 text-text-secondary">Loading analyses...</span>
            </div>
          ) : filteredAnalyses.length === 0 ? (
            <div className="text-center py-12">
              <History className="w-12 h-12 mx-auto text-text-secondary/50 mb-4" />
              <p className="text-text-secondary">
                {searchQuery ? "No analyses found matching your search" : "No analysis history yet"}
              </p>
              {!searchQuery && (
                <Button
                  onClick={() => navigate("/")}
                  className="btn-primary mt-4"
                >
                  Run Your First Analysis
                </Button>
              )}
            </div>
          ) : (
            <ScrollArea className="h-[600px] pr-4">
              <div className="space-y-3">
                <AnimatePresence>
                  {filteredAnalyses.map((analysis, index) => (
                    <motion.div
                      key={analysis.id}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0, scale: 0.95 }}
                      transition={{ delay: index * 0.05 }}
                      className="p-4 rounded-lg bg-surface-highlight border border-[#1F1F1F] hover:border-primary/30 transition-all"
                      data-testid={`analysis-${analysis.id}`}
                    >
                      <div className="flex items-start justify-between gap-4">
                        {/* Left: Signal Icon & Info */}
                        <div className="flex items-start gap-4 flex-1">
                          <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
                            analysis.recommendation?.toUpperCase() === "BUY" 
                              ? "bg-success-dim" 
                              : analysis.recommendation?.toUpperCase() === "SELL"
                              ? "bg-danger-dim"
                              : "bg-[#1F1F1F]"
                          }`}>
                            {getSignalIcon(analysis.recommendation)}
                          </div>

                          <div className="flex-1">
                            <div className="flex items-center gap-3 mb-2">
                              <h3 className="text-lg font-heading font-bold text-text-primary">
                                {analysis.symbol}
                              </h3>
                              {getSignalBadge(analysis.recommendation)}
                              <Badge variant="outline" className="text-xs">
                                {analysis.model_used || "Unknown Model"}
                              </Badge>
                            </div>

                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-3">
                              <div>
                                <p className="text-xs text-text-secondary">Confidence</p>
                                <p className={`font-data text-sm font-medium ${getConfidenceColor(analysis.confidence)}`}>
                                  {analysis.confidence}%
                                </p>
                              </div>
                              <div>
                                <p className="text-xs text-text-secondary">Entry Price</p>
                                <p className="font-data text-sm text-text-primary">
                                  ₹{analysis.entry_price?.toFixed(2) || "N/A"}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs text-text-secondary">Target</p>
                                <p className="font-data text-sm text-success">
                                  ₹{analysis.target_price?.toFixed(2) || "N/A"}
                                </p>
                              </div>
                              <div>
                                <p className="text-xs text-text-secondary">Stop Loss</p>
                                <p className="font-data text-sm text-danger">
                                  ₹{analysis.stop_loss?.toFixed(2) || "N/A"}
                                </p>
                              </div>
                            </div>

                            <div className="flex items-center gap-2 text-xs text-text-secondary">
                              <Calendar className="w-3.5 h-3.5" />
                              <span>
                                {analysis.created_at 
                                  ? format(new Date(analysis.created_at), "MMM dd, yyyy 'at' hh:mm a")
                                  : "Date unknown"}
                              </span>
                            </div>
                          </div>
                        </div>

                        {/* Right: Actions */}
                        <div className="flex items-center gap-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => navigate(`/analysis/${analysis.id}`)}
                            className="hover:bg-primary/10 hover:text-primary"
                          >
                            <Eye className="w-4 h-4" />
                          </Button>
                          
                          <AlertDialog>
                            <AlertDialogTrigger asChild>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => setDeleteId(analysis.id)}
                                className="hover:bg-danger/10 hover:text-danger"
                              >
                                <Trash2 className="w-4 h-4" />
                              </Button>
                            </AlertDialogTrigger>
                            <AlertDialogContent className="bg-surface border-[#1F1F1F]">
                              <AlertDialogHeader>
                                <AlertDialogTitle className="text-text-primary">
                                  Delete Analysis?
                                </AlertDialogTitle>
                                <AlertDialogDescription className="text-text-secondary">
                                  This action cannot be undone. This will permanently delete the
                                  analysis for {analysis.symbol}.
                                </AlertDialogDescription>
                              </AlertDialogHeader>
                              <AlertDialogFooter>
                                <AlertDialogCancel className="bg-surface-highlight border-[#1F1F1F] text-text-primary hover:bg-[#1F1F1F]">
                                  Cancel
                                </AlertDialogCancel>
                                <AlertDialogAction
                                  onClick={() => deleteAnalysis(analysis.id)}
                                  className="bg-danger hover:bg-danger/90 text-white"
                                >
                                  Delete
                                </AlertDialogAction>
                              </AlertDialogFooter>
                            </AlertDialogContent>
                          </AlertDialog>
                        </div>
                      </div>
                    </motion.div>
                  ))}
                </AnimatePresence>
              </div>
            </ScrollArea>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

