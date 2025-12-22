import { useState } from "react";
import { useNavigate } from "react-router-dom";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Filter,
  Search,
  TrendingUp,
  DollarSign,
  Percent,
  BarChart3,
  Loader2,
  ArrowUpCircle,
  ArrowDownCircle,
  Star,
  Sparkles
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { API_URL } from "@/config/api";

export default function Screener() {
  const navigate = useNavigate();
  const [isScreening, setIsScreening] = useState(false);
  const [results, setResults] = useState([]);

  // Screening criteria
  const [criteria, setCriteria] = useState({
    peMax: "",
    roeMin: "",
    debtToEquityMax: "",
    marketCapMin: "",
    dividendYieldMin: "",
    profitMarginMin: "",
    limit: 50
  });

  const handleScreen = async (e) => {
    e.preventDefault();
    setIsScreening(true);

    try {
      const params = {};
      if (criteria.peMax) params.pe_max = parseFloat(criteria.peMax);
      if (criteria.roeMin) params.roe_min = parseFloat(criteria.roeMin);
      if (criteria.debtToEquityMax) params.debt_to_equity_max = parseFloat(criteria.debtToEquityMax);
      if (criteria.marketCapMin) params.market_cap_min = parseFloat(criteria.marketCapMin);
      if (criteria.dividendYieldMin) params.dividend_yield_min = parseFloat(criteria.dividendYieldMin);
      if (criteria.profitMarginMin) params.profit_margin_min = parseFloat(criteria.profitMarginMin);
      params.limit = criteria.limit;

      const response = await axios.post(`${API_URL}/screener/screen`, params);
      setResults(response.data.stocks);

      if (response.data.stocks.length === 0) {
        toast.info("No stocks match your criteria. Try relaxing your filters.");
      } else {
        toast.success(`Found ${response.data.stocks.length} stocks matching criteria!`);
      }
    } catch (error) {
      console.error("Error screening stocks:", error);
      const detail = error.response?.data?.detail;
      const errorMessage = Array.isArray(detail)
        ? detail.map(e => e.msg).join(', ')
        : (typeof detail === 'string' ? detail : "Failed to screen stocks");
      toast.error(errorMessage);
    } finally {
      setIsScreening(false);
    }
  };

  const handleReset = () => {
    setCriteria({
      peMax: "",
      roeMin: "",
      debtToEquityMax: "",
      marketCapMin: "",
      dividendYieldMin: "",
      profitMarginMin: "",
      limit: 50
    });
    setResults([]);
  };

  const handleLoadPreset = async (preset) => {
    const presets = {
      value: {
        peMax: "15",
        roeMin: "20",
        debtToEquityMax: "0.5",
        marketCapMin: "1000",
        dividendYieldMin: "",
        profitMarginMin: "10",
        limit: 50
      },
      growth: {
        peMax: "",
        roeMin: "25",
        debtToEquityMax: "",
        marketCapMin: "500",
        dividendYieldMin: "",
        profitMarginMin: "15",
        limit: 50
      },
      dividend: {
        peMax: "",
        roeMin: "15",
        debtToEquityMax: "0.8",
        marketCapMin: "2000",
        dividendYieldMin: "3",
        profitMarginMin: "",
        limit: 50
      },
      quality: {
        peMax: "",
        roeMin: "25",
        debtToEquityMax: "0.3",
        marketCapMin: "1000",
        dividendYieldMin: "",
        profitMarginMin: "20",
        limit: 50
      }
    };

    if (presets[preset]) {
      const presetCriteria = presets[preset];
      setCriteria(presetCriteria);
      toast.success(`Loaded ${preset.charAt(0).toUpperCase() + preset.slice(1)} preset - Screening...`);

      // Automatically run the screen with the preset criteria
      setIsScreening(true);
      try {
        const params = {};
        if (presetCriteria.peMax) params.pe_max = parseFloat(presetCriteria.peMax);
        if (presetCriteria.roeMin) params.roe_min = parseFloat(presetCriteria.roeMin);
        if (presetCriteria.debtToEquityMax) params.debt_to_equity_max = parseFloat(presetCriteria.debtToEquityMax);
        if (presetCriteria.marketCapMin) params.market_cap_min = parseFloat(presetCriteria.marketCapMin);
        if (presetCriteria.dividendYieldMin) params.dividend_yield_min = parseFloat(presetCriteria.dividendYieldMin);
        if (presetCriteria.profitMarginMin) params.profit_margin_min = parseFloat(presetCriteria.profitMarginMin);
        params.limit = presetCriteria.limit;

        const response = await axios.post(`${API_URL}/screener/screen`, params);
        setResults(response.data.stocks);

        if (response.data.stocks.length === 0) {
          toast.info("No stocks match your criteria. Try relaxing your filters.");
        } else {
          toast.success(`Found ${response.data.stocks.length} stocks matching criteria!`);
        }
      } catch (error) {
        console.error("Error screening stocks:", error);
        const detail = error.response?.data?.detail;
        const errorMessage = Array.isArray(detail)
          ? detail.map(e => e.msg).join(', ')
          : (typeof detail === 'string' ? detail : "Failed to screen stocks");
        toast.error(errorMessage);
      } finally {
        setIsScreening(false);
      }
    }
  };

  const formatCurrency = (value) => {
    if (!value) return "N/A";
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2,
      notation: 'compact'
    }).format(value);
  };

  const formatNumber = (value, decimals = 2) => {
    if (value === null || value === undefined) return "N/A";
    return value.toFixed(decimals);
  };

  const handleViewStock = (symbol) => {
    navigate(`/stock/${symbol}`);
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Header */}
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-ai-accent/10 flex items-center justify-center">
            <Filter className="w-6 h-6 text-ai-accent" />
          </div>
          <div>
            <h1 className="text-2xl font-heading font-bold text-text-primary">Stock Screener</h1>
            <p className="text-text-secondary">Find stocks by fundamental criteria using FREE TradingView data</p>
          </div>
        </div>

        {/* Info Card */}
        <Card className="card-surface border-ai-accent/20">
          <CardContent className="py-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-ai-accent/10 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-ai-accent" />
              </div>
              <div>
                <p className="text-sm font-medium text-text-primary">Powered by TradingView Screener</p>
                <p className="text-xs text-text-secondary mt-1">
                  Real-time fundamental data for NSE/BSE stocks with NO API KEY required!
                  Data includes P/E, ROE, Debt/Equity, Market Cap, and more.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Preset Buttons */}
        <div className="flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleLoadPreset('value')}
            className="flex items-center gap-2"
          >
            <DollarSign className="w-4 h-4" />
            Value Stocks
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleLoadPreset('growth')}
            className="flex items-center gap-2"
          >
            <TrendingUp className="w-4 h-4" />
            Growth Stocks
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleLoadPreset('dividend')}
            className="flex items-center gap-2"
          >
            <Percent className="w-4 h-4" />
            Dividend Stocks
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => handleLoadPreset('quality')}
            className="flex items-center gap-2"
          >
            <Star className="w-4 h-4" />
            Quality Stocks
          </Button>
        </div>

        {/* Screening Form */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Filter className="w-5 h-5 text-primary" />
              Screening Criteria
            </CardTitle>
            <CardDescription>
              Set your criteria to filter stocks. Leave fields empty to skip that filter.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleScreen} className="space-y-6">
              {/* Valuation Metrics */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-text-primary">Valuation Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="pe-max">P/E Ratio (Max)</Label>
                    <Input
                      id="pe-max"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 15"
                      value={criteria.peMax}
                      onChange={(e) => setCriteria({ ...criteria, peMax: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Lower P/E = potentially undervalued</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="market-cap-min">Market Cap (Min, Crores)</Label>
                    <Input
                      id="market-cap-min"
                      type="number"
                      step="100"
                      placeholder="e.g., 1000"
                      value={criteria.marketCapMin}
                      onChange={(e) => setCriteria({ ...criteria, marketCapMin: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Larger caps = more stability</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="dividend-yield-min">Dividend Yield (Min %)</Label>
                    <Input
                      id="dividend-yield-min"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 2.5"
                      value={criteria.dividendYieldMin}
                      onChange={(e) => setCriteria({ ...criteria, dividendYieldMin: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Higher yield = more income</p>
                  </div>
                </div>
              </div>

              {/* Profitability Metrics */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-text-primary">Profitability Metrics</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="roe-min">ROE - Return on Equity (Min %)</Label>
                    <Input
                      id="roe-min"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 20"
                      value={criteria.roeMin}
                      onChange={(e) => setCriteria({ ...criteria, roeMin: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Higher ROE = better profitability</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="profit-margin-min">Profit Margin (Min %)</Label>
                    <Input
                      id="profit-margin-min"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 10"
                      value={criteria.profitMarginMin}
                      onChange={(e) => setCriteria({ ...criteria, profitMarginMin: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Higher margin = better efficiency</p>
                  </div>
                </div>
              </div>

              {/* Financial Health */}
              <div className="space-y-4">
                <h3 className="text-sm font-medium text-text-primary">Financial Health</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="debt-equity-max">Debt to Equity (Max)</Label>
                    <Input
                      id="debt-equity-max"
                      type="number"
                      step="0.1"
                      placeholder="e.g., 0.5"
                      value={criteria.debtToEquityMax}
                      onChange={(e) => setCriteria({ ...criteria, debtToEquityMax: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Lower D/E = less leverage risk</p>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="limit">Max Results</Label>
                    <Input
                      id="limit"
                      type="number"
                      min="1"
                      max="200"
                      value={criteria.limit}
                      onChange={(e) => setCriteria({ ...criteria, limit: parseInt(e.target.value) })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Number of stocks to return</p>
                  </div>
                </div>
              </div>

              {/* Action Buttons */}
              <div className="flex gap-2">
                <Button
                  type="submit"
                  disabled={isScreening}
                  className="btn-primary"
                >
                  {isScreening ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Screening...
                    </>
                  ) : (
                    <>
                      <Search className="w-4 h-4 mr-2" />
                      Screen Stocks
                    </>
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={handleReset}
                >
                  Reset
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>

        {/* Results */}
        {results.length > 0 && (
          <Card className="card-surface">
            <CardHeader>
              <CardTitle className="text-lg font-heading flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-success" />
                Screening Results ({results.length} stocks)
              </CardTitle>
              <CardDescription>
                Click on a stock to view detailed analysis
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Name</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Market Cap</TableHead>
                    <TableHead>P/E Ratio</TableHead>
                    <TableHead>ROE %</TableHead>
                    <TableHead>D/E</TableHead>
                    <TableHead>Div Yield %</TableHead>
                    <TableHead>Profit Margin %</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {results.map((stock, idx) => (
                    <TableRow
                      key={idx}
                      className="cursor-pointer hover:bg-surface-highlight/50"
                      onClick={() => handleViewStock(stock.symbol)}
                    >
                      <TableCell className="font-medium text-primary">
                        {stock.symbol}
                      </TableCell>
                      <TableCell className="max-w-xs truncate">
                        {stock.name || "N/A"}
                      </TableCell>
                      <TableCell>{formatCurrency(stock.price)}</TableCell>
                      <TableCell>{formatCurrency(stock.market_cap)}</TableCell>
                      <TableCell>
                        {stock.pe_ratio ? (
                          <Badge variant={stock.pe_ratio < 15 ? "default" : "secondary"}>
                            {formatNumber(stock.pe_ratio)}
                          </Badge>
                        ) : (
                          "N/A"
                        )}
                      </TableCell>
                      <TableCell>
                        {stock.roe ? (
                          <span className={stock.roe >= 20 ? "text-success" : "text-text-primary"}>
                            {formatNumber(stock.roe)}%
                          </span>
                        ) : (
                          "N/A"
                        )}
                      </TableCell>
                      <TableCell>
                        {stock.debt_to_equity ? (
                          <Badge variant={stock.debt_to_equity < 0.5 ? "default" : "secondary"}>
                            {formatNumber(stock.debt_to_equity)}
                          </Badge>
                        ) : (
                          "N/A"
                        )}
                      </TableCell>
                      <TableCell>
                        {stock.dividend_yield ? (
                          <span className="text-success">
                            {formatNumber(stock.dividend_yield)}%
                          </span>
                        ) : (
                          "N/A"
                        )}
                      </TableCell>
                      <TableCell>
                        {stock.profit_margin ? (
                          <span className={stock.profit_margin >= 15 ? "text-success" : "text-text-primary"}>
                            {formatNumber(stock.profit_margin)}%
                          </span>
                        ) : (
                          "N/A"
                        )}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>
        )}

        {/* Example Queries */}
        {results.length === 0 && !isScreening && (
          <Card className="card-surface border-primary/20">
            <CardHeader>
              <CardTitle className="text-lg font-heading">Example Screening Strategies</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
                  <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                    <DollarSign className="w-4 h-4 text-success" />
                    Value Investing
                  </h4>
                  <ul className="text-sm text-text-secondary space-y-1">
                    <li>• P/E &lt; 15 (undervalued)</li>
                    <li>• ROE &gt; 20% (profitable)</li>
                    <li>• Debt/Equity &lt; 0.5 (low debt)</li>
                    <li>• Market Cap &gt; 1000 Cr (stable)</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
                  <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-primary" />
                    Growth Stocks
                  </h4>
                  <ul className="text-sm text-text-secondary space-y-1">
                    <li>• ROE &gt; 25% (high returns)</li>
                    <li>• Profit Margin &gt; 15% (efficient)</li>
                    <li>• Market Cap &gt; 500 Cr</li>
                    <li>• Focus on expansion potential</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
                  <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                    <Percent className="w-4 h-4 text-warning" />
                    Dividend Income
                  </h4>
                  <ul className="text-sm text-text-secondary space-y-1">
                    <li>• Dividend Yield &gt; 3% (income)</li>
                    <li>• ROE &gt; 15% (sustainable)</li>
                    <li>• Debt/Equity &lt; 0.8 (safe)</li>
                    <li>• Market Cap &gt; 2000 Cr (mature)</li>
                  </ul>
                </div>

                <div className="p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
                  <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                    <Star className="w-4 h-4 text-ai-accent" />
                    Quality Stocks
                  </h4>
                  <ul className="text-sm text-text-secondary space-y-1">
                    <li>• ROE &gt; 25% (excellent returns)</li>
                    <li>• Debt/Equity &lt; 0.3 (very safe)</li>
                    <li>• Profit Margin &gt; 20% (top efficiency)</li>
                    <li>• Market Cap &gt; 1000 Cr</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>
        )}
      </motion.div>
    </div>
  );
}
