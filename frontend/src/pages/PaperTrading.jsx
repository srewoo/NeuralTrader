import { useState, useEffect, useRef } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Wallet,
  TrendingUp,
  TrendingDown,
  DollarSign,
  Activity,
  History,
  RotateCcw,
  ArrowUpCircle,
  ArrowDownCircle,
  Clock,
  CheckCircle,
  XCircle,
  Loader2,
  Search
} from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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

export default function PaperTrading() {
  const [portfolio, setPortfolio] = useState(null);
  const [positions, setPositions] = useState([]);
  const [trades, setTrades] = useState([]);
  const [orders, setOrders] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isPlacingOrder, setIsPlacingOrder] = useState(false);

  // Order form state
  const [orderForm, setOrderForm] = useState({
    symbol: "",
    side: "BUY",
    quantity: "",
    orderType: "MARKET",
    limitPrice: "",
    currentPrice: ""
  });

  // Stock search autocomplete state
  const [stockSuggestions, setStockSuggestions] = useState([]);
  const [showSuggestions, setShowSuggestions] = useState(false);
  const [isSearching, setIsSearching] = useState(false);
  const searchRef = useRef(null);

  useEffect(() => {
    fetchPortfolio();
    fetchTrades();
    fetchOrders();
  }, []);

  // Handle click outside to close suggestions
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowSuggestions(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Search stocks as user types
  const searchStocks = async (query) => {
    if (!query || query.length < 1) {
      setStockSuggestions([]);
      setShowSuggestions(false);
      return;
    }

    setIsSearching(true);
    try {
      const response = await axios.get(`${API_URL}/stocks/search?q=${query}`);
      setStockSuggestions(response.data.slice(0, 10)); // Limit to 10 suggestions
      setShowSuggestions(true);
    } catch (error) {
      console.error("Error searching stocks:", error);
      setStockSuggestions([]);
    } finally {
      setIsSearching(false);
    }
  };

  // Debounced search
  useEffect(() => {
    const timer = setTimeout(() => {
      if (orderForm.symbol) {
        searchStocks(orderForm.symbol);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [orderForm.symbol]);

  // Handle stock selection from dropdown
  const handleSelectStock = (stock) => {
    setOrderForm({ ...orderForm, symbol: stock.symbol });
    setShowSuggestions(false);
    setStockSuggestions([]);
  };

  const fetchPortfolio = async () => {
    try {
      const response = await axios.get(`${API_URL}/paper-trading/portfolio`);
      setPortfolio(response.data.summary);
      setPositions(response.data.positions);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching portfolio:", error);
      toast.error("Failed to load portfolio");
      setIsLoading(false);
    }
  };

  const fetchTrades = async () => {
    try {
      const response = await axios.get(`${API_URL}/paper-trading/trades?limit=20`);
      setTrades(response.data.trades);
    } catch (error) {
      console.error("Error fetching trades:", error);
    }
  };

  const fetchOrders = async () => {
    try {
      const response = await axios.get(`${API_URL}/paper-trading/orders?limit=20`);
      setOrders(response.data.orders);
    } catch (error) {
      console.error("Error fetching orders:", error);
    }
  };

  const handlePlaceOrder = async (e) => {
    e.preventDefault();
    setIsPlacingOrder(true);

    try {
      // First get current price if not provided
      let currentPrice = parseFloat(orderForm.currentPrice);
      if (!currentPrice || currentPrice <= 0) {
        const quoteResponse = await axios.get(`${API_URL}/stocks/quote/${orderForm.symbol}`);
        currentPrice = quoteResponse.data.current_price;
      }

      const orderData = {
        symbol: orderForm.symbol.toUpperCase(),
        side: orderForm.side,
        quantity: parseInt(orderForm.quantity),
        current_price: currentPrice,
        order_type: orderForm.orderType,
        limit_price: orderForm.limitPrice ? parseFloat(orderForm.limitPrice) : null
      };

      await axios.post(`${API_URL}/paper-trading/order`, orderData);

      toast.success(`${orderForm.side} order placed for ${orderForm.symbol}!`);

      // Reset form
      setOrderForm({
        symbol: "",
        side: "BUY",
        quantity: "",
        orderType: "MARKET",
        limitPrice: "",
        currentPrice: ""
      });

      // Refresh data
      fetchPortfolio();
      fetchTrades();
      fetchOrders();
    } catch (error) {
      console.error("Error placing order:", error);
      // Handle FastAPI validation errors (422) which return array of objects
      const errorDetail = error.response?.data?.detail;
      let errorMessage = "Failed to place order";

      if (typeof errorDetail === 'string') {
        errorMessage = errorDetail;
      } else if (Array.isArray(errorDetail)) {
        // FastAPI validation error format
        errorMessage = errorDetail.map(e => e.msg || e.message).join(', ');
      } else if (errorDetail?.msg) {
        errorMessage = errorDetail.msg;
      }

      toast.error(errorMessage);
    } finally {
      setIsPlacingOrder(false);
    }
  };

  const handleResetAccount = async () => {
    if (!confirm("Are you sure you want to reset your paper trading account? This will clear all positions and reset to initial capital.")) {
      return;
    }

    try {
      await axios.post(`${API_URL}/paper-trading/reset`);
      toast.success("Paper trading account reset successfully!");
      fetchPortfolio();
      fetchTrades();
      fetchOrders();
    } catch (error) {
      console.error("Error resetting account:", error);
      toast.error("Failed to reset account");
    }
  };

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatDateTime = (timestamp) => {
    return new Date(timestamp).toLocaleString('en-IN', {
      dateStyle: 'short',
      timeStyle: 'short'
    });
  };

  const getStatusBadge = (status) => {
    const variants = {
      FILLED: "default",
      PENDING: "secondary",
      CANCELLED: "destructive",
      REJECTED: "destructive",
      PARTIALLY_FILLED: "secondary"
    };

    return (
      <Badge variant={variants[status] || "secondary"}>
        {status}
      </Badge>
    );
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
              <Wallet className="w-6 h-6 text-primary" />
            </div>
            <div>
              <h1 className="text-2xl font-heading font-bold text-text-primary">Paper Trading</h1>
              <p className="text-text-secondary">Practice trading without real money</p>
            </div>
          </div>
          <Button
            onClick={handleResetAccount}
            variant="outline"
            className="flex items-center gap-2"
          >
            <RotateCcw className="w-4 h-4" />
            Reset Account
          </Button>
        </div>

        {/* Portfolio Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Cash Available</p>
                  <p className="text-2xl font-bold text-text-primary">
                    {formatCurrency(portfolio?.cash || 0)}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-success/10 flex items-center justify-center">
                  <DollarSign className="w-6 h-6 text-success" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Positions Value</p>
                  <p className="text-2xl font-bold text-text-primary">
                    {formatCurrency(portfolio?.positions_value || 0)}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-primary/10 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Total P&L</p>
                  <p className={`text-2xl font-bold ${
                    (portfolio?.total_pnl || 0) >= 0 ? 'text-success' : 'text-danger'
                  }`}>
                    {formatCurrency(portfolio?.total_pnl || 0)}
                  </p>
                  <p className={`text-xs ${
                    (portfolio?.return_pct || 0) >= 0 ? 'text-success' : 'text-danger'
                  }`}>
                    {portfolio?.return_pct?.toFixed(2)}%
                  </p>
                </div>
                <div className={`w-12 h-12 rounded-full flex items-center justify-center ${
                  (portfolio?.total_pnl || 0) >= 0 ? 'bg-success/10' : 'bg-danger/10'
                }`}>
                  {(portfolio?.total_pnl || 0) >= 0 ? (
                    <TrendingUp className="w-6 h-6 text-success" />
                  ) : (
                    <TrendingDown className="w-6 h-6 text-danger" />
                  )}
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="card-surface">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-text-secondary">Total Value</p>
                  <p className="text-2xl font-bold text-text-primary">
                    {formatCurrency(portfolio?.total_value || 0)}
                  </p>
                </div>
                <div className="w-12 h-12 rounded-full bg-ai-accent/10 flex items-center justify-center">
                  <Wallet className="w-6 h-6 text-ai-accent" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Place Order Form */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <ArrowUpCircle className="w-5 h-5 text-success" />
              Place Order
            </CardTitle>
            <CardDescription>
              Simulate buying or selling stocks with realistic slippage (0.1%) and commission (0.03%)
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handlePlaceOrder} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 relative">
                <div className="space-y-2 relative z-20" ref={searchRef}>
                  <Label htmlFor="symbol">Symbol</Label>
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
                    <Input
                      id="symbol"
                      type="text"
                      placeholder="Search stocks..."
                      value={orderForm.symbol}
                      onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value.toUpperCase() })}
                      onFocus={() => orderForm.symbol && stockSuggestions.length > 0 && setShowSuggestions(true)}
                      required
                      className="bg-surface-highlight border-[#1F1F1F] pl-10"
                      autoComplete="off"
                    />
                    {isSearching && (
                      <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-text-secondary" />
                    )}
                  </div>

                  {/* Stock Suggestions Dropdown */}
                  {showSuggestions && stockSuggestions.length > 0 && (
                    <div className="absolute top-full left-0 z-[9999] w-full mt-1 bg-surface-card border border-[#1F1F1F] rounded-lg shadow-2xl max-h-64 overflow-y-auto" style={{ zIndex: 9999 }}>
                      {stockSuggestions.map((stock, index) => (
                        <button
                          key={stock.symbol || index}
                          type="button"
                          onClick={() => handleSelectStock(stock)}
                          className="w-full px-4 py-3 text-left hover:bg-surface-highlight transition-colors border-b border-[#1F1F1F] last:border-0"
                        >
                          <div className="flex items-center justify-between">
                            <div>
                              <span className="font-medium text-text-primary">{stock.symbol}</span>
                              <p className="text-xs text-text-secondary truncate">{stock.name}</p>
                            </div>
                            <Badge variant="outline" className="text-xs">
                              {stock.sector || 'NSE'}
                            </Badge>
                          </div>
                        </button>
                      ))}
                    </div>
                  )}
                </div>

                <div className="space-y-2">
                  <Label htmlFor="side">Side</Label>
                  <Select
                    value={orderForm.side}
                    onValueChange={(value) => setOrderForm({ ...orderForm, side: value })}
                  >
                    <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="BUY">
                        <div className="flex items-center gap-2">
                          <ArrowUpCircle className="w-4 h-4 text-success" />
                          Buy
                        </div>
                      </SelectItem>
                      <SelectItem value="SELL">
                        <div className="flex items-center gap-2">
                          <ArrowDownCircle className="w-4 h-4 text-danger" />
                          Sell
                        </div>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="quantity">Quantity</Label>
                  <Input
                    id="quantity"
                    type="number"
                    min="1"
                    placeholder="10"
                    value={orderForm.quantity}
                    onChange={(e) => setOrderForm({ ...orderForm, quantity: e.target.value })}
                    required
                    className="bg-surface-highlight border-[#1F1F1F]"
                  />
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="orderType">Order Type</Label>
                  <Select
                    value={orderForm.orderType}
                    onValueChange={(value) => setOrderForm({ ...orderForm, orderType: value })}
                  >
                    <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="MARKET">Market Order</SelectItem>
                      <SelectItem value="LIMIT">Limit Order</SelectItem>
                      <SelectItem value="STOP_LOSS">Stop Loss</SelectItem>
                    </SelectContent>
                  </Select>
                </div>

                {orderForm.orderType === "LIMIT" && (
                  <div className="space-y-2">
                    <Label htmlFor="limitPrice">Limit Price</Label>
                    <Input
                      id="limitPrice"
                      type="number"
                      step="0.01"
                      placeholder="2500.00"
                      value={orderForm.limitPrice}
                      onChange={(e) => setOrderForm({ ...orderForm, limitPrice: e.target.value })}
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                  </div>
                )}

                <div className="space-y-2">
                  <Label htmlFor="currentPrice">Current Price (optional)</Label>
                  <Input
                    id="currentPrice"
                    type="number"
                    step="0.01"
                    placeholder="Auto-fetch if empty"
                    value={orderForm.currentPrice}
                    onChange={(e) => setOrderForm({ ...orderForm, currentPrice: e.target.value })}
                    className="bg-surface-highlight border-[#1F1F1F]"
                  />
                </div>
              </div>

              <Button
                type="submit"
                disabled={isPlacingOrder}
                className="btn-primary w-full md:w-auto"
              >
                {isPlacingOrder ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Placing Order...
                  </>
                ) : (
                  <>
                    {orderForm.side === "BUY" ? (
                      <ArrowUpCircle className="w-4 h-4 mr-2" />
                    ) : (
                      <ArrowDownCircle className="w-4 h-4 mr-2" />
                    )}
                    Place {orderForm.side} Order
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Current Positions */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary" />
              Current Positions
            </CardTitle>
          </CardHeader>
          <CardContent>
            {positions.length === 0 ? (
              <p className="text-text-secondary text-center py-8">No open positions</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>Avg Price</TableHead>
                    <TableHead>Current Price</TableHead>
                    <TableHead>P&L</TableHead>
                    <TableHead>Return %</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {positions.map((position, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="font-medium">{position.symbol}</TableCell>
                      <TableCell>{position.quantity}</TableCell>
                      <TableCell>{formatCurrency(position.average_price)}</TableCell>
                      <TableCell>{formatCurrency(position.current_price)}</TableCell>
                      <TableCell className={position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger'}>
                        {formatCurrency(position.unrealized_pnl)}
                      </TableCell>
                      <TableCell className={position.return_pct >= 0 ? 'text-success' : 'text-danger'}>
                        {position.return_pct?.toFixed(2)}%
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* Recent Trades */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <History className="w-5 h-5 text-ai-accent" />
              Recent Trades
            </CardTitle>
          </CardHeader>
          <CardContent>
            {trades.length === 0 ? (
              <p className="text-text-secondary text-center py-8">No trades yet</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Commission</TableHead>
                    <TableHead>Total</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {trades.map((trade, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="text-xs">{formatDateTime(trade.timestamp)}</TableCell>
                      <TableCell className="font-medium">{trade.symbol}</TableCell>
                      <TableCell>
                        <Badge variant={trade.side === "BUY" ? "default" : "secondary"}>
                          {trade.side}
                        </Badge>
                      </TableCell>
                      <TableCell>{trade.quantity}</TableCell>
                      <TableCell>{formatCurrency(trade.execution_price)}</TableCell>
                      <TableCell className="text-danger">
                        {formatCurrency(trade.commission)}
                      </TableCell>
                      <TableCell className="font-medium">
                        {formatCurrency(trade.total_value)}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        {/* Order History */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Clock className="w-5 h-5 text-warning" />
              Order History
            </CardTitle>
          </CardHeader>
          <CardContent>
            {orders.length === 0 ? (
              <p className="text-text-secondary text-center py-8">No orders yet</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Time</TableHead>
                    <TableHead>Symbol</TableHead>
                    <TableHead>Side</TableHead>
                    <TableHead>Type</TableHead>
                    <TableHead>Quantity</TableHead>
                    <TableHead>Price</TableHead>
                    <TableHead>Status</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {orders.map((order, idx) => (
                    <TableRow key={idx}>
                      <TableCell className="text-xs">{formatDateTime(order.timestamp)}</TableCell>
                      <TableCell className="font-medium">{order.symbol}</TableCell>
                      <TableCell>
                        <Badge variant={order.side === "BUY" ? "default" : "secondary"}>
                          {order.side}
                        </Badge>
                      </TableCell>
                      <TableCell>{order.order_type}</TableCell>
                      <TableCell>
                        {order.filled_quantity}/{order.quantity}
                      </TableCell>
                      <TableCell>{formatCurrency(order.price || 0)}</TableCell>
                      <TableCell>{getStatusBadge(order.status)}</TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
