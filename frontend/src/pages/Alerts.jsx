import { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Bell,
  Plus,
  Trash2,
  TrendingUp,
  Loader2,
  Mail,
  MessageSquare,
  CheckCircle,
  XCircle,
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
import { Checkbox } from "@/components/ui/checkbox";
import { API_URL } from "@/config/api";

const PRICE_CONDITIONS = [
  { value: "ABOVE", label: "Above" },
  { value: "BELOW", label: "Below" },
  { value: "CROSSES_ABOVE", label: "Crosses Above" },
  { value: "CROSSES_BELOW", label: "Crosses Below" },
  { value: "PERCENT_CHANGE_ABOVE", label: "% Change Above" },
  { value: "PERCENT_CHANGE_BELOW", label: "% Change Below" }
];

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);

  // Price alert form
  const [formData, setFormData] = useState({
    symbol: "",
    condition: "CROSSES_ABOVE",
    targetPrice: "",
    percentChange: "",
    telegram: true,
    email: false
  });

  // Stock search autocomplete state
  const [stockSearchResults, setStockSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const searchRef = useRef(null);
  const searchTimeoutRef = useRef(null);

  // Debounced stock search
  const searchStocks = useCallback(async (query) => {
    if (!query || query.length < 1) {
      setStockSearchResults([]);
      return;
    }

    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    searchTimeoutRef.current = setTimeout(async () => {
      setIsSearching(true);
      try {
        const response = await axios.get(`${API_URL}/stocks/search?q=${query}`);
        setStockSearchResults(response.data);
        setShowDropdown(true);
      } catch (error) {
        console.error("Stock search failed:", error);
        setStockSearchResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 300);
  }, []);

  // Handle click outside to close dropdown
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (searchRef.current && !searchRef.current.contains(event.target)) {
        setShowDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  useEffect(() => {
    fetchAlerts();
  }, []);

  const fetchAlerts = async () => {
    try {
      const response = await axios.get(`${API_URL}/alerts?user_id=default`);
      setAlerts(response.data.alerts);
      setIsLoading(false);
    } catch (error) {
      console.error("Error fetching alerts:", error);
      toast.error("Failed to load alerts");
      setIsLoading(false);
    }
  };

  const getDeliveryChannels = () => {
    const channels = [];
    if (formData.telegram) channels.push("TELEGRAM");
    if (formData.email) channels.push("EMAIL");
    return channels;
  };

  const handleCreateAlert = async (e) => {
    e.preventDefault();
    setIsCreating(true);

    try {
      const isPercentCondition = formData.condition.includes("PERCENT");

      const alertData = {
        user_id: "default",
        symbol: formData.symbol.toUpperCase(),
        condition: formData.condition,
        target_price: isPercentCondition ? 0 : parseFloat(formData.targetPrice),
        delivery_channels: getDeliveryChannels(),
        percent_change: formData.percentChange ? parseFloat(formData.percentChange) : null
      };

      await axios.post(`${API_URL}/alerts/price`, alertData);
      toast.success("Price alert created successfully!");

      setFormData({
        symbol: "",
        condition: "CROSSES_ABOVE",
        targetPrice: "",
        percentChange: "",
        telegram: true,
        email: false
      });

      fetchAlerts();
    } catch (error) {
      console.error("Error creating price alert:", error);
      toast.error(error.response?.data?.detail || "Failed to create alert");
    } finally {
      setIsCreating(false);
    }
  };

  const handleDeleteAlert = async (alertId) => {
    if (!confirm("Are you sure you want to delete this alert?")) {
      return;
    }

    try {
      await axios.delete(`${API_URL}/alerts/${alertId}`);
      toast.success("Alert deleted successfully!");
      fetchAlerts();
    } catch (error) {
      console.error("Error deleting alert:", error);
      toast.error("Failed to delete alert");
    }
  };

  const formatDateTime = (timestamp) => {
    return new Date(timestamp).toLocaleString('en-IN', {
      dateStyle: 'short',
      timeStyle: 'short'
    });
  };

  const getStatusBadge = (status) => {
    const icons = {
      ACTIVE: <CheckCircle className="w-3 h-3" />,
      TRIGGERED: <Bell className="w-3 h-3" />,
      CANCELLED: <XCircle className="w-3 h-3" />
    };

    const variants = {
      ACTIVE: "default",
      TRIGGERED: "secondary",
      CANCELLED: "destructive"
    };

    return (
      <Badge variant={variants[status] || "secondary"} className="flex items-center gap-1">
        {icons[status]}
        {status}
      </Badge>
    );
  };

  const getDeliveryChannelIcons = (channels) => {
    return (
      <div className="flex flex-wrap items-center gap-2">
        {channels.includes("TELEGRAM") && (
          <span className="flex items-center gap-1 text-xs text-success">
            <MessageSquare className="w-3 h-3" />
            Telegram
          </span>
        )}
        {channels.includes("EMAIL") && (
          <span className="flex items-center gap-1 text-xs text-primary">
            <Mail className="w-3 h-3" />
            Email
          </span>
        )}
      </div>
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
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-warning/10 flex items-center justify-center">
            <Bell className="w-6 h-6 text-warning" />
          </div>
          <div>
            <h1 className="text-2xl font-heading font-bold text-text-primary">Alert Management</h1>
            <p className="text-text-secondary">Create and manage price alerts with Telegram and Email notifications</p>
          </div>
        </div>

        {/* Create Price Alert Form */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Plus className="w-5 h-5 text-success" />
              Create Price Alert
            </CardTitle>
            <CardDescription>
              Configure alerts for real-time notifications via Telegram or Email
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreateAlert} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2" ref={searchRef}>
                  <Label htmlFor="price-symbol">Symbol</Label>
                  <div className="relative">
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-secondary" />
                      <Input
                        id="price-symbol"
                        type="text"
                        placeholder="Search stocks... (e.g., SBI, RELIANCE)"
                        value={formData.symbol}
                        onChange={(e) => {
                          const value = e.target.value.toUpperCase();
                          setFormData({ ...formData, symbol: value });
                          searchStocks(value);
                        }}
                        onFocus={() => {
                          if (formData.symbol) {
                            searchStocks(formData.symbol);
                          }
                        }}
                        required
                        className="pl-10 bg-surface-highlight border-[#1F1F1F]"
                        autoComplete="off"
                      />
                      {isSearching && (
                        <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 animate-spin text-text-secondary" />
                      )}
                    </div>
                    {/* Autocomplete Dropdown */}
                    {showDropdown && stockSearchResults.length > 0 && (
                      <div className="absolute z-50 w-full mt-1 bg-surface-highlight border border-[#1F1F1F] rounded-md shadow-lg max-h-60 overflow-auto">
                        {stockSearchResults.map((stock) => (
                          <button
                            key={stock.symbol}
                            type="button"
                            className="w-full px-4 py-2 text-left hover:bg-primary/10 flex items-center justify-between"
                            onClick={() => {
                              setFormData({ ...formData, symbol: stock.symbol });
                              setShowDropdown(false);
                              setStockSearchResults([]);
                            }}
                          >
                            <div>
                              <span className="font-medium text-text-primary">{stock.symbol}</span>
                              <span className="ml-2 text-sm text-text-secondary">{stock.name}</span>
                            </div>
                            <span className="text-xs text-text-secondary">{stock.exchange}</span>
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="price-condition">Condition</Label>
                  <Select
                    value={formData.condition}
                    onValueChange={(value) => setFormData({ ...formData, condition: value })}
                  >
                    <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {PRICE_CONDITIONS.map(cond => (
                        <SelectItem key={cond.value} value={cond.value}>
                          {cond.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {formData.condition.includes("PERCENT") ? (
                  <div className="space-y-2">
                    <Label htmlFor="percent-change">Percent Change (%)</Label>
                    <Input
                      id="percent-change"
                      type="number"
                      step="0.01"
                      placeholder="5.0"
                      value={formData.percentChange}
                      onChange={(e) => setFormData({ ...formData, percentChange: e.target.value, targetPrice: "0" })}
                      required
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                    <p className="text-xs text-text-secondary">Alert when price changes by this percentage</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    <Label htmlFor="target-price">Target Price</Label>
                    <Input
                      id="target-price"
                      type="number"
                      step="0.01"
                      placeholder="2500.00"
                      value={formData.targetPrice}
                      onChange={(e) => setFormData({ ...formData, targetPrice: e.target.value })}
                      required
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                  </div>
                )}
              </div>

              <div className="space-y-2">
                <Label>Delivery Channels</Label>
                <div className="flex flex-wrap gap-4">
                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="price-telegram"
                      checked={formData.telegram}
                      onCheckedChange={(checked) => setFormData({ ...formData, telegram: checked })}
                    />
                    <Label htmlFor="price-telegram" className="flex items-center gap-2 cursor-pointer">
                      <MessageSquare className="w-4 h-4 text-success" />
                      Telegram
                    </Label>
                  </div>

                  <div className="flex items-center space-x-2">
                    <Checkbox
                      id="price-email"
                      checked={formData.email}
                      onCheckedChange={(checked) => setFormData({ ...formData, email: checked })}
                    />
                    <Label htmlFor="price-email" className="flex items-center gap-2 cursor-pointer">
                      <Mail className="w-4 h-4 text-primary" />
                      Email
                    </Label>
                  </div>
                </div>
              </div>

              <Button type="submit" disabled={isCreating} className="btn-primary">
                {isCreating ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Creating...
                  </>
                ) : (
                  <>
                    <Plus className="w-4 h-4 mr-2" />
                    Create Price Alert
                  </>
                )}
              </Button>
            </form>
          </CardContent>
        </Card>

        {/* Active Alerts */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Bell className="w-5 h-5 text-warning" />
              Active Alerts ({alerts.length})
            </CardTitle>
          </CardHeader>
          <CardContent>
            {alerts.length === 0 ? (
              <p className="text-text-secondary text-center py-8">No alerts configured. Create one above to get started!</p>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Type</TableHead>
                    <TableHead>Details</TableHead>
                    <TableHead>Channels</TableHead>
                    <TableHead>Created</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {alerts.map((alert) => (
                    <TableRow key={alert.alert_id}>
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <TrendingUp className="w-4 h-4" />
                          <span className="text-sm">Price Alert</span>
                        </div>
                      </TableCell>
                      <TableCell className="max-w-xs">
                        <p className="text-sm text-text-primary truncate">{alert.message || JSON.stringify(alert.condition)}</p>
                      </TableCell>
                      <TableCell>
                        {getDeliveryChannelIcons(alert.delivery_channels)}
                      </TableCell>
                      <TableCell className="text-xs text-text-secondary">
                        {formatDateTime(alert.created_at)}
                      </TableCell>
                      <TableCell>{getStatusBadge(alert.status)}</TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => handleDeleteAlert(alert.alert_id)}
                          className="text-danger hover:text-danger hover:bg-danger/10"
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      </TableCell>
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
