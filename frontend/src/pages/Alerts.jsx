import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import {
  Bell,
  Plus,
  Trash2,
  TrendingUp,
  TrendingDown,
  Activity,
  Newspaper,
  DollarSign,
  Loader2,
  Mail,
  MessageSquare,
  Webhook as WebhookIcon,
  CheckCircle,
  XCircle,
  Clock
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
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { API_URL } from "@/config/api";

const ALERT_TYPES = {
  PRICE: "Price Alert",
  PATTERN: "Pattern Alert",
  PORTFOLIO: "Portfolio Alert"
};

const PRICE_CONDITIONS = [
  { value: "ABOVE", label: "Above" },
  { value: "BELOW", label: "Below" },
  { value: "CROSSES_ABOVE", label: "Crosses Above" },
  { value: "CROSSES_BELOW", label: "Crosses Below" },
  { value: "PERCENT_CHANGE_ABOVE", label: "% Change Above" },
  { value: "PERCENT_CHANGE_BELOW", label: "% Change Below" }
];

const CANDLESTICK_PATTERNS = [
  "HAMMER",
  "INVERTED_HAMMER",
  "BULLISH_ENGULFING",
  "BEARISH_ENGULFING",
  "MORNING_STAR",
  "EVENING_STAR",
  "DOJI",
  "SHOOTING_STAR"
];

const PORTFOLIO_METRICS = [
  { value: "drawdown", label: "Drawdown %" },
  { value: "total_pnl", label: "Total P&L" },
  { value: "total_return_pct", label: "Total Return %" }
];

export default function Alerts() {
  const [alerts, setAlerts] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [activeTab, setActiveTab] = useState("price");

  // Price alert form
  const [priceAlertForm, setPriceAlertForm] = useState({
    symbol: "",
    condition: "CROSSES_ABOVE",
    targetPrice: "",
    percentChange: "",
    telegram: true,
    email: false,
    slack: false,
    whatsapp: false,
    webhook: false
  });

  // Pattern alert form
  const [patternAlertForm, setPatternAlertForm] = useState({
    symbol: "",
    patterns: [],
    telegram: true,
    email: false,
    slack: false,
    whatsapp: false,
    webhook: false
  });

  // Portfolio alert form
  const [portfolioAlertForm, setPortfolioAlertForm] = useState({
    metric: "drawdown",
    threshold: "",
    condition: "above",
    telegram: true,
    email: false,
    slack: false,
    whatsapp: false,
    webhook: false
  });

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

  const getDeliveryChannels = (form) => {
    const channels = [];
    if (form.telegram) channels.push("TELEGRAM");
    if (form.email) channels.push("EMAIL");
    if (form.slack) channels.push("SLACK");
    if (form.whatsapp) channels.push("WHATSAPP");
    if (form.webhook) channels.push("WEBHOOK");
    return channels;
  };

  const handleCreatePriceAlert = async (e) => {
    e.preventDefault();
    setIsCreating(true);

    try {
      const isPercentCondition = priceAlertForm.condition.includes("PERCENT");

      const alertData = {
        user_id: "default",
        symbol: priceAlertForm.symbol.toUpperCase(),
        condition: priceAlertForm.condition,
        target_price: isPercentCondition ? 0 : parseFloat(priceAlertForm.targetPrice),
        delivery_channels: getDeliveryChannels(priceAlertForm),
        percent_change: priceAlertForm.percentChange ? parseFloat(priceAlertForm.percentChange) : null
      };

      await axios.post(`${API_URL}/alerts/price`, alertData);
      toast.success("Price alert created successfully!");

      // Reset form
      setPriceAlertForm({
        symbol: "",
        condition: "CROSSES_ABOVE",
        targetPrice: "",
        percentChange: "",
        telegram: true,
        email: false,
        slack: false,
        whatsapp: false,
        webhook: false
      });

      fetchAlerts();
    } catch (error) {
      console.error("Error creating price alert:", error);
      toast.error(error.response?.data?.detail || "Failed to create alert");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCreatePatternAlert = async (e) => {
    e.preventDefault();
    setIsCreating(true);

    try {
      const alertData = {
        user_id: "default",
        symbol: patternAlertForm.symbol.toUpperCase(),
        pattern_types: patternAlertForm.patterns,
        delivery_channels: getDeliveryChannels(patternAlertForm)
      };

      await axios.post(`${API_URL}/alerts/pattern`, alertData);
      toast.success("Pattern alert created successfully!");

      // Reset form
      setPatternAlertForm({
        symbol: "",
        patterns: [],
        telegram: true,
        email: false,
        slack: false,
        whatsapp: false,
        webhook: false
      });

      fetchAlerts();
    } catch (error) {
      console.error("Error creating pattern alert:", error);
      toast.error(error.response?.data?.detail || "Failed to create alert");
    } finally {
      setIsCreating(false);
    }
  };

  const handleCreatePortfolioAlert = async (e) => {
    e.preventDefault();
    setIsCreating(true);

    try {
      const alertData = {
        user_id: "default",
        metric: portfolioAlertForm.metric,
        threshold: parseFloat(portfolioAlertForm.threshold),
        condition: portfolioAlertForm.condition,
        delivery_channels: getDeliveryChannels(portfolioAlertForm)
      };

      await axios.post(`${API_URL}/alerts/portfolio`, alertData);
      toast.success("Portfolio alert created successfully!");

      // Reset form
      setPortfolioAlertForm({
        metric: "drawdown",
        threshold: "",
        condition: "above",
        telegram: true,
        email: false,
        slack: false,
        whatsapp: false,
        webhook: false
      });

      fetchAlerts();
    } catch (error) {
      console.error("Error creating portfolio alert:", error);
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

  const togglePattern = (pattern) => {
    setPatternAlertForm(prev => ({
      ...prev,
      patterns: prev.patterns.includes(pattern)
        ? prev.patterns.filter(p => p !== pattern)
        : [...prev.patterns, pattern]
    }));
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

  const getAlertIcon = (alertType) => {
    const icons = {
      PRICE: <TrendingUp className="w-4 h-4" />,
      PATTERN: <Activity className="w-4 h-4" />,
      NEWS: <Newspaper className="w-4 h-4" />,
      PORTFOLIO: <DollarSign className="w-4 h-4" />,
      TECHNICAL: <Activity className="w-4 h-4" />
    };
    return icons[alertType] || <Bell className="w-4 h-4" />;
  };

  const getDeliveryChannelIcons = (channels) => {
    return (
      <div className="flex items-center gap-1">
        {channels.includes("TELEGRAM") && <MessageSquare className="w-3 h-3 text-success" />}
        {channels.includes("EMAIL") && <Mail className="w-3 h-3 text-primary" />}
        {channels.includes("SLACK") && <MessageSquare className="w-3 h-3 text-purple-500" />}
        {channels.includes("WHATSAPP") && <MessageSquare className="w-3 h-3 text-success" />}
        {channels.includes("WEBHOOK") && <WebhookIcon className="w-3 h-3 text-ai-accent" />}
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
            <p className="text-text-secondary">Create and manage price, pattern, and portfolio alerts</p>
          </div>
        </div>

        {/* Create Alert Form */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Plus className="w-5 h-5 text-success" />
              Create New Alert
            </CardTitle>
            <CardDescription>
              Configure alerts for real-time notifications via Telegram, Email, Slack, WhatsApp, or Webhook
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Tabs value={activeTab} onValueChange={setActiveTab}>
              <TabsList className="grid w-full grid-cols-3">
                <TabsTrigger value="price">Price Alert</TabsTrigger>
                <TabsTrigger value="pattern">Pattern Alert</TabsTrigger>
                <TabsTrigger value="portfolio">Portfolio Alert</TabsTrigger>
              </TabsList>

              {/* Price Alert Form */}
              <TabsContent value="price" className="space-y-4 mt-4">
                <form onSubmit={handleCreatePriceAlert} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="price-symbol">Symbol</Label>
                      <Input
                        id="price-symbol"
                        type="text"
                        placeholder="RELIANCE"
                        value={priceAlertForm.symbol}
                        onChange={(e) => setPriceAlertForm({ ...priceAlertForm, symbol: e.target.value.toUpperCase() })}
                        required
                        className="bg-surface-highlight border-[#1F1F1F]"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="price-condition">Condition</Label>
                      <Select
                        value={priceAlertForm.condition}
                        onValueChange={(value) => setPriceAlertForm({ ...priceAlertForm, condition: value })}
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
                    {priceAlertForm.condition.includes("PERCENT") ? (
                      <div className="space-y-2">
                        <Label htmlFor="percent-change">Percent Change (%)</Label>
                        <Input
                          id="percent-change"
                          type="number"
                          step="0.01"
                          placeholder="5.0"
                          value={priceAlertForm.percentChange}
                          onChange={(e) => setPriceAlertForm({ ...priceAlertForm, percentChange: e.target.value, targetPrice: "0" })}
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
                          value={priceAlertForm.targetPrice}
                          onChange={(e) => setPriceAlertForm({ ...priceAlertForm, targetPrice: e.target.value })}
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
                          checked={priceAlertForm.telegram}
                          onCheckedChange={(checked) => setPriceAlertForm({ ...priceAlertForm, telegram: checked })}
                        />
                        <Label htmlFor="price-telegram" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          Telegram
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="price-email"
                          checked={priceAlertForm.email}
                          onCheckedChange={(checked) => setPriceAlertForm({ ...priceAlertForm, email: checked })}
                        />
                        <Label htmlFor="price-email" className="flex items-center gap-2 cursor-pointer">
                          <Mail className="w-4 h-4 text-primary" />
                          Email
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="price-slack"
                          checked={priceAlertForm.slack}
                          onCheckedChange={(checked) => setPriceAlertForm({ ...priceAlertForm, slack: checked })}
                        />
                        <Label htmlFor="price-slack" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-purple-500" />
                          Slack
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="price-whatsapp"
                          checked={priceAlertForm.whatsapp}
                          onCheckedChange={(checked) => setPriceAlertForm({ ...priceAlertForm, whatsapp: checked })}
                        />
                        <Label htmlFor="price-whatsapp" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          WhatsApp
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="price-webhook"
                          checked={priceAlertForm.webhook}
                          onCheckedChange={(checked) => setPriceAlertForm({ ...priceAlertForm, webhook: checked })}
                        />
                        <Label htmlFor="price-webhook" className="flex items-center gap-2 cursor-pointer">
                          <WebhookIcon className="w-4 h-4 text-ai-accent" />
                          Webhook
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
              </TabsContent>

              {/* Pattern Alert Form */}
              <TabsContent value="pattern" className="space-y-4 mt-4">
                <form onSubmit={handleCreatePatternAlert} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="pattern-symbol">Symbol</Label>
                    <Input
                      id="pattern-symbol"
                      type="text"
                      placeholder="RELIANCE"
                      value={patternAlertForm.symbol}
                      onChange={(e) => setPatternAlertForm({ ...patternAlertForm, symbol: e.target.value.toUpperCase() })}
                      required
                      className="bg-surface-highlight border-[#1F1F1F]"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Candlestick Patterns (select multiple)</Label>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                      {CANDLESTICK_PATTERNS.map(pattern => (
                        <div key={pattern} className="flex items-center space-x-2">
                          <Checkbox
                            id={`pattern-${pattern}`}
                            checked={patternAlertForm.patterns.includes(pattern)}
                            onCheckedChange={() => togglePattern(pattern)}
                          />
                          <Label htmlFor={`pattern-${pattern}`} className="text-sm cursor-pointer">
                            {pattern.replace(/_/g, ' ')}
                          </Label>
                        </div>
                      ))}
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Delivery Channels</Label>
                    <div className="flex flex-wrap gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="pattern-telegram"
                          checked={patternAlertForm.telegram}
                          onCheckedChange={(checked) => setPatternAlertForm({ ...patternAlertForm, telegram: checked })}
                        />
                        <Label htmlFor="pattern-telegram" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          Telegram
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="pattern-email"
                          checked={patternAlertForm.email}
                          onCheckedChange={(checked) => setPatternAlertForm({ ...patternAlertForm, email: checked })}
                        />
                        <Label htmlFor="pattern-email" className="flex items-center gap-2 cursor-pointer">
                          <Mail className="w-4 h-4 text-primary" />
                          Email
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="pattern-slack"
                          checked={patternAlertForm.slack}
                          onCheckedChange={(checked) => setPatternAlertForm({ ...patternAlertForm, slack: checked })}
                        />
                        <Label htmlFor="pattern-slack" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-purple-500" />
                          Slack
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="pattern-whatsapp"
                          checked={patternAlertForm.whatsapp}
                          onCheckedChange={(checked) => setPatternAlertForm({ ...patternAlertForm, whatsapp: checked })}
                        />
                        <Label htmlFor="pattern-whatsapp" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          WhatsApp
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="pattern-webhook"
                          checked={patternAlertForm.webhook}
                          onCheckedChange={(checked) => setPatternAlertForm({ ...patternAlertForm, webhook: checked })}
                        />
                        <Label htmlFor="pattern-webhook" className="flex items-center gap-2 cursor-pointer">
                          <WebhookIcon className="w-4 h-4 text-ai-accent" />
                          Webhook
                        </Label>
                      </div>
                    </div>
                  </div>

                  <Button type="submit" disabled={isCreating || patternAlertForm.patterns.length === 0} className="btn-primary">
                    {isCreating ? (
                      <>
                        <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                        Creating...
                      </>
                    ) : (
                      <>
                        <Plus className="w-4 h-4 mr-2" />
                        Create Pattern Alert
                      </>
                    )}
                  </Button>
                </form>
              </TabsContent>

              {/* Portfolio Alert Form */}
              <TabsContent value="portfolio" className="space-y-4 mt-4">
                <form onSubmit={handleCreatePortfolioAlert} className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="portfolio-metric">Metric</Label>
                      <Select
                        value={portfolioAlertForm.metric}
                        onValueChange={(value) => setPortfolioAlertForm({ ...portfolioAlertForm, metric: value })}
                      >
                        <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          {PORTFOLIO_METRICS.map(metric => (
                            <SelectItem key={metric.value} value={metric.value}>
                              {metric.label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="portfolio-condition">Condition</Label>
                      <Select
                        value={portfolioAlertForm.condition}
                        onValueChange={(value) => setPortfolioAlertForm({ ...portfolioAlertForm, condition: value })}
                      >
                        <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="above">Above</SelectItem>
                          <SelectItem value="below">Below</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    <div className="space-y-2">
                      <Label htmlFor="portfolio-threshold">Threshold</Label>
                      <Input
                        id="portfolio-threshold"
                        type="number"
                        step="0.01"
                        placeholder="5.0"
                        value={portfolioAlertForm.threshold}
                        onChange={(e) => setPortfolioAlertForm({ ...portfolioAlertForm, threshold: e.target.value })}
                        required
                        className="bg-surface-highlight border-[#1F1F1F]"
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label>Delivery Channels</Label>
                    <div className="flex flex-wrap gap-4">
                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="portfolio-telegram"
                          checked={portfolioAlertForm.telegram}
                          onCheckedChange={(checked) => setPortfolioAlertForm({ ...portfolioAlertForm, telegram: checked })}
                        />
                        <Label htmlFor="portfolio-telegram" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          Telegram
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="portfolio-email"
                          checked={portfolioAlertForm.email}
                          onCheckedChange={(checked) => setPortfolioAlertForm({ ...portfolioAlertForm, email: checked })}
                        />
                        <Label htmlFor="portfolio-email" className="flex items-center gap-2 cursor-pointer">
                          <Mail className="w-4 h-4 text-primary" />
                          Email
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="portfolio-slack"
                          checked={portfolioAlertForm.slack}
                          onCheckedChange={(checked) => setPortfolioAlertForm({ ...portfolioAlertForm, slack: checked })}
                        />
                        <Label htmlFor="portfolio-slack" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-purple-500" />
                          Slack
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="portfolio-whatsapp"
                          checked={portfolioAlertForm.whatsapp}
                          onCheckedChange={(checked) => setPortfolioAlertForm({ ...portfolioAlertForm, whatsapp: checked })}
                        />
                        <Label htmlFor="portfolio-whatsapp" className="flex items-center gap-2 cursor-pointer">
                          <MessageSquare className="w-4 h-4 text-success" />
                          WhatsApp
                        </Label>
                      </div>

                      <div className="flex items-center space-x-2">
                        <Checkbox
                          id="portfolio-webhook"
                          checked={portfolioAlertForm.webhook}
                          onCheckedChange={(checked) => setPortfolioAlertForm({ ...portfolioAlertForm, webhook: checked })}
                        />
                        <Label htmlFor="portfolio-webhook" className="flex items-center gap-2 cursor-pointer">
                          <WebhookIcon className="w-4 h-4 text-ai-accent" />
                          Webhook
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
                        Create Portfolio Alert
                      </>
                    )}
                  </Button>
                </form>
              </TabsContent>
            </Tabs>
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
                          {getAlertIcon(alert.alert_type)}
                          <span className="text-sm">{ALERT_TYPES[alert.alert_type] || alert.alert_type}</span>
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
