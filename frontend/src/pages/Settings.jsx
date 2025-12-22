import { useState, useEffect } from "react";
import axios from "axios";
import { toast } from "sonner";
import { motion } from "framer-motion";
import { 
  getStoredSettings, 
  saveStoredSettings, 
  mergeWithDefaults 
} from "@/utils/settingsStorage";
import { API_URL } from "@/config/api";
import {
  Settings as SettingsIcon,
  Key,
  Save,
  Eye,
  EyeOff,
  CheckCircle,
  AlertCircle,
  Sparkles,
  Bell,
  Mail,
  Webhook,
  MessageSquare
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

const MODELS = {
  openai: [
    { value: "gpt-4.1", label: "GPT-4.1 (Recommended)" },
    { value: "gpt-4o", label: "GPT-4o" },
    { value: "gpt-4.1-mini", label: "GPT-4.1 Mini" },
    { value: "o3-mini", label: "o3-mini (Fast Thinking)" },
    { value: "o1", label: "o1 (Deep Reasoning)" },
  ],
  gemini: [
    { value: "gemini-2.5-flash", label: "Gemini 2.5 Flash" },
    { value: "gemini-2.5-pro", label: "Gemini 2.5 Pro" },
    { value: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
  ],
};

export default function Settings() {
  const [settings, setSettings] = useState({
    openai_api_key: "",
    gemini_api_key: "",
    anthropic_api_key: "",
    finnhub_api_key: "",
    alpaca_api_key: "",
    alpaca_api_secret: "",
    fmp_api_key: "",
    iex_api_key: "",
    polygon_api_key: "",
    twelve_data_api_key: "",
    newsapi_key: "",
    alphavantage_api_key: "",
    telegram_bot_token: "",
    telegram_chat_id: "",
    smtp_host: "",
    smtp_port: "",
    smtp_user: "",
    smtp_password: "",
    smtp_from_email: "",
    webhook_url: "",
    slack_webhook_url: "",
    twilio_account_sid: "",
    twilio_auth_token: "",
    twilio_whatsapp_number: "",
    user_whatsapp_number: "",
    use_tvscreener: true,
    selected_model: "gpt-4.1",
    selected_provider: "openai",
  });

  // Ensure all fields always have string values (never undefined)
  const normalizeSettings = (data) => ({
    openai_api_key: data?.openai_api_key || "",
    gemini_api_key: data?.gemini_api_key || "",
    anthropic_api_key: data?.anthropic_api_key || "",
    finnhub_api_key: data?.finnhub_api_key || "",
    alpaca_api_key: data?.alpaca_api_key || "",
    alpaca_api_secret: data?.alpaca_api_secret || "",
    fmp_api_key: data?.fmp_api_key || "",
    iex_api_key: data?.iex_api_key || "",
    polygon_api_key: data?.polygon_api_key || "",
    twelve_data_api_key: data?.twelve_data_api_key || "",
    newsapi_key: data?.newsapi_key || "",
    alphavantage_api_key: data?.alphavantage_api_key || "",
    telegram_bot_token: data?.telegram_bot_token || "",
    telegram_chat_id: data?.telegram_chat_id || "",
    smtp_host: data?.smtp_host || "",
    smtp_port: data?.smtp_port || "",
    smtp_user: data?.smtp_user || "",
    smtp_password: data?.smtp_password || "",
    smtp_from_email: data?.smtp_from_email || "",
    webhook_url: data?.webhook_url || "",
    slack_webhook_url: data?.slack_webhook_url || "",
    twilio_account_sid: data?.twilio_account_sid || "",
    twilio_auth_token: data?.twilio_auth_token || "",
    twilio_whatsapp_number: data?.twilio_whatsapp_number || "",
    user_whatsapp_number: data?.user_whatsapp_number || "",
    use_tvscreener: data?.use_tvscreener !== undefined ? data.use_tvscreener : true,
    selected_model: data?.selected_model || "gpt-4.1",
    selected_provider: data?.selected_provider || "openai",
  });
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [showFinnhubKey, setShowFinnhubKey] = useState(false);
  const [showAlpacaKey, setShowAlpacaKey] = useState(false);
  const [showAlpacaSecret, setShowAlpacaSecret] = useState(false);
  const [showFMPKey, setShowFMPKey] = useState(false);
  const [showIEXKey, setShowIEXKey] = useState(false);
  const [showPolygonKey, setShowPolygonKey] = useState(false);
  const [showTwelveDataKey, setShowTwelveDataKey] = useState(false);
  const [showNewsAPIKey, setShowNewsAPIKey] = useState(false);
  const [showAlphaVantageKey, setShowAlphaVantageKey] = useState(false);
  const [showTelegramToken, setShowTelegramToken] = useState(false);
  const [showSMTPPassword, setShowSMTPPassword] = useState(false);
  const [showTwilioAuthToken, setShowTwilioAuthToken] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      // Try to fetch from backend first
      const response = await axios.get(`${API_URL}/settings`);
      const backendSettings = normalizeSettings(mergeWithDefaults(response.data));
      setSettings(backendSettings);
      // Also save to localStorage as backup
      saveStoredSettings(backendSettings);
    } catch (error) {
      console.error("Error fetching settings:", error);
      // Fallback to localStorage if backend is unavailable
      const cachedSettings = getStoredSettings();
      if (cachedSettings) {
        setSettings(normalizeSettings(mergeWithDefaults(cachedSettings)));
        toast.info("Loaded cached settings (backend unavailable)");
      } else {
        // If no cached settings, ensure we have default values
        setSettings(normalizeSettings({}));
      }
    }
  };

  const handleChange = (field, value) => {
    setSettings(prev => ({ ...prev, [field]: value }));
    setHasChanges(true);
  };

  const handleProviderChange = (provider) => {
    const defaultModel = MODELS[provider][0].value;
    setSettings(prev => ({ 
      ...prev, 
      selected_provider: provider,
      selected_model: defaultModel
    }));
    setHasChanges(true);
  };

  const saveSettings = async () => {
    setIsSaving(true);
    try {
      // Save to backend
      await axios.post(`${API_URL}/settings`, settings);
      // Also save to localStorage for persistence
      saveStoredSettings(settings);
      toast.success("Settings saved successfully!");
      setHasChanges(false);
      fetchSettings(); // Refresh to get masked keys
    } catch (error) {
      // Even if backend fails, save to localStorage
      saveStoredSettings(settings);
      toast.error("Failed to save to backend, but cached locally");
      console.error(error);
      setHasChanges(false);
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8" data-testid="settings-page">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Header */}
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
            <SettingsIcon className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-2xl font-heading font-bold text-text-primary">Settings</h1>
            <p className="text-text-secondary">Configure your API keys and model preferences</p>
          </div>
        </div>

        {/* API Keys Section */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Key className="w-5 h-5 text-ai-accent" />
              API Keys
            </CardTitle>
            <CardDescription>
              Enter your API keys to enable AI analysis. Your keys are stored securely and never shared.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* OpenAI Key */}
            <div className="space-y-2">
              <Label htmlFor="openai-key" className="text-text-primary">OpenAI API Key</Label>
              <div className="relative">
                <Input
                  id="openai-key"
                  type={showOpenAIKey ? "text" : "password"}
                  value={settings.openai_api_key}
                  onChange={(e) => handleChange("openai_api_key", e.target.value)}
                  placeholder="sk-..."
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  data-testid="openai-key-input"
                />
                <button
                  type="button"
                  onClick={() => setShowOpenAIKey(!showOpenAIKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showOpenAIKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get your API key from{" "}
                <a 
                  href="https://platform.openai.com/api-keys" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  platform.openai.com
                </a>
              </p>
            </div>

            {/* Gemini Key */}
            <div className="space-y-2">
              <Label htmlFor="gemini-key" className="text-text-primary">Google Gemini API Key</Label>
              <div className="relative">
                <Input
                  id="gemini-key"
                  type={showGeminiKey ? "text" : "password"}
                  value={settings.gemini_api_key}
                  onChange={(e) => handleChange("gemini_api_key", e.target.value)}
                  placeholder="AIza..."
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  data-testid="gemini-key-input"
                />
                <button
                  type="button"
                  onClick={() => setShowGeminiKey(!showGeminiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showGeminiKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get your API key from{" "}
                <a
                  href="https://aistudio.google.com/app/apikey"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  aistudio.google.com
                </a>
              </p>
            </div>

            {/* Anthropic Key */}
            <div className="space-y-2">
              <Label htmlFor="anthropic-key" className="text-text-primary">Anthropic API Key (Claude)</Label>
              <div className="relative">
                <Input
                  id="anthropic-key"
                  type={showAnthropicKey ? "text" : "password"}
                  value={settings.anthropic_api_key}
                  onChange={(e) => handleChange("anthropic_api_key", e.target.value)}
                  placeholder="sk-ant-..."
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowAnthropicKey(!showAnthropicKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showAnthropicKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get your API key from{" "}
                <a
                  href="https://console.anthropic.com/settings/keys"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  console.anthropic.com
                </a>
                {" "}• For ensemble analysis
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Data Provider API Keys Section */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Key className="w-5 h-5 text-success" />
              Data Provider API Keys (Optional)
            </CardTitle>
            <CardDescription>
              Add API keys for better data reliability and higher rate limits. All providers have free tiers!
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Finnhub Key */}
            <div className="space-y-2">
              <Label htmlFor="finnhub-key" className="text-text-primary flex items-center gap-2">
                Finnhub API Key
                <span className="text-xs text-success font-normal">(60 calls/min free)</span>
              </Label>
              <div className="relative">
                <Input
                  id="finnhub-key"
                  type={showFinnhubKey ? "text" : "password"}
                  value={settings.finnhub_api_key}
                  onChange={(e) => handleChange("finnhub_api_key", e.target.value)}
                  placeholder="Optional - adds Finnhub as data source"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowFinnhubKey(!showFinnhubKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showFinnhubKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://finnhub.io/register"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  finnhub.io
                </a>
                {" "}• Best for Indian stocks, real-time quotes
              </p>
            </div>

            {/* Alpaca Keys */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <Label className="text-text-primary">
                  Alpaca API (200 calls/min free)
                </Label>
              </div>

              <div className="space-y-2">
                <Label htmlFor="alpaca-key" className="text-sm text-text-secondary">API Key</Label>
                <div className="relative">
                  <Input
                    id="alpaca-key"
                    type={showAlpacaKey ? "text" : "password"}
                    value={settings.alpaca_api_key}
                    onChange={(e) => handleChange("alpaca_api_key", e.target.value)}
                    placeholder="Optional - requires both key and secret"
                    className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                  <button
                    type="button"
                    onClick={() => setShowAlpacaKey(!showAlpacaKey)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                  >
                    {showAlpacaKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="alpaca-secret" className="text-sm text-text-secondary">API Secret</Label>
                <div className="relative">
                  <Input
                    id="alpaca-secret"
                    type={showAlpacaSecret ? "text" : "password"}
                    value={settings.alpaca_api_secret}
                    onChange={(e) => handleChange("alpaca_api_secret", e.target.value)}
                    placeholder="Optional - requires both key and secret"
                    className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                  <button
                    type="button"
                    onClick={() => setShowAlpacaSecret(!showAlpacaSecret)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                  >
                    {showAlpacaSecret ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <p className="text-xs text-text-secondary">
                Get free API credentials from{" "}
                <a
                  href="https://alpaca.markets/docs/api-references/market-data-api/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  alpaca.markets
                </a>
                {" "}• Best for US stocks
              </p>
            </div>

            {/* FMP Key */}
            <div className="space-y-2">
              <Label htmlFor="fmp-key" className="text-text-primary flex items-center gap-2">
                Financial Modeling Prep API Key
                <span className="text-xs text-success font-normal">(250 calls/day free)</span>
              </Label>
              <div className="relative">
                <Input
                  id="fmp-key"
                  type={showFMPKey ? "text" : "password"}
                  value={settings.fmp_api_key}
                  onChange={(e) => handleChange("fmp_api_key", e.target.value)}
                  placeholder="Optional - adds FMP as data source"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowFMPKey(!showFMPKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showFMPKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://site.financialmodelingprep.com/developer/docs"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  financialmodelingprep.com
                </a>
                {" "}• Best for fundamentals and US stocks
              </p>
            </div>

            {/* IEX Cloud Key */}
            <div className="space-y-2">
              <Label htmlFor="iex-key" className="text-text-primary flex items-center gap-2">
                IEX Cloud API Key
                <span className="text-xs text-warning font-normal">(May be down)</span>
              </Label>
              <div className="relative">
                <Input
                  id="iex-key"
                  type={showIEXKey ? "text" : "password"}
                  value={settings.iex_api_key}
                  onChange={(e) => handleChange("iex_api_key", e.target.value)}
                  placeholder="Optional - may be unavailable"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowIEXKey(!showIEXKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showIEXKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get API key from{" "}
                <a
                  href="https://iexcloud.io/console/tokens"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  iexcloud.io
                </a>
                {" "}• Note: Service may be experiencing issues
              </p>
            </div>

            {/* Polygon.io Key */}
            <div className="space-y-2">
              <Label htmlFor="polygon-key" className="text-text-primary flex items-center gap-2">
                Polygon.io API Key
                <span className="text-xs text-success font-normal">(5 calls/min free)</span>
              </Label>
              <div className="relative">
                <Input
                  id="polygon-key"
                  type={showPolygonKey ? "text" : "password"}
                  value={settings.polygon_api_key}
                  onChange={(e) => handleChange("polygon_api_key", e.target.value)}
                  placeholder="Optional - EOD data, 2 years history"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowPolygonKey(!showPolygonKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showPolygonKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://polygon.io/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  polygon.io
                </a>
                {" "}• End-of-day data, no credit card required
              </p>
            </div>

            {/* Twelve Data Key */}
            <div className="space-y-2">
              <Label htmlFor="twelve-data-key" className="text-text-primary flex items-center gap-2">
                Twelve Data API Key
                <span className="text-xs text-success font-normal">(8 calls/min, 800/day free) ⭐</span>
              </Label>
              <div className="relative">
                <Input
                  id="twelve-data-key"
                  type={showTwelveDataKey ? "text" : "password"}
                  value={settings.twelve_data_api_key}
                  onChange={(e) => handleChange("twelve_data_api_key", e.target.value)}
                  placeholder="Recommended - real-time data, best free tier"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowTwelveDataKey(!showTwelveDataKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showTwelveDataKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://twelvedata.com/"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  twelvedata.com
                </a>
                {" "}• Real-time quotes ~170ms, stocks/forex/crypto
              </p>
            </div>

            {/* NewsAPI Key */}
            <div className="space-y-2">
              <Label htmlFor="newsapi-key" className="text-text-primary flex items-center gap-2">
                NewsAPI Key
                <span className="text-xs text-success font-normal">(100 calls/day free)</span>
              </Label>
              <div className="relative">
                <Input
                  id="newsapi-key"
                  type={showNewsAPIKey ? "text" : "password"}
                  value={settings.newsapi_key}
                  onChange={(e) => handleChange("newsapi_key", e.target.value)}
                  placeholder="Optional - for news sentiment analysis"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowNewsAPIKey(!showNewsAPIKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showNewsAPIKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://newsapi.org/register"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  newsapi.org
                </a>
                {" "}• Financial news headlines and sentiment
              </p>
            </div>

            {/* AlphaVantage Key */}
            <div className="space-y-2">
              <Label htmlFor="alphavantage-key" className="text-text-primary flex items-center gap-2">
                Alpha Vantage API Key
                <span className="text-xs text-success font-normal">(25 calls/day free)</span>
              </Label>
              <div className="relative">
                <Input
                  id="alphavantage-key"
                  type={showAlphaVantageKey ? "text" : "password"}
                  value={settings.alphavantage_api_key}
                  onChange={(e) => handleChange("alphavantage_api_key", e.target.value)}
                  placeholder="Optional - news + market data"
                  className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <button
                  type="button"
                  onClick={() => setShowAlphaVantageKey(!showAlphaVantageKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                >
                  {showAlphaVantageKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
              <p className="text-xs text-text-secondary">
                Get free API key from{" "}
                <a
                  href="https://www.alphavantage.co/support/#api-key"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  alphavantage.co
                </a>
                {" "}• News sentiment + technical indicators
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Alert & Notification Settings */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Bell className="w-5 h-5 text-warning" />
              Alert & Notification Settings
            </CardTitle>
            <CardDescription>
              Configure Telegram bot, email SMTP, and webhook for real-time alerts
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Telegram Settings */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <Bell className="w-4 h-4 text-warning" />
                <Label className="text-text-primary">
                  Telegram Bot (Recommended for Price Alerts)
                </Label>
              </div>

              <div className="space-y-2">
                <Label htmlFor="telegram-token" className="text-sm text-text-secondary">Bot Token</Label>
                <div className="relative">
                  <Input
                    id="telegram-token"
                    type={showTelegramToken ? "text" : "password"}
                    value={settings.telegram_bot_token}
                    onChange={(e) => handleChange("telegram_bot_token", e.target.value)}
                    placeholder="Get from @BotFather on Telegram"
                    className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                  <button
                    type="button"
                    onClick={() => setShowTelegramToken(!showTelegramToken)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                  >
                    {showTelegramToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="telegram-chat-id" className="text-sm text-text-secondary">Chat ID</Label>
                <Input
                  id="telegram-chat-id"
                  type="text"
                  value={settings.telegram_chat_id}
                  onChange={(e) => handleChange("telegram_chat_id", e.target.value)}
                  placeholder="Get from @userinfobot on Telegram"
                  className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
              </div>

              <p className="text-xs text-text-secondary">
                1. Create bot with @BotFather → /newbot<br/>
                2. Get your Chat ID from @userinfobot<br/>
                3. Start conversation with your bot
              </p>
            </div>

            {/* Email SMTP Settings */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <Mail className="w-4 h-4 text-primary" />
                <Label className="text-text-primary">
                  Email SMTP (For Email Alerts)
                </Label>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="smtp-host" className="text-sm text-text-secondary">SMTP Host</Label>
                  <Input
                    id="smtp-host"
                    type="text"
                    value={settings.smtp_host}
                    onChange={(e) => handleChange("smtp_host", e.target.value)}
                    placeholder="smtp.gmail.com"
                    className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="smtp-port" className="text-sm text-text-secondary">SMTP Port</Label>
                  <Input
                    id="smtp-port"
                    type="number"
                    value={settings.smtp_port}
                    onChange={(e) => handleChange("smtp_port", e.target.value)}
                    placeholder="587"
                    className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="smtp-user" className="text-sm text-text-secondary">SMTP Username (Email)</Label>
                <Input
                  id="smtp-user"
                  type="email"
                  value={settings.smtp_user}
                  onChange={(e) => handleChange("smtp_user", e.target.value)}
                  placeholder="your-email@gmail.com"
                  className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="smtp-password" className="text-sm text-text-secondary">SMTP Password (App Password)</Label>
                <div className="relative">
                  <Input
                    id="smtp-password"
                    type={showSMTPPassword ? "text" : "password"}
                    value={settings.smtp_password}
                    onChange={(e) => handleChange("smtp_password", e.target.value)}
                    placeholder="App-specific password"
                    className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                  <button
                    type="button"
                    onClick={() => setShowSMTPPassword(!showSMTPPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                  >
                    {showSMTPPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                  </button>
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="smtp-from-email" className="text-sm text-text-secondary">From Email Address</Label>
                <Input
                  id="smtp-from-email"
                  type="email"
                  value={settings.smtp_from_email}
                  onChange={(e) => handleChange("smtp_from_email", e.target.value)}
                  placeholder="alerts@yourapp.com"
                  className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
              </div>

              <p className="text-xs text-text-secondary">
                For Gmail: Enable 2FA, then create App Password at{" "}
                <a
                  href="https://myaccount.google.com/apppasswords"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  myaccount.google.com/apppasswords
                </a>
              </p>
            </div>

            {/* Slack Settings */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-purple-500" />
                <Label className="text-text-primary">
                  Slack Webhook (FREE - Recommended for Teams)
                </Label>
              </div>

              <div className="space-y-2">
                <Input
                  id="slack-webhook"
                  type="url"
                  value={settings.slack_webhook_url}
                  onChange={(e) => handleChange("slack_webhook_url", e.target.value)}
                  placeholder="https://hooks.slack.com/services/..."
                  className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <p className="text-xs text-text-secondary">
                  1. Go to{" "}
                  <a
                    href="https://api.slack.com/messaging/webhooks"
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-primary hover:underline"
                  >
                    api.slack.com/messaging/webhooks
                  </a>
                  <br />
                  2. Create app → Enable Incoming Webhooks<br />
                  3. Add webhook to your channel → Copy URL
                </p>
              </div>
            </div>

            {/* WhatsApp Settings */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <MessageSquare className="w-4 h-4 text-success" />
                <Label className="text-text-primary">
                  WhatsApp via Twilio ($15 free trial)
                </Label>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="twilio-sid" className="text-sm text-text-secondary">Account SID</Label>
                  <Input
                    id="twilio-sid"
                    type="text"
                    value={settings.twilio_account_sid}
                    onChange={(e) => handleChange("twilio_account_sid", e.target.value)}
                    placeholder="ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
                    className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="twilio-token" className="text-sm text-text-secondary">Auth Token</Label>
                  <div className="relative">
                    <Input
                      id="twilio-token"
                      type={showTwilioAuthToken ? "text" : "password"}
                      value={settings.twilio_auth_token}
                      onChange={(e) => handleChange("twilio_auth_token", e.target.value)}
                      placeholder="Your auth token"
                      className="pr-12 bg-surface-highlight border-[#1F1F1F] text-text-primary"
                    />
                    <button
                      type="button"
                      onClick={() => setShowTwilioAuthToken(!showTwilioAuthToken)}
                      className="absolute right-3 top-1/2 -translate-y-1/2 text-text-secondary hover:text-text-primary"
                    >
                      {showTwilioAuthToken ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                    </button>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="twilio-whatsapp" className="text-sm text-text-secondary">Twilio WhatsApp Number</Label>
                  <Input
                    id="twilio-whatsapp"
                    type="text"
                    value={settings.twilio_whatsapp_number}
                    onChange={(e) => handleChange("twilio_whatsapp_number", e.target.value)}
                    placeholder="whatsapp:+14155238886"
                    className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="user-whatsapp" className="text-sm text-text-secondary">Your WhatsApp Number</Label>
                  <Input
                    id="user-whatsapp"
                    type="text"
                    value={settings.user_whatsapp_number}
                    onChange={(e) => handleChange("user_whatsapp_number", e.target.value)}
                    placeholder="whatsapp:+919876543210"
                    className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                  />
                </div>
              </div>

              <p className="text-xs text-text-secondary">
                Sign up at{" "}
                <a
                  href="https://www.twilio.com/try-twilio"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-primary hover:underline"
                >
                  twilio.com/try-twilio
                </a>
                {" "}→ Get $15 free credit → Go to WhatsApp Sandbox → Join sandbox → Get credentials
              </p>
            </div>

            {/* Webhook Settings */}
            <div className="space-y-4 p-4 rounded-lg bg-surface-highlight/50 border border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <Webhook className="w-4 h-4 text-ai-accent" />
                <Label className="text-text-primary">
                  Webhook URL (For Custom Integrations)
                </Label>
              </div>

              <div className="space-y-2">
                <Input
                  id="webhook-url"
                  type="url"
                  value={settings.webhook_url}
                  onChange={(e) => handleChange("webhook_url", e.target.value)}
                  placeholder="https://your-webhook-endpoint.com/alerts"
                  className="bg-surface-highlight border-[#1F1F1F] text-text-primary"
                />
                <p className="text-xs text-text-secondary">
                  Alerts will be sent as POST requests to this URL
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Model Selection */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-ai-accent" />
              Default Model
            </CardTitle>
            <CardDescription>
              Choose your preferred AI model for stock analysis
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Provider Selection */}
            <div className="space-y-2">
              <Label className="text-text-primary">AI Provider</Label>
              <Select 
                value={settings.selected_provider} 
                onValueChange={handleProviderChange}
              >
                <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]" data-testid="provider-select">
                  <SelectValue placeholder="Select provider" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="openai">OpenAI</SelectItem>
                  <SelectItem value="gemini">Google Gemini</SelectItem>
                </SelectContent>
              </Select>
            </div>

            {/* Model Selection */}
            <div className="space-y-2">
              <Label className="text-text-primary">Model</Label>
              <Select 
                value={settings.selected_model} 
                onValueChange={(value) => handleChange("selected_model", value)}
              >
                <SelectTrigger className="bg-surface-highlight border-[#1F1F1F]" data-testid="model-select">
                  <SelectValue placeholder="Select model" />
                </SelectTrigger>
                <SelectContent>
                  {MODELS[settings.selected_provider]?.map((model) => (
                    <SelectItem key={model.value} value={model.value}>
                      {model.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Model Info Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
              <div className={`p-4 rounded-lg border ${
                settings.selected_provider === "openai" 
                  ? "bg-primary/5 border-primary/20" 
                  : "bg-surface-highlight border-[#1F1F1F]"
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {settings.selected_provider === "openai" && settings.openai_api_key ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-text-secondary" />
                  )}
                  <span className="font-medium text-text-primary">OpenAI</span>
                </div>
                <p className="text-xs text-text-secondary">
                  GPT-4.1 for deep analysis, o3-mini for fast thinking
                </p>
              </div>

              <div className={`p-4 rounded-lg border ${
                settings.selected_provider === "gemini" 
                  ? "bg-primary/5 border-primary/20" 
                  : "bg-surface-highlight border-[#1F1F1F]"
              }`}>
                <div className="flex items-center gap-2 mb-2">
                  {settings.selected_provider === "gemini" && settings.gemini_api_key ? (
                    <CheckCircle className="w-4 h-4 text-success" />
                  ) : (
                    <AlertCircle className="w-4 h-4 text-text-secondary" />
                  )}
                  <span className="font-medium text-text-primary">Gemini</span>
                </div>
                <p className="text-xs text-text-secondary">
                  Flash models for fast, cost-effective analysis
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Save Button */}
        <div className="flex justify-end">
          <Button
            onClick={saveSettings}
            disabled={isSaving || !hasChanges}
            className="btn-primary"
            data-testid="save-settings-btn"
          >
            {isSaving ? (
              "Saving..."
            ) : (
              <>
                <Save className="w-4 h-4 mr-2" />
                Save Settings
              </>
            )}
          </Button>
        </div>

        {/* Info Note */}
        <Card className="card-surface border-ai-accent/20">
          <CardContent className="py-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-ai-accent/10 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-ai-accent" />
              </div>
              <div>
                <p className="text-sm font-medium text-text-primary">Multi-Agent AI System</p>
                <p className="text-xs text-text-secondary mt-1">
                  NeuralTrader uses a sophisticated multi-agent workflow including Data Collection, 
                  Technical Analysis, RAG Knowledge Retrieval, Deep Reasoning, and Validation agents 
                  to provide comprehensive trading recommendations.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
