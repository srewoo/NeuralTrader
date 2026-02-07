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
    telegram_bot_token: "",
    telegram_chat_id: "",
    smtp_host: "",
    smtp_port: "",
    smtp_user: "",
    smtp_password: "",
    smtp_from_email: "",
    selected_model: "gpt-4.1",
    selected_provider: "openai",
  });

  // Ensure all fields always have string values (never undefined)
  const normalizeSettings = (data) => ({
    openai_api_key: data?.openai_api_key || "",
    gemini_api_key: data?.gemini_api_key || "",
    anthropic_api_key: data?.anthropic_api_key || "",
    telegram_bot_token: data?.telegram_bot_token || "",
    telegram_chat_id: data?.telegram_chat_id || "",
    smtp_host: data?.smtp_host || "",
    smtp_port: data?.smtp_port || "",
    smtp_user: data?.smtp_user || "",
    smtp_password: data?.smtp_password || "",
    smtp_from_email: data?.smtp_from_email || "",
    selected_model: data?.selected_model || "gpt-4.1",
    selected_provider: data?.selected_provider || "openai",
  });

  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [showAnthropicKey, setShowAnthropicKey] = useState(false);
  const [showTelegramToken, setShowTelegramToken] = useState(false);
  const [showSMTPPassword, setShowSMTPPassword] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async (skipLocalStorageSave = false) => {
    try {
      // Try to fetch from backend first
      const response = await axios.get(`${API_URL}/settings`);
      const backendSettings = normalizeSettings(mergeWithDefaults(response.data));
      setSettings(backendSettings);
      // Only save to localStorage on initial load (masked keys are fine for display)
      // Skip saving when refreshing after a save to preserve unmasked keys
      if (!skipLocalStorageSave) {
        // Note: localStorage will have masked keys, which is expected for security
        // The real keys are stored in MongoDB backend
        saveStoredSettings(backendSettings);
      }
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
      // Save to backend - this is the source of truth
      await axios.post(`${API_URL}/settings`, settings);
      toast.success("Settings saved successfully!");
      setHasChanges(false);
      // Refresh UI with masked keys, but skip saving to localStorage
      // This preserves the fact that backend is the source of truth
      // and localStorage is only used for display fallback (with masked keys)
      fetchSettings(true);
    } catch (error) {
      // Don't save to localStorage on backend failure - prevents data inconsistency
      toast.error("Failed to save settings. Please try again.");
      console.error(error);
      // Keep hasChanges true so user knows settings weren't saved
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

        {/* Section 1: LLM Provider & Model Selection */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-ai-accent" />
              LLM Provider & Model
            </CardTitle>
            <CardDescription>
              Choose your preferred AI provider and model for stock analysis
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

            {/* Provider Info Cards */}
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

        {/* Section 2: AI API Keys */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Key className="w-5 h-5 text-ai-accent" />
              AI API Keys
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
                {" "}-- For ensemble analysis
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Section 3: Notification Settings */}
        <Card className="card-surface">
          <CardHeader>
            <CardTitle className="text-lg font-heading flex items-center gap-2">
              <Bell className="w-5 h-5 text-warning" />
              Notification Settings
            </CardTitle>
            <CardDescription>
              Configure Telegram bot and email SMTP for real-time alerts
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
                1. Create bot with @BotFather on Telegram, send /newbot<br/>
                2. Get your Chat ID from @userinfobot<br/>
                3. Start a conversation with your bot before sending alerts
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

        {/* Multi-Agent Info Note */}
        <Card className="card-surface border-ai-accent/20">
          <CardContent className="py-4">
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-ai-accent/10 flex items-center justify-center flex-shrink-0">
                <Sparkles className="w-4 h-4 text-ai-accent" />
              </div>
              <div>
                <p className="text-sm font-medium text-text-primary">Multi-Agent AI System</p>
                <p className="text-xs text-text-secondary mt-1">
                  NeuralTrader uses a streamlined multi-agent pipeline: Data Collection gathers
                  market data, Technical Analysis computes indicators and patterns, Deep Reasoning
                  synthesizes insights with advanced LLM capabilities, and Validation cross-checks
                  the final recommendation for consistency and risk assessment.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </motion.div>
    </div>
  );
}
