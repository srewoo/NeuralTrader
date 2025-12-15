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
  Sparkles
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
    selected_model: "gpt-4.1",
    selected_provider: "openai",
  });
  const [showOpenAIKey, setShowOpenAIKey] = useState(false);
  const [showGeminiKey, setShowGeminiKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      // Try to fetch from backend first
      const response = await axios.get(`${API_URL}/settings`);
      const backendSettings = mergeWithDefaults(response.data);
      setSettings(backendSettings);
      // Also save to localStorage as backup
      saveStoredSettings(backendSettings);
    } catch (error) {
      console.error("Error fetching settings:", error);
      // Fallback to localStorage if backend is unavailable
      const cachedSettings = getStoredSettings();
      if (cachedSettings) {
        setSettings(mergeWithDefaults(cachedSettings));
        toast.info("Loaded cached settings (backend unavailable)");
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
