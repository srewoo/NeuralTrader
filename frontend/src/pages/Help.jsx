import { useState } from "react";
import { motion } from "framer-motion";
import {
  HelpCircle,
  Search,
  BarChart3,
  Brain,
  History,
  Settings,
  TrendingUp,
  TrendingDown,
  Target,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Zap,
  Play,
  RefreshCw,
  Activity,
  PieChart,
  LineChart,
  BookOpen,
  Lightbulb,
  CheckCircle2,
  ArrowRight,
  Bell,
  Wallet,
  Filter,
  Database,
  Key,
  Globe,
  Server,
  Clock,
  Shield
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

export default function Help() {
  const [expandedSection, setExpandedSection] = useState(null);

  const features = [
    {
      id: "dashboard",
      icon: <Search className="w-5 h-5" />,
      title: "Dashboard - Stock Search",
      color: "text-primary",
      description: "Find and analyze any stock from NSE/BSE with real-time data",
      steps: [
        "Type a stock symbol (like RELIANCE, TCS, INFY) in the search box",
        "Click on the stock from the dropdown suggestions",
        "View quick overview: price, change %, volume, P/E ratio, 52-week range",
        "Click 'Analyze' to get detailed AI-powered analysis"
      ],
      tips: [
        "Use NSE stock symbols without any suffix (e.g., WIPRO not WIPRO.NS)",
        "Data refreshes automatically every 30 seconds",
        "206 NSE stocks are available for analysis"
      ]
    },
    {
      id: "ai-analysis",
      icon: <Brain className="w-5 h-5" />,
      title: "AI Ensemble Analysis",
      color: "text-ai-accent",
      description: "Multi-model AI analysis using GPT-4, Gemini, and Claude for high-confidence recommendations",
      steps: [
        "Search for a stock on the Dashboard or Stock Detail page",
        "Click the 'Analyze' or 'Regenerate Analysis' button",
        "AI analyzes using multiple models in parallel (10-30 seconds)",
        "View the ensemble recommendation: BUY, SELL, or HOLD",
        "Check the confidence score (weighted voting from multiple models)",
        "Review detailed reasoning, entry/exit prices, and risk management"
      ],
      tips: [
        "Ensemble uses up to 3 AI models: OpenAI GPT-4, Google Gemini, Anthropic Claude",
        "Confidence above 70% indicates strong multi-model agreement",
        "Entry price, target price, and stop-loss are calculated using ATR-based risk management",
        "Minimum risk-reward ratio of 1.5x is enforced"
      ]
    },
    {
      id: "ai-recommends",
      icon: <Zap className="w-5 h-5" />,
      title: "AI Picks - Batch Analysis",
      color: "text-yellow-500",
      description: "Scan 200+ NSE stocks at once for BUY/SELL opportunities",
      steps: [
        "Go to 'AI Picks' from the navigation",
        "Click 'Generate New' to analyze all 206 stocks",
        "Wait 2-5 minutes for enhanced analysis to complete",
        "View Buy recommendations (bullish) sorted by confidence",
        "View Sell recommendations (bearish) sorted by confidence",
        "Filter by sector to focus on specific industries",
        "Click any stock card to see detailed analysis"
      ],
      tips: [
        "Enhanced mode (default): includes sentiment analysis + backtest validation",
        "65% minimum confidence threshold filters out weak signals",
        "Recommendations are cached and auto-updated within 1 hour",
        "Top 20 BUY and top 15 SELL signals are displayed"
      ]
    },
    {
      id: "stock-detail",
      icon: <LineChart className="w-5 h-5" />,
      title: "Stock Detail Page",
      color: "text-success",
      description: "Deep dive into any stock with charts, indicators, and AI analysis",
      sections: [
        { name: "Price Chart", desc: "6-month interactive candlestick chart with volume" },
        { name: "Key Metrics", desc: "Volume, Market Cap, P/E Ratio, 52-Week Range" },
        { name: "Technical Indicators", desc: "RSI, MACD, SMA 20/50/200, Bollinger Bands, ADX, Stochastic" },
        { name: "AI Analysis", desc: "Ensemble recommendation with entry/target/stop-loss" },
        { name: "Regenerate Button", desc: "Refresh AI analysis with latest data" }
      ],
      tips: [
        "Hover over chart candles to see OHLC values",
        "Green indicators = Bullish, Red = Bearish, Amber = Neutral",
        "Auto-refresh every 30 seconds for real-time data"
      ]
    },
    {
      id: "paper-trading",
      icon: <Wallet className="w-5 h-5" />,
      title: "Paper Trading",
      color: "text-emerald-500",
      description: "Practice trading with virtual money - no risk involved",
      steps: [
        "Go to 'Paper Trading' from the navigation",
        "Start with ₹10,00,000 virtual capital",
        "Search for a stock and enter quantity",
        "Click 'Buy' or 'Sell' to execute paper trades",
        "Track your positions, P&L, and portfolio value",
        "Use 'Reset Portfolio' to start fresh"
      ],
      tips: [
        "Trades are persisted in database - survives server restarts",
        "Commission (0.1%) and slippage (5 bps) are simulated",
        "Maximum position size: 20% of portfolio",
        "Great for testing strategies before using real money"
      ]
    },
    {
      id: "alerts",
      icon: <Bell className="w-5 h-5" />,
      title: "Price Alerts",
      color: "text-orange-500",
      description: "Get notified when stocks hit your target prices",
      steps: [
        "Go to 'Alerts' from the navigation",
        "Click 'Create Alert' and enter stock symbol",
        "Set condition: Above, Below, or % Change",
        "Enter target price or percentage",
        "Choose notification channel (Telegram, Email, Webhook)",
        "Save alert - it will monitor the stock continuously"
      ],
      notifications: [
        { name: "Telegram", desc: "Instant messages via Telegram Bot" },
        { name: "Email (SMTP)", desc: "Email alerts to your inbox" },
        { name: "Webhook", desc: "HTTP POST to your custom endpoint" },
        { name: "Slack", desc: "Messages to Slack channel" }
      ],
      tips: [
        "Alerts are persisted in SQLite database",
        "Configure notification channels in Settings",
        "Supports price alerts, pattern alerts, and portfolio alerts"
      ]
    },
    {
      id: "screener",
      icon: <Filter className="w-5 h-5" />,
      title: "Stock Screener",
      color: "text-cyan-500",
      description: "Filter stocks by fundamental and technical criteria",
      steps: [
        "Go to 'Screener' from the navigation",
        "Set filters: P/E ratio, Market Cap, ROE, etc.",
        "Use preset screens: Value Stocks, Growth Stocks, Dividend Yield",
        "Click 'Screen' to find matching stocks",
        "Sort results by any column",
        "Click a stock to see detailed analysis"
      ],
      tips: [
        "Screens all 206 NSE stocks",
        "Combine multiple filters for precise results",
        "Data includes P/E, ROE, Debt/Equity, Market Cap, and more"
      ]
    },
    {
      id: "backtesting",
      icon: <BarChart3 className="w-5 h-5" />,
      title: "Backtesting",
      color: "text-blue-500",
      description: "Test trading strategies on historical data",
      steps: [
        "Go to 'Backtesting' from the navigation",
        "Enter a stock symbol",
        "Select a trading strategy",
        "Choose date range (up to 5 years)",
        "Set initial capital and position sizing",
        "Click 'Run Backtest' and review results"
      ],
      strategies: [
        { name: "Momentum", desc: "Buy when RSI + MACD show bullish momentum" },
        { name: "Mean Reversion", desc: "Buy oversold, sell overbought (Bollinger + RSI)" },
        { name: "Trend Following", desc: "Follow SMA crossovers and ADX trend strength" },
        { name: "Breakout", desc: "Trade when price breaks support/resistance levels" },
        { name: "MACD Crossover", desc: "Buy/sell on MACD signal line crossovers" }
      ],
      tips: [
        "Results cached in SQLite for fast re-runs",
        "Walk-forward testing available for realistic results",
        "Strategy optimization finds best parameters automatically"
      ]
    },
    {
      id: "history",
      icon: <History className="w-5 h-5" />,
      title: "Analysis History",
      color: "text-purple-500",
      description: "Review all your past stock analyses",
      steps: [
        "Go to 'History' from the navigation",
        "See all previous analyses sorted by date",
        "Filter by stock symbol if needed",
        "Click any entry to see the full analysis",
        "Compare how recommendations performed"
      ],
      tips: [
        "History is stored in MongoDB for 30 days (auto-cleanup)",
        "Track which recommendations were accurate",
        "Learn from past analyses to improve decisions"
      ]
    },
    {
      id: "settings",
      icon: <Settings className="w-5 h-5" />,
      title: "Settings & API Keys",
      color: "text-text-secondary",
      description: "Configure AI providers, data sources, and notifications",
      sections: [
        { name: "AI Providers", desc: "OpenAI, Google Gemini, Anthropic Claude API keys" },
        { name: "Data Providers", desc: "Finnhub, Alpaca, FMP, Alpha Vantage, IEX Cloud" },
        { name: "Notifications", desc: "Telegram Bot, SMTP Email, Slack Webhook, Twilio WhatsApp" },
        { name: "Model Selection", desc: "Choose default AI model and provider" }
      ],
      tips: [
        "Get OpenAI key: platform.openai.com/api-keys",
        "Get Gemini key: aistudio.google.com/apikey",
        "Get Claude key: console.anthropic.com",
        "API keys are stored securely in MongoDB (masked in UI)",
        "Settings persist across browser sessions and server restarts"
      ]
    }
  ];

  const apiEndpoints = [
    {
      category: "Stock Data",
      endpoints: [
        { method: "GET", path: "/api/stocks/{symbol}", desc: "Get real-time stock quote" },
        { method: "GET", path: "/api/stocks/{symbol}/history", desc: "Get historical price data" },
        { method: "GET", path: "/api/stocks/{symbol}/indicators", desc: "Get technical indicators" },
        { method: "GET", path: "/api/stocks/search?q={query}", desc: "Search stocks by name/symbol" }
      ]
    },
    {
      category: "AI Analysis",
      endpoints: [
        { method: "POST", path: "/api/analyze/ensemble", desc: "Run multi-model AI analysis" },
        { method: "POST", path: "/api/recommendations/generate", desc: "Generate AI picks for 200 stocks" },
        { method: "GET", path: "/api/recommendations", desc: "Get cached recommendations" }
      ]
    },
    {
      category: "Paper Trading",
      endpoints: [
        { method: "GET", path: "/api/paper-trading/portfolio", desc: "Get portfolio status" },
        { method: "POST", path: "/api/paper-trading/order", desc: "Place buy/sell order" },
        { method: "POST", path: "/api/paper-trading/reset", desc: "Reset portfolio" }
      ]
    },
    {
      category: "Alerts",
      endpoints: [
        { method: "GET", path: "/api/alerts", desc: "List all alerts" },
        { method: "POST", path: "/api/alerts", desc: "Create new alert" },
        { method: "DELETE", path: "/api/alerts/{id}", desc: "Delete alert" }
      ]
    },
    {
      category: "Database Admin",
      endpoints: [
        { method: "GET", path: "/api/admin/db/stats", desc: "View database statistics" },
        { method: "DELETE", path: "/api/admin/db/cleanup/all", desc: "Clean up old data" },
        { method: "POST", path: "/api/admin/db/reset/{collection}", desc: "Reset a collection" }
      ]
    }
  ];

  const indicators = [
    {
      name: "RSI (Relative Strength Index)",
      range: "0-100",
      interpretation: [
        "Below 30 = Oversold (potential buy)",
        "Above 70 = Overbought (potential sell)",
        "Around 50 = Neutral"
      ]
    },
    {
      name: "MACD",
      interpretation: [
        "MACD above Signal = Bullish",
        "MACD below Signal = Bearish",
        "Histogram growing = Momentum increasing"
      ]
    },
    {
      name: "Moving Averages (SMA)",
      interpretation: [
        "Price above SMA = Uptrend",
        "Price below SMA = Downtrend",
        "Golden Cross (20 above 50) = Strong buy",
        "Death Cross (20 below 50) = Strong sell"
      ]
    },
    {
      name: "Bollinger Bands",
      interpretation: [
        "Price near lower band = Oversold",
        "Price near upper band = Overbought",
        "Bands squeezing = Big move coming"
      ]
    },
    {
      name: "ADX (Trend Strength)",
      range: "0-100",
      interpretation: [
        "Below 20 = No trend (sideways)",
        "20-40 = Developing trend",
        "Above 40 = Strong trend"
      ]
    },
    {
      name: "Stochastic Oscillator",
      range: "0-100",
      interpretation: [
        "Below 20 = Oversold",
        "Above 80 = Overbought",
        "%K crossing %D = Signal"
      ]
    }
  ];

  const dataRetention = [
    { collection: "Settings", retention: "Permanent", desc: "API keys and preferences" },
    { collection: "Recommendations", retention: "7 days", desc: "AI picks results (TTL index)" },
    { collection: "Analysis History", retention: "30 days", desc: "Individual stock analyses" },
    { collection: "Backtests", retention: "90 days", desc: "Backtest results" },
    { collection: "Alerts", retention: "Permanent", desc: "SQLite database" },
    { collection: "Paper Trading", retention: "Permanent", desc: "SQLite database" },
    { collection: "Price Cache", retention: "24 hours", desc: "SQLite for backtest speed" }
  ];

  const faqs = [
    {
      question: "What stocks can I analyze?",
      answer: "You can analyze 206 stocks from NSE (National Stock Exchange). The list includes NIFTY 100 plus additional mid-cap stocks across all sectors. Just type the symbol without any suffix - for example, RELIANCE, TCS, INFY, HDFCBANK."
    },
    {
      question: "How does ensemble AI analysis work?",
      answer: "The ensemble system queries up to 3 AI models (OpenAI GPT-4, Google Gemini, Anthropic Claude) in parallel. Each model provides a recommendation with confidence score. The final recommendation is determined by weighted voting - models that agree get higher weight. This reduces bias and increases reliability."
    },
    {
      question: "What does the confidence percentage mean?",
      answer: "Confidence shows how strongly the AI models agree on the direction. 80%+ = Very strong signal (all models agree). 65-80% = Strong signal. 50-65% = Moderate signal. Below 50% = Weak/mixed signals. The minimum threshold is 65% for AI Picks."
    },
    {
      question: "Why do I need multiple API keys?",
      answer: "Different API keys serve different purposes: AI keys (OpenAI, Gemini, Claude) power the analysis. Data provider keys (Finnhub, Alpaca, FMP) provide real-time market data. Notification keys (Telegram, SMTP) enable alerts. You can use the app with just one AI key - others are optional."
    },
    {
      question: "How is my data stored?",
      answer: "Settings and analysis data are stored in MongoDB with automatic TTL (Time-To-Live) cleanup. Alerts and paper trading use SQLite for faster local access. API keys are stored encrypted and masked in the UI. Data is retained based on type: recommendations (7 days), analysis (30 days), backtests (90 days)."
    },
    {
      question: "What is paper trading?",
      answer: "Paper trading lets you practice trading with virtual money (₹10 lakh starting capital). Trades are simulated with realistic commission (0.1%) and slippage. Your portfolio, positions, and trade history persist across sessions. Use it to test strategies without risking real money."
    },
    {
      question: "How do price alerts work?",
      answer: "Create alerts for specific price levels or percentage changes. When triggered, you receive notifications via your configured channels (Telegram, Email, Slack, Webhook). Alerts run in the background and check prices continuously during market hours."
    },
    {
      question: "Is this financial advice?",
      answer: "No. This tool provides technical analysis and AI-generated insights for educational purposes only. It is NOT financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results."
    },
    {
      question: "How often should I regenerate recommendations?",
      answer: "AI Picks are cached for 1 hour - regenerating within this window updates the existing recommendation. For best results, generate once in the morning (9:00 AM IST) before market open, and optionally after market close (3:30 PM IST) for next-day planning."
    },
    {
      question: "What are the system requirements?",
      answer: "Backend: Python 3.9+, MongoDB, Node.js 18+. The app uses yfinance for free stock data (no key required), with optional premium data providers for real-time quotes. Redis is optional (falls back to in-memory cache)."
    }
  ];

  return (
    <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {/* Header */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center">
            <HelpCircle className="w-6 h-6 text-primary" />
          </div>
          <div>
            <h1 className="text-3xl font-heading font-bold text-text-primary">
              Help & Documentation
            </h1>
            <p className="text-text-secondary">
              Complete guide to NeuralTrader - AI-powered stock analysis platform
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 mt-4">
          <Badge variant="outline" className="text-xs">206 NSE Stocks</Badge>
          <Badge variant="outline" className="text-xs">3 AI Models</Badge>
          <Badge variant="outline" className="text-xs">Real-time Data</Badge>
          <Badge variant="outline" className="text-xs">Paper Trading</Badge>
          <Badge variant="outline" className="text-xs">Price Alerts</Badge>
          <Badge variant="outline" className="text-xs">Backtesting</Badge>
        </div>
      </div>

      {/* Tabs for different sections */}
      <Tabs defaultValue="features" className="mb-8">
        <TabsList className="grid w-full grid-cols-5 mb-6">
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="indicators">Indicators</TabsTrigger>
          <TabsTrigger value="api">API Reference</TabsTrigger>
          <TabsTrigger value="data">Data & Storage</TabsTrigger>
          <TabsTrigger value="faq">FAQ</TabsTrigger>
        </TabsList>

        {/* Features Tab */}
        <TabsContent value="features">
          {/* Quick Start */}
          <Card className="card-surface mb-8 border-primary/30">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Play className="w-5 h-5 text-primary" />
                <CardTitle>Quick Start Guide</CardTitle>
              </div>
              <CardDescription>Get started in 3 simple steps</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.1 }}
                  className="flex flex-col items-center text-center p-6 rounded-lg bg-surface-highlight"
                >
                  <div className="w-12 h-12 rounded-full bg-primary/20 flex items-center justify-center mb-4">
                    <span className="text-xl font-bold text-primary">1</span>
                  </div>
                  <h3 className="font-medium text-text-primary mb-2">Configure API Keys</h3>
                  <p className="text-sm text-text-secondary">
                    Go to Settings and add at least one AI API key (OpenAI, Gemini, or Claude)
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.2 }}
                  className="flex flex-col items-center text-center p-6 rounded-lg bg-surface-highlight"
                >
                  <div className="w-12 h-12 rounded-full bg-ai-accent/20 flex items-center justify-center mb-4">
                    <span className="text-xl font-bold text-ai-accent">2</span>
                  </div>
                  <h3 className="font-medium text-text-primary mb-2">Search & Analyze</h3>
                  <p className="text-sm text-text-secondary">
                    Search any stock on Dashboard and click Analyze for AI recommendations
                  </p>
                </motion.div>

                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.3 }}
                  className="flex flex-col items-center text-center p-6 rounded-lg bg-surface-highlight"
                >
                  <div className="w-12 h-12 rounded-full bg-success/20 flex items-center justify-center mb-4">
                    <span className="text-xl font-bold text-success">3</span>
                  </div>
                  <h3 className="font-medium text-text-primary mb-2">Explore Features</h3>
                  <p className="text-sm text-text-secondary">
                    Try AI Picks, Paper Trading, Alerts, Screener, and Backtesting
                  </p>
                </motion.div>
              </div>
            </CardContent>
          </Card>

          {/* Features Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={feature.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.03 }}
              >
                <Card className="card-surface h-full">
                  <CardHeader>
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg bg-surface-highlight flex items-center justify-center ${feature.color}`}>
                        {feature.icon}
                      </div>
                      <div>
                        <CardTitle className="text-lg">{feature.title}</CardTitle>
                        <CardDescription>{feature.description}</CardDescription>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Steps */}
                    {feature.steps && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-text-primary mb-3 flex items-center gap-2">
                          <CheckCircle2 className="w-4 h-4 text-success" />
                          How to use:
                        </h4>
                        <ol className="space-y-2">
                          {feature.steps.map((step, idx) => (
                            <li key={idx} className="flex items-start gap-2 text-sm text-text-secondary">
                              <span className="flex-shrink-0 w-5 h-5 rounded-full bg-surface-highlight text-xs flex items-center justify-center text-text-secondary">
                                {idx + 1}
                              </span>
                              {step}
                            </li>
                          ))}
                        </ol>
                      </div>
                    )}

                    {/* Sections */}
                    {feature.sections && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-text-primary mb-3">What's included:</h4>
                        <div className="space-y-2">
                          {feature.sections.map((section, idx) => (
                            <div key={idx} className="flex items-center gap-2 text-sm">
                              <ArrowRight className="w-4 h-4 text-primary flex-shrink-0" />
                              <span className="text-text-primary">{section.name}:</span>
                              <span className="text-text-secondary">{section.desc}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Strategies */}
                    {feature.strategies && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-text-primary mb-3">Available Strategies:</h4>
                        <div className="space-y-2">
                          {feature.strategies.map((strategy, idx) => (
                            <div key={idx} className="p-2 rounded bg-surface-highlight">
                              <span className="text-sm text-text-primary font-medium">{strategy.name}: </span>
                              <span className="text-sm text-text-secondary">{strategy.desc}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Notifications */}
                    {feature.notifications && (
                      <div className="mb-4">
                        <h4 className="text-sm font-medium text-text-primary mb-3">Notification Channels:</h4>
                        <div className="grid grid-cols-2 gap-2">
                          {feature.notifications.map((notif, idx) => (
                            <div key={idx} className="p-2 rounded bg-surface-highlight">
                              <span className="text-sm text-text-primary font-medium">{notif.name}</span>
                              <p className="text-xs text-text-secondary">{notif.desc}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Tips */}
                    {feature.tips && (
                      <div className="pt-3 border-t border-[#1F1F1F]">
                        <h4 className="text-xs font-medium text-yellow-500 mb-2 flex items-center gap-1">
                          <Lightbulb className="w-3 h-3" />
                          Pro Tips:
                        </h4>
                        <ul className="space-y-1">
                          {feature.tips.map((tip, idx) => (
                            <li key={idx} className="text-xs text-text-secondary flex items-start gap-1">
                              <span className="text-yellow-500">•</span>
                              {tip}
                            </li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </TabsContent>

        {/* Indicators Tab */}
        <TabsContent value="indicators">
          <Card className="card-surface">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Activity className="w-5 h-5 text-ai-accent" />
                <CardTitle>Technical Indicators Guide</CardTitle>
              </div>
              <CardDescription>Understanding what each indicator means and how to interpret signals</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {indicators.map((indicator, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.1 }}
                    className="p-4 rounded-lg bg-surface-highlight"
                  >
                    <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                      <BarChart3 className="w-4 h-4 text-ai-accent" />
                      {indicator.name}
                    </h4>
                    {indicator.range && (
                      <Badge variant="outline" className="mb-2 text-xs">
                        Range: {indicator.range}
                      </Badge>
                    )}
                    <ul className="space-y-1">
                      {indicator.interpretation.map((item, i) => (
                        <li key={i} className="text-xs text-text-secondary flex items-start gap-1">
                          <ChevronRight className="w-3 h-3 mt-0.5 text-text-secondary flex-shrink-0" />
                          {item}
                        </li>
                      ))}
                    </ul>
                  </motion.div>
                ))}
              </div>

              {/* Color Legend */}
              <div className="mt-6 p-4 rounded-lg bg-surface-highlight">
                <h4 className="font-medium text-text-primary mb-3">Signal Colors</h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-success"></div>
                    <span className="text-sm text-text-secondary">Bullish / Buy Signal</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-danger"></div>
                    <span className="text-sm text-text-secondary">Bearish / Sell Signal</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-amber-500"></div>
                    <span className="text-sm text-text-secondary">Neutral / Caution</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-4 h-4 rounded bg-ai-accent"></div>
                    <span className="text-sm text-text-secondary">AI-powered Feature</span>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* API Reference Tab */}
        <TabsContent value="api">
          <Card className="card-surface">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Server className="w-5 h-5 text-primary" />
                <CardTitle>API Reference</CardTitle>
              </div>
              <CardDescription>Backend API endpoints for developers</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                {apiEndpoints.map((category, idx) => (
                  <div key={idx}>
                    <h4 className="font-medium text-text-primary mb-3 flex items-center gap-2">
                      <Globe className="w-4 h-4 text-primary" />
                      {category.category}
                    </h4>
                    <div className="space-y-2">
                      {category.endpoints.map((endpoint, i) => (
                        <div key={i} className="flex items-center gap-3 p-3 rounded bg-surface-highlight">
                          <Badge
                            className={`text-xs font-mono ${
                              endpoint.method === 'GET' ? 'bg-green-500/20 text-green-400' :
                              endpoint.method === 'POST' ? 'bg-blue-500/20 text-blue-400' :
                              'bg-red-500/20 text-red-400'
                            }`}
                          >
                            {endpoint.method}
                          </Badge>
                          <code className="text-sm text-text-primary font-mono">{endpoint.path}</code>
                          <span className="text-sm text-text-secondary ml-auto">{endpoint.desc}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>

              <div className="mt-6 p-4 rounded-lg bg-surface-highlight border border-primary/30">
                <h4 className="font-medium text-text-primary mb-2 flex items-center gap-2">
                  <Key className="w-4 h-4 text-primary" />
                  API Base URL
                </h4>
                <code className="text-sm text-primary font-mono">http://localhost:8005/api</code>
                <p className="text-xs text-text-secondary mt-2">
                  All endpoints are prefixed with /api. Example: GET http://localhost:8005/api/stocks/RELIANCE
                </p>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Data & Storage Tab */}
        <TabsContent value="data">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card className="card-surface">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Database className="w-5 h-5 text-primary" />
                  <CardTitle>Data Storage</CardTitle>
                </div>
                <CardDescription>Where your data is stored and retention policies</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {dataRetention.map((item, idx) => (
                    <div key={idx} className="flex items-center justify-between p-3 rounded bg-surface-highlight">
                      <div>
                        <span className="text-sm text-text-primary font-medium">{item.collection}</span>
                        <p className="text-xs text-text-secondary">{item.desc}</p>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        <Clock className="w-3 h-3 mr-1" />
                        {item.retention}
                      </Badge>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="card-surface">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <Shield className="w-5 h-5 text-success" />
                  <CardTitle>Security & Privacy</CardTitle>
                </div>
                <CardDescription>How your data and API keys are protected</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="p-3 rounded bg-surface-highlight">
                    <h4 className="text-sm font-medium text-text-primary mb-1">API Key Storage</h4>
                    <p className="text-xs text-text-secondary">
                      API keys are stored in MongoDB and masked in the UI (only first 8 and last 4 characters shown).
                      Keys are never logged or exposed in API responses.
                    </p>
                  </div>
                  <div className="p-3 rounded bg-surface-highlight">
                    <h4 className="text-sm font-medium text-text-primary mb-1">Local Storage</h4>
                    <p className="text-xs text-text-secondary">
                      Settings are cached in browser localStorage as backup. Sensitive data like full API keys
                      are only stored server-side.
                    </p>
                  </div>
                  <div className="p-3 rounded bg-surface-highlight">
                    <h4 className="text-sm font-medium text-text-primary mb-1">Auto-Cleanup</h4>
                    <p className="text-xs text-text-secondary">
                      TTL (Time-To-Live) indexes automatically delete old data: recommendations after 7 days,
                      analysis after 30 days, backtests after 90 days.
                    </p>
                  </div>
                  <div className="p-3 rounded bg-surface-highlight">
                    <h4 className="text-sm font-medium text-text-primary mb-1">Manual Cleanup</h4>
                    <p className="text-xs text-text-secondary">
                      Use admin endpoints to manually clean old data or reset collections:
                      DELETE /api/admin/db/cleanup/all
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* FAQ Tab */}
        <TabsContent value="faq">
          <Card className="card-surface">
            <CardHeader>
              <div className="flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-primary" />
                <CardTitle>Frequently Asked Questions</CardTitle>
              </div>
            </CardHeader>
            <CardContent>
              <Accordion type="single" collapsible className="w-full">
                {faqs.map((faq, idx) => (
                  <AccordionItem key={idx} value={`faq-${idx}`} className="border-[#1F1F1F]">
                    <AccordionTrigger className="text-left text-text-primary hover:text-primary">
                      {faq.question}
                    </AccordionTrigger>
                    <AccordionContent className="text-text-secondary">
                      {faq.answer}
                    </AccordionContent>
                  </AccordionItem>
                ))}
              </Accordion>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Disclaimer */}
      <Card className="card-surface mt-6 border-yellow-500/30">
        <CardContent className="py-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-text-secondary">
              <p className="font-medium text-yellow-500 mb-1">Important Disclaimer</p>
              <p>
                NeuralTrader provides technical analysis and AI-generated insights for educational and informational purposes only.
                This is NOT financial advice. Stock market investments are subject to market risks.
                Past performance does not guarantee future results. Always conduct your own research
                and consult with a qualified financial advisor before making any investment decisions.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Version Info */}
      <div className="mt-4 text-center text-xs text-text-secondary">
        <p>NeuralTrader v2.0 | Backend: FastAPI + MongoDB | Frontend: React + Vite</p>
        <p className="mt-1">Supports: OpenAI GPT-4, Google Gemini, Anthropic Claude | 206 NSE Stocks</p>
      </div>
    </div>
  );
}
