import { useState } from "react";
import { motion } from "framer-motion";
import {
  HelpCircle,
  Search,
  BarChart3,
  Brain,
  Settings,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  ChevronRight,
  Zap,
  Play,
  Activity,
  BookOpen,
  Lightbulb,
  CheckCircle2,
  ArrowRight,
  Bell,
  Key,
  Info
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
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
      title: "Dashboard",
      color: "text-primary",
      description: "Search and analyze any NSE/BSE stock with AI-powered insights",
      steps: [
        "Type a stock symbol (like RELIANCE, TCS, INFY) in the search box",
        "Click on the stock from the dropdown suggestions",
        "View quick overview: price, change %, volume, 52-week range",
        "Click 'Analyze' to trigger AI analysis with candle pattern detection",
        "Review technical indicators, AI reasoning, and BUY/SELL/HOLD recommendation"
      ],
      tips: [
        "Use NSE stock symbols without any suffix (e.g., WIPRO not WIPRO.NS)",
        "AI analysis detects candlestick patterns like Doji, Hammer, Engulfing, etc.",
        "The ensemble system queries multiple AI models in parallel for higher accuracy",
        "Confidence above 70% indicates strong multi-model agreement"
      ]
    },
    {
      id: "ai-picks",
      icon: <Zap className="w-5 h-5" />,
      title: "AI Picks",
      color: "text-yellow-500",
      description: "Batch AI recommendations powered by a LangGraph multi-agent pipeline",
      steps: [
        "Go to 'AI Picks' from the navigation",
        "Click 'Generate New' to analyze all NSE/BSE stocks",
        "Wait 2-5 minutes for the LangGraph agent pipeline to complete",
        "View Buy recommendations (bullish) sorted by confidence score",
        "View Sell recommendations (bearish) sorted by confidence score",
        "Filter by sector to focus on specific industries"
      ],
      sections: [
        { name: "LangGraph Pipeline", desc: "Data Collection, Technical Analysis, Deep Reasoning, and Validation agents" },
        { name: "Confidence Score", desc: "Weighted voting from multiple AI models -- 65% minimum threshold" },
        { name: "BUY Signals", desc: "Bullish stocks with entry price, target, and stop-loss" },
        { name: "SELL Signals", desc: "Bearish stocks with exit reasoning and risk analysis" }
      ],
      tips: [
        "Analysis includes sentiment analysis + technical validation",
        "65% minimum confidence threshold filters out weak signals",
        "Recommendations are cached and auto-updated within 1 hour",
        "Generate once before market open (9:00 AM IST) for best results"
      ]
    },
    {
      id: "alerts",
      icon: <Bell className="w-5 h-5" />,
      title: "Alerts",
      color: "text-orange-500",
      description: "Set up price alerts and get notified via Telegram or Email",
      steps: [
        "Go to 'Alerts' from the navigation",
        "Click 'Create Alert' and enter stock symbol",
        "Set condition: Above, Below, or % Change",
        "Enter target price or percentage threshold",
        "Choose notification channel (Telegram or Email)",
        "Save alert -- it monitors the stock continuously"
      ],
      notifications: [
        { name: "Telegram", desc: "Instant messages via Telegram Bot -- recommended for real-time alerts" },
        { name: "Email (SMTP)", desc: "Email alerts to your inbox via Gmail or any SMTP server" }
      ],
      tips: [
        "Configure Telegram Bot Token and Chat ID in Settings first",
        "For Email alerts, set up SMTP credentials (Gmail App Password works well)",
        "Alerts check prices continuously during market hours",
        "You can create multiple alerts for the same stock at different price levels"
      ]
    },
    {
      id: "settings",
      icon: <Settings className="w-5 h-5" />,
      title: "Settings",
      color: "text-text-secondary",
      description: "Configure LLM providers, API keys, and notification channels",
      sections: [
        { name: "OpenAI", desc: "GPT-4.1, GPT-4o, o3-mini -- get key from platform.openai.com" },
        { name: "Google Gemini", desc: "Gemini 2.5 Flash/Pro -- get key from aistudio.google.com" },
        { name: "Anthropic Claude", desc: "Claude models for ensemble analysis -- get key from console.anthropic.com" },
        { name: "Notifications", desc: "Telegram Bot Token + Chat ID, SMTP Email configuration" }
      ],
      tips: [
        "You only need one AI API key to get started (OpenAI or Gemini)",
        "Adding multiple AI keys enables ensemble analysis for higher accuracy",
        "API keys are stored securely and masked in the UI",
        "Settings persist across browser sessions and server restarts"
      ]
    },
    {
      id: "about",
      icon: <Info className="w-5 h-5" />,
      title: "About NeuralTrader",
      color: "text-ai-accent",
      description: "AI-powered stock analysis platform for Indian markets (NSE/BSE)",
      sections: [
        { name: "What It Is", desc: "An open-source AI trading assistant that uses multiple LLM models to analyze stocks" },
        { name: "Data Sources", desc: "TVScreener (free real-time data) + yfinance (historical data) -- no paid data keys required" },
        { name: "AI Engine", desc: "LangGraph multi-agent pipeline with ensemble voting across OpenAI, Gemini, and Claude" },
        { name: "Tech Stack", desc: "FastAPI + MongoDB backend, React + Vite frontend" }
      ],
      tips: [
        "Stock data comes from TVScreener (free) -- no API key needed for basic functionality",
        "yfinance provides historical price data for charts and technical indicators",
        "The app works with just one AI API key, but more keys = better ensemble accuracy",
        "All analysis is for educational purposes only -- not financial advice"
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

  const faqs = [
    {
      question: "What stocks can I analyze on the Dashboard?",
      answer: "You can analyze NSE and BSE stocks. The list includes NIFTY 100 plus additional mid-cap stocks across all sectors. Just type the symbol without any suffix -- for example, RELIANCE, TCS, INFY, HDFCBANK. The Dashboard provides instant price data, charts, and AI-powered analysis."
    },
    {
      question: "How does the AI analysis work on the Dashboard?",
      answer: "When you click 'Analyze', the system runs a multi-model ensemble. It queries up to 3 AI models (OpenAI GPT-4, Google Gemini, Anthropic Claude) in parallel. Each model analyzes technical indicators, candlestick patterns, and market context. The final recommendation is determined by weighted voting -- models that agree get higher weight. This reduces individual model bias."
    },
    {
      question: "What is the confidence percentage in AI Picks?",
      answer: "Confidence shows how strongly the AI models agree on the direction. 80%+ = Very strong signal (all models agree). 65-80% = Strong signal. 50-65% = Moderate signal. Below 50% = Weak/mixed signals. The minimum threshold for AI Picks is 65%, so only high-confidence recommendations are shown."
    },
    {
      question: "How do I set up Telegram alerts?",
      answer: "Go to Settings and enter your Telegram Bot Token and Chat ID. To get these: 1) Create a bot with @BotFather on Telegram using /newbot command. 2) Get your Chat ID by messaging @userinfobot. 3) Start a conversation with your new bot. Then go to Alerts, create an alert for any stock, and select Telegram as the notification channel."
    },
    {
      question: "Do I need multiple API keys?",
      answer: "No, you only need one AI API key (OpenAI or Gemini) to get started. However, adding multiple AI keys (OpenAI + Gemini + Claude) enables ensemble analysis, which provides more accurate recommendations through multi-model voting. Stock data comes from TVScreener and yfinance for free -- no data provider key is required."
    },
    {
      question: "How often should I regenerate AI Picks?",
      answer: "AI Picks are cached for 1 hour. For best results, generate once in the morning before market open (9:00 AM IST), and optionally after market close (3:30 PM IST) for next-day planning. Regenerating within the cache window updates the existing analysis with fresh data."
    },
    {
      question: "Where does the stock data come from?",
      answer: "NeuralTrader uses TVScreener for free real-time market data (no API key required) and yfinance for historical price data, charts, and technical indicator calculations. This means you can use the app without any paid data subscriptions. The data covers NSE and BSE listed stocks."
    },
    {
      question: "Is this financial advice?",
      answer: "No. NeuralTrader provides technical analysis and AI-generated insights for educational and informational purposes only. It is NOT financial advice. Stock market investments are subject to market risks. Past performance does not guarantee future results. Always conduct your own research and consult with a qualified financial advisor before making any investment decisions."
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
          <Badge variant="outline" className="text-xs">NSE/BSE Stocks</Badge>
          <Badge variant="outline" className="text-xs">AI Analysis</Badge>
          <Badge variant="outline" className="text-xs">Price Alerts</Badge>
        </div>
      </div>

      {/* Tabs for different sections */}
      <Tabs defaultValue="features" className="mb-8">
        <TabsList className="grid w-full grid-cols-3 mb-6">
          <TabsTrigger value="features">Features</TabsTrigger>
          <TabsTrigger value="indicators">Indicators</TabsTrigger>
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
                  <h3 className="font-medium text-text-primary mb-2">Explore AI Picks</h3>
                  <p className="text-sm text-text-secondary">
                    Try AI Picks to scan all NSE/BSE stocks for BUY/SELL opportunities at once
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
        <p className="mt-1">Data: TVScreener + yfinance (free) | AI: OpenAI, Gemini, Claude | NSE/BSE Stocks</p>
      </div>
    </div>
  );
}
