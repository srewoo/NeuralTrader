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
  ArrowRight
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

export default function Help() {
  const [expandedSection, setExpandedSection] = useState(null);

  const features = [
    {
      id: "dashboard",
      icon: <Search className="w-5 h-5" />,
      title: "Dashboard - Stock Search",
      color: "text-primary",
      description: "Find and analyze any stock from NSE/BSE",
      steps: [
        "Type a stock symbol (like RELIANCE, TCS, INFY) in the search box",
        "Click on the stock from the dropdown suggestions",
        "View the quick overview with price, change, and basic indicators",
        "Click 'Analyze' to get detailed AI-powered analysis"
      ],
      tips: [
        "Use NSE stock symbols without any suffix",
        "Popular stocks: RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK"
      ]
    },
    {
      id: "ai-analysis",
      icon: <Brain className="w-5 h-5" />,
      title: "AI Stock Analysis",
      color: "text-ai-accent",
      description: "Get AI-powered buy/sell recommendations with confidence scores",
      steps: [
        "Search for a stock on the Dashboard",
        "Click the 'Analyze' button",
        "Wait for AI to process (takes 10-30 seconds)",
        "View the recommendation: BUY, SELL, or HOLD",
        "Check the confidence score (higher is better)",
        "Review the reasoning chain to understand why"
      ],
      tips: [
        "Confidence above 70% indicates a strong signal",
        "Always check the AI reasoning for context",
        "Look at multiple indicators, not just the recommendation"
      ]
    },
    {
      id: "ai-recommends",
      icon: <Zap className="w-5 h-5" />,
      title: "AI Recommendations Page",
      color: "text-yellow-500",
      description: "Scan 100 stocks at once for opportunities",
      steps: [
        "Go to 'AI Recommends' from the navigation",
        "Click 'Generate New' to analyze 100 top stocks",
        "Wait 30-60 seconds for analysis to complete",
        "View Buy recommendations (bullish stocks) on the left",
        "View Sell recommendations (bearish stocks) on the right",
        "Use the sector filter to focus on specific industries",
        "Click any stock card to see detailed analysis"
      ],
      tips: [
        "Recommendations are sorted by confidence",
        "Green cards = Buy signals, Red cards = Sell signals",
        "Filter by sector to find opportunities in your preferred industry"
      ]
    },
    {
      id: "stock-detail",
      icon: <LineChart className="w-5 h-5" />,
      title: "Stock Detail Page",
      color: "text-success",
      description: "Deep dive into any stock's analysis",
      sections: [
        {
          name: "Price Chart",
          desc: "Interactive candlestick chart with volume"
        },
        {
          name: "Technical Indicators",
          desc: "RSI, MACD, Moving Averages, Bollinger Bands"
        },
        {
          name: "AI Reasoning",
          desc: "Step-by-step explanation of the analysis"
        },
        {
          name: "Candlestick Patterns",
          desc: "Detected chart patterns like Doji, Hammer, etc."
        },
        {
          name: "News & Sentiment",
          desc: "Latest news with sentiment analysis"
        }
      ],
      tips: [
        "Hover over chart candles to see OHLC values",
        "Green indicators = Bullish, Red = Bearish",
        "Check pattern significance (higher = more reliable)"
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
        "Select a trading strategy (Momentum, Mean Reversion, etc.)",
        "Choose date range for testing",
        "Click 'Run Backtest'",
        "Review results: returns, win rate, max drawdown"
      ],
      strategies: [
        {
          name: "Momentum",
          desc: "Buy when price is trending up (RSI + MACD signals)"
        },
        {
          name: "Mean Reversion",
          desc: "Buy when oversold, sell when overbought"
        },
        {
          name: "Trend Following",
          desc: "Follow the direction of moving averages"
        },
        {
          name: "Breakout",
          desc: "Trade when price breaks key levels"
        }
      ],
      tips: [
        "Past performance doesn't guarantee future results",
        "Check win rate and max drawdown together",
        "Use walk-forward testing for more realistic results"
      ]
    },
    {
      id: "history",
      icon: <History className="w-5 h-5" />,
      title: "Analysis History",
      color: "text-orange-500",
      description: "Review your past stock analyses",
      steps: [
        "Go to 'History' from the navigation",
        "See all your previous analyses",
        "Filter by date or stock symbol",
        "Click any entry to see the full analysis",
        "Compare how recommendations performed over time"
      ],
      tips: [
        "Track which recommendations were accurate",
        "Learn from past analyses to improve decisions"
      ]
    },
    {
      id: "settings",
      icon: <Settings className="w-5 h-5" />,
      title: "Settings",
      color: "text-text-secondary",
      description: "Configure AI provider and API keys",
      steps: [
        "Go to 'Settings' from the navigation",
        "Choose your AI provider (OpenAI or Google Gemini)",
        "Enter your API key",
        "Save settings",
        "Your analyses will now use the selected AI"
      ],
      providers: [
        {
          name: "OpenAI (GPT-4)",
          desc: "Best accuracy, requires OpenAI API key"
        },
        {
          name: "Google Gemini",
          desc: "Good alternative, requires Google API key"
        }
      ],
      tips: [
        "Get OpenAI key from: platform.openai.com",
        "Get Gemini key from: makersuite.google.com",
        "Keys are stored securely and never shared"
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
    }
  ];

  const faqs = [
    {
      question: "What stocks can I analyze?",
      answer: "You can analyze any stock from NSE (National Stock Exchange) and BSE (Bombay Stock Exchange). Just type the symbol without any suffix - for example, RELIANCE, TCS, INFY, HDFCBANK."
    },
    {
      question: "How accurate are the AI recommendations?",
      answer: "AI recommendations are based on technical analysis and should be used as one of many inputs for your investment decisions. The confidence score indicates how strong the signals are. Higher confidence (70%+) means multiple indicators agree. Always do your own research and consider fundamental factors too."
    },
    {
      question: "What does the confidence percentage mean?",
      answer: "Confidence shows how many technical indicators agree on the direction. 90%+ = Very strong signal (most indicators agree). 70-90% = Strong signal. 50-70% = Moderate signal. Below 50% = Weak/mixed signals."
    },
    {
      question: "Why do I need an API key?",
      answer: "The AI analysis uses either OpenAI GPT-4 or Google Gemini to generate insights. You need to provide your own API key for these services. Get a key from platform.openai.com (OpenAI) or makersuite.google.com (Google)."
    },
    {
      question: "What is backtesting?",
      answer: "Backtesting tests a trading strategy on historical data to see how it would have performed. For example, if you used the 'Momentum' strategy on RELIANCE for the past year, it shows what returns you would have made. This helps evaluate strategy effectiveness before using real money."
    },
    {
      question: "Is this financial advice?",
      answer: "No. This tool provides technical analysis and AI-generated insights for educational purposes only. It is NOT financial advice. Always consult a qualified financial advisor before making investment decisions. Past performance does not guarantee future results."
    },
    {
      question: "How often should I regenerate recommendations?",
      answer: "Markets change daily. We recommend regenerating AI Recommendations at least once a day, preferably before market hours (9:00 AM IST) or after market close (3:30 PM IST) for the most relevant signals."
    },
    {
      question: "What do the colors mean?",
      answer: "Green = Bullish/Positive (buy signals, uptrends, good indicators). Red = Bearish/Negative (sell signals, downtrends, warning indicators). Yellow = Caution/Neutral (mixed signals, need more confirmation). Purple = AI-related features."
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
              Help & User Guide
            </h1>
            <p className="text-text-secondary">
              Learn how to use NeuralTrader effectively
            </p>
          </div>
        </div>
      </div>

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
              <h3 className="font-medium text-text-primary mb-2">Search a Stock</h3>
              <p className="text-sm text-text-secondary">
                Type any NSE/BSE stock symbol in the search box on the Dashboard
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
              <h3 className="font-medium text-text-primary mb-2">Click Analyze</h3>
              <p className="text-sm text-text-secondary">
                Press the Analyze button to get AI-powered recommendations
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
              <h3 className="font-medium text-text-primary mb-2">Review Results</h3>
              <p className="text-sm text-text-secondary">
                Check the recommendation, confidence score, and reasoning
              </p>
            </motion.div>
          </div>
        </CardContent>
      </Card>

      {/* Features */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {features.map((feature, index) => (
          <motion.div
            key={feature.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.05 }}
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

                {/* Sections (for stock detail) */}
                {feature.sections && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-text-primary mb-3">What you'll see:</h4>
                    <div className="space-y-2">
                      {feature.sections.map((section, idx) => (
                        <div key={idx} className="flex items-center gap-2 text-sm">
                          <ArrowRight className="w-4 h-4 text-primary" />
                          <span className="text-text-primary">{section.name}:</span>
                          <span className="text-text-secondary">{section.desc}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Strategies (for backtesting) */}
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

                {/* Providers (for settings) */}
                {feature.providers && (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium text-text-primary mb-3">AI Providers:</h4>
                    <div className="space-y-2">
                      {feature.providers.map((provider, idx) => (
                        <div key={idx} className="p-2 rounded bg-surface-highlight">
                          <span className="text-sm text-text-primary font-medium">{provider.name}: </span>
                          <span className="text-sm text-text-secondary">{provider.desc}</span>
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

      {/* Technical Indicators Guide */}
      <Card className="card-surface mb-8">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-ai-accent" />
            <CardTitle>Understanding Technical Indicators</CardTitle>
          </div>
          <CardDescription>What each indicator means and how to interpret it</CardDescription>
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
                      <ChevronRight className="w-3 h-3 mt-0.5 text-text-secondary" />
                      {item}
                    </li>
                  ))}
                </ul>
              </motion.div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* FAQ */}
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
    </div>
  );
}
