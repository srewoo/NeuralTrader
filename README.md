# 🧠 NeuralTrader - AI-Powered Trading Advisor

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**NeuralTrader** is a sophisticated AI-powered stock trading advisor that combines multi-agent systems, RAG (Retrieval-Augmented Generation), technical analysis, and machine learning to provide intelligent trading recommendations for Indian stock markets (NSE/BSE).

## ✨ Features

### 🤖 Multi-Agent AI System
- **5 Specialized Agents** working in orchestrated workflow:
  - **Data Collection Agent**: Fetches real-time and historical stock data
  - **Technical Analysis Agent**: Calculates 10+ technical indicators
  - **RAG Knowledge Agent**: Retrieves relevant trading knowledge from vector database
  - **Deep Reasoning Agent**: Performs chain-of-thought analysis
  - **Validator Agent**: Self-critiques and validates recommendations

### 📊 Technical Analysis
- **17+ Candlestick Patterns**: Real pattern detection algorithms
- **Technical Indicators**: RSI, MACD, SMA, EMA, Bollinger Bands, ATR, OBV, Stochastic
- **Price Charts**: Interactive historical price visualization
- **Pattern Strength Classification**: Strong/Medium/Weak reliability indicators

### 🧪 Backtesting Engine
- Test trading strategies on historical data
- Performance metrics: Sharpe Ratio, Max Drawdown, Win Rate, Profit Factor
- SQLite-based price caching for fast backtests
- CSV export of backtest results
- Multiple built-in strategies

### 📰 News Sentiment Analysis
- Real-time financial news aggregation
- Sentiment analysis on news articles
- Integration with MoneyControl, Economic Times, and other sources
- Sentiment scoring for trading decisions

### 🧠 RAG System
- ChromaDB vector database with 23+ trading documents
- Semantic search for relevant trading knowledge
- Context-aware AI recommendations
- Continuous learning from historical data

### 💾 Settings Persistence
- Dual-layer storage (MongoDB + localStorage)
- API keys persist across browser sessions
- Offline fallback when backend unavailable
- Secure settings management

## 🚀 Quick Start

### Prerequisites

- **Python 3.9+**
- **Node.js 16+**
- **MongoDB** (local or cloud)
- **API Keys**:
  - OpenAI API key (for GPT models)
  - Google Gemini API key (optional)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/srewoo/NeuralTrader.git
cd NeuralTrader
```

2. **Backend Setup**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Environment Configuration**

Create `backend/.env`:
```bash
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=neuraltrader_db
```

Create `frontend/.env.local`:
```bash
REACT_APP_BACKEND_URL=http://localhost:8000
```

5. **Start the Application**
```bash
# From project root
./start.sh

# Or manually:
# Terminal 1 - Backend
cd backend
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 - Frontend
cd frontend
npm start
```

6. **Access the Application**
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📖 Documentation

- **[Quick Start Guide](QUICK_START_GUIDE.md)** - Get up and running quickly
- **[Usage Guide](USAGE_GUIDE.md)** - Comprehensive usage instructions
- **[Environment Setup](ENV_SETUP_GUIDE.md)** - Environment configuration
- **[Settings Persistence](SETTINGS_PERSISTENCE.md)** - How settings are stored

## 🏗️ Architecture

### Backend (FastAPI + Python)
```
backend/
├── agents/              # Multi-agent system
│   ├── data_agent.py
│   ├── analysis_agent.py
│   ├── knowledge_agent.py
│   ├── reasoning_agent.py
│   └── validator_agent.py
├── rag/                 # RAG system
│   ├── vector_store.py
│   ├── embeddings.py
│   └── retrieval.py
├── backtesting/         # Backtesting engine
│   ├── engine.py
│   ├── strategies.py
│   └── metrics.py
├── news/                # News sentiment
│   ├── sources.py
│   └── sentiment.py
├── patterns/            # Candlestick patterns
│   └── candlestick.py
└── server.py            # FastAPI server
```

### Frontend (React 19)
```
frontend/
├── src/
│   ├── pages/           # Main pages
│   │   ├── Dashboard.jsx
│   │   ├── Settings.jsx
│   │   ├── Backtesting.jsx
│   │   └── AnalysisHistory.jsx
│   ├── components/      # Reusable components
│   │   ├── StockChart.jsx
│   │   ├── AgentWorkflow.jsx
│   │   ├── ReasoningLog.jsx
│   │   └── CandlestickPatterns.jsx
│   ├── config/          # Configuration
│   │   └── api.js
│   └── utils/           # Utilities
│       └── settingsStorage.js
```

## 🎯 Usage Examples

### Running AI Analysis
```python
# Select a stock (e.g., RELIANCE, TCS, INFY)
# Choose AI model (GPT-4.1, Gemini 2.5, etc.)
# Click "Run Analysis"
# View multi-agent workflow and recommendations
```

### Backtesting a Strategy
```python
# Navigate to Backtesting page
# Select strategy (e.g., RSI Mean Reversion)
# Choose stock and date range
# Set initial capital
# Run backtest
# View performance metrics and equity curve
```

### Viewing Candlestick Patterns
```python
# Select a stock in Dashboard
# Scroll to Candlestick Patterns section
# View detected patterns (bullish/bearish/indecision)
# Filter by time period (1d, 7d, 15d, 30d, 60d)
```

## 🔧 Configuration

### Supported AI Models

**OpenAI:**
- GPT-4.1 (Recommended)
- GPT-4o
- o3-mini (Fast)
- o1 (Deep Reasoning)

**Google Gemini:**
- Gemini 2.5 Flash
- Gemini 2.5 Pro
- Gemini 2.0 Flash

### Supported Exchanges
- **NSE** (National Stock Exchange of India)
- **BSE** (Bombay Stock Exchange)

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- SMA (Simple Moving Average)
- EMA (Exponential Moving Average)
- Bollinger Bands
- ATR (Average True Range)
- OBV (On-Balance Volume)
- Stochastic Oscillator

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest

# Run frontend tests
cd frontend
npm test
```

## 📊 Performance Metrics

### Backtesting Metrics
- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted return
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Total Trades**: Number of executed trades

## 🛠️ Tech Stack

### Backend
- **FastAPI** - Modern Python web framework
- **MongoDB** - Document database
- **ChromaDB** - Vector database for RAG
- **LangGraph** - Multi-agent orchestration
- **yfinance** - Stock data fetching
- **ta** - Technical analysis library
- **LiteLLM** - LLM orchestration

### Frontend
- **React 19** - UI framework
- **Shadcn/UI** - Component library
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Recharts** - Charts and graphs
- **Axios** - HTTP client

## 🔒 Security

- API keys stored securely in MongoDB
- Environment variables for sensitive data
- CORS protection
- Input validation and sanitization
- No hardcoded credentials

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**NeuralTrader is for educational and research purposes only. It is NOT financial advice.**

- Trading stocks involves significant risk
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- Consult with a qualified financial advisor
- The developers are not responsible for any financial losses

## 🙏 Acknowledgments

- **OpenAI** for GPT models
- **Google** for Gemini models
- **Yahoo Finance** for stock data
- **Shadcn** for UI components
- **LangChain** for agent orchestration

## 📧 Contact

- **GitHub**: [@srewoo](https://github.com/srewoo)
- **Repository**: [NeuralTrader](https://github.com/srewoo/NeuralTrader)

## 🌟 Star History

If you find this project useful, please consider giving it a star! ⭐

---

**Built with ❤️ using AI and modern web technologies**
