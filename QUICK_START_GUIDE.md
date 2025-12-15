# 🚀 Quick Start Guide - NeuralTrader AI Trading Advisor

**Last Updated:** December 15, 2025  
**Status:** Phase 1 & 2 Complete

---

## 📋 Prerequisites

- **Python 3.10+**
- **Node.js 18+** and **Yarn**
- **MongoDB** (local or cloud)
- **API Keys:**
  - OpenAI API Key (for GPT-4.1, GPT-4o, o3-mini)
  - OR Google Gemini API Key (for Gemini 2.5 Flash, 2.0 Flash)

---

## 🔧 Setup Instructions

### 1. Clone and Navigate

```bash
cd /Users/sharajrewoo/DemoReposQA/NeuralTrader
```

### 2. Backend Setup

```bash
# Navigate to backend
cd backend

# Install Python dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
MONGO_URL=mongodb://localhost:27017
DB_NAME=NeuralTrader
EOF

# Initialize RAG Knowledge Base
python -m rag.seed_data
```

**Expected Output:**
```
INFO - Starting knowledge base seeding...
INFO - Seeded 5 trading patterns
INFO - Seeded 5 trading strategies
INFO - Seeded 8 market insights
INFO - Seeded 5 Indian market context items
INFO - Knowledge base seeding complete. Total documents: 23
```

### 3. Frontend Setup

```bash
# Navigate to frontend
cd ../frontend

# Install dependencies
yarn install

# Create .env file
echo "REACT_APP_BACKEND_URL=http://localhost:8000" > .env
```

---

## ▶️ Running the Application

### Terminal 1: Start Backend

```bash
cd backend
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete.
```

### Terminal 2: Start Frontend

```bash
cd frontend
yarn start
```

**Expected Output:**
```
Compiled successfully!
You can now view frontend in the browser.
  Local:            http://localhost:3000
```

---

## 🎯 First Time Usage

### 1. Configure API Keys

1. Navigate to http://localhost:3000
2. Click **Settings** in the navigation
3. Enter your API keys:
   - **OpenAI API Key:** `sk-...`
   - **OR Gemini API Key:** `AIza...`
4. Select your preferred model
5. Click **Save Settings**

### 2. Run Your First Analysis

1. Go back to **Dashboard**
2. Search for a stock (e.g., "RELIANCE", "TCS", "INFY")
3. Select a stock from results
4. View stock data, chart, and technical indicators
5. Select AI model (GPT-4.1, Gemini 2.5 Flash, etc.)
6. Click **Run Analysis**
7. Watch the agent workflow execute:
   - ✅ Data Collection Agent
   - ✅ Technical Analysis Agent
   - ✅ RAG Knowledge Agent (with real semantic search!)
   - ✅ Deep Reasoning Agent
   - ✅ Validator Agent
8. View AI recommendation with:
   - BUY/SELL/HOLD signal
   - Confidence percentage
   - Entry price, target price, stop loss
   - Reasoning chain
   - Key risks

### 3. Explore Features

- **Analysis History:** View all past analyses
- **Watchlist:** Add stocks to watchlist for quick access
- **Stock Detail:** Click on any analysis for detailed view
- **Export/Share:** Export analysis as JSON or share

---

## 🧪 Testing RAG System

### Check RAG Statistics

```bash
curl http://localhost:8000/api/rag/stats
```

**Expected Response:**
```json
{
  "name": "stock_knowledge",
  "count": 23,
  "metadata": {
    "description": "Stock market knowledge base for RAG",
    "hnsw:space": "cosine"
  }
}
```

### Test RAG Search

```bash
curl -X POST http://localhost:8000/api/rag/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "RSI oversold bullish reversal pattern",
    "n_results": 3,
    "min_similarity": 0.5
  }'
```

**Expected Response:**
```json
{
  "query": "RSI oversold bullish reversal pattern",
  "results": [
    {
      "id": "abc123...",
      "content": "Pattern: Bullish RSI Divergence\nDescription: Price makes lower lows while RSI makes higher lows...",
      "metadata": {
        "category": "patterns",
        "pattern_name": "Bullish RSI Divergence",
        "confidence": 75
      },
      "similarity": 0.82,
      "distance": 0.22
    },
    ...
  ],
  "count": 3
}
```

---

## 📊 API Endpoints

### Stock Data
- `GET /api/stocks/{symbol}` - Get stock data
- `GET /api/stocks/{symbol}/indicators` - Get technical indicators
- `GET /api/stocks/{symbol}/history?period=6mo` - Get price history
- `GET /api/stocks/search?q={query}` - Search stocks

### Analysis
- `POST /api/analyze` - Run AI analysis
- `GET /api/analysis/history?limit=20` - Get analysis history
- `GET /api/analysis/{id}` - Get specific analysis
- `DELETE /api/analysis/{id}` - Delete analysis

### RAG System
- `GET /api/rag/stats` - Get RAG statistics
- `POST /api/rag/seed` - Seed knowledge base (background)
- `POST /api/rag/ingest` - Ingest custom document
- `POST /api/rag/search` - Search knowledge base

### Watchlist
- `GET /api/watchlist` - Get watchlist
- `POST /api/watchlist/{symbol}` - Add to watchlist
- `DELETE /api/watchlist/{symbol}` - Remove from watchlist

### Settings
- `GET /api/settings` - Get settings
- `POST /api/settings` - Save settings

---

## 🔍 Verification Checklist

### ✅ Backend Health
- [ ] Backend starts without errors
- [ ] MongoDB connection successful
- [ ] RAG system initialized (23 documents)
- [ ] API endpoints responding

### ✅ Frontend Health
- [ ] Frontend compiles without errors
- [ ] No import errors in console
- [ ] All pages render correctly
- [ ] Components display properly

### ✅ Core Features
- [ ] Stock search works
- [ ] Stock data displays
- [ ] Price chart renders
- [ ] Technical indicators show
- [ ] AI analysis runs successfully
- [ ] Agent workflow displays
- [ ] Reasoning log shows
- [ ] Analysis history works

### ✅ RAG Integration
- [ ] RAG Knowledge Agent shows real data
- [ ] Patterns found count is accurate
- [ ] Similarity scores displayed
- [ ] LLM prompt includes RAG context

---

## 🐛 Common Issues

### Issue: MongoDB Connection Failed
**Error:** `pymongo.errors.ServerSelectionTimeoutError`

**Solution:**
```bash
# Start MongoDB
mongod --dbpath /path/to/data

# Or use MongoDB Atlas (cloud)
# Update MONGO_URL in .env
```

### Issue: RAG System Not Initialized
**Error:** `Collection count: 0`

**Solution:**
```bash
cd backend
python -m rag.seed_data
```

### Issue: Frontend Build Errors
**Error:** `Module not found: Can't resolve '@/components/StockChart'`

**Solution:**
```bash
cd frontend
rm -rf node_modules yarn.lock
yarn install
yarn start
```

### Issue: API Key Not Working
**Error:** `401 Unauthorized` or `Invalid API key`

**Solution:**
1. Verify API key is correct
2. Check API key has credits/quota
3. Ensure correct provider selected (OpenAI vs Gemini)
4. Save settings and refresh page

### Issue: Stock Data Not Loading
**Error:** `Failed to fetch stock data`

**Solution:**
- Stock symbol must be valid NSE/BSE symbol
- Try adding `.NS` suffix (e.g., `RELIANCE.NS`)
- Check internet connection
- yfinance may have rate limits

---

## 📚 Project Structure

```
NeuralTrader/
├── backend/
│   ├── server.py                 # FastAPI server
│   ├── requirements.txt          # Python dependencies
│   ├── .env                      # Environment variables
│   ├── rag/                      # RAG system
│   │   ├── vector_store.py       # ChromaDB
│   │   ├── embeddings.py         # Sentence transformers
│   │   ├── ingestion.py          # Document processing
│   │   ├── retrieval.py          # Semantic search
│   │   └── seed_data.py          # Initial knowledge
│   └── chroma_db/                # Vector database (auto-created)
├── frontend/
│   ├── src/
│   │   ├── App.js                # Main app
│   │   ├── components/           # React components
│   │   │   ├── StockChart.jsx    # Price chart
│   │   │   ├── AgentWorkflow.jsx # Agent visualization
│   │   │   ├── ReasoningLog.jsx  # AI reasoning
│   │   │   └── ui/               # Shadcn components
│   │   └── pages/                # Pages
│   │       ├── Dashboard.jsx     # Main dashboard
│   │       ├── AnalysisHistory.jsx
│   │       ├── StockDetail.jsx
│   │       └── Settings.jsx
│   ├── package.json              # Node dependencies
│   └── .env                      # Frontend config
├── PHASE1_COMPLETION_REPORT.md   # Phase 1 report
├── PHASE2_COMPLETION_REPORT.md   # Phase 2 report
└── QUICK_START_GUIDE.md          # This file
```

---

## 🎯 What's Implemented

### ✅ Phase 1: Frontend Components (COMPLETE)
- StockChart component with Recharts
- AgentWorkflow visualization
- ReasoningLog display
- AnalysisHistory page
- StockDetail page

### ✅ Phase 2: RAG System (COMPLETE)
- ChromaDB vector store
- Sentence transformer embeddings
- Document ingestion pipeline
- Semantic search and retrieval
- 23+ seeded knowledge documents
- Integration with AI analysis

### ⏳ Phase 3: Multi-Agent System (PENDING)
- LangGraph orchestration
- Real agent implementations
- State management
- Observable workflow

### ⏳ Phase 4: Backtesting (PENDING)
- Backtesting engine
- Strategy framework
- Performance metrics
- CSV export

### ⏳ Phase 5: News Sentiment (PENDING)
- News API integration
- Sentiment analysis
- News feed UI

---

## 📞 Support

For issues or questions:
1. Check this guide
2. Review Phase 1 & 2 completion reports
3. Check server logs for errors
4. Verify all dependencies installed

---

## 🎉 You're Ready!

Your NeuralTrader AI Trading Advisor is now running with:
- ✅ Beautiful React UI
- ✅ Real-time stock data
- ✅ Technical indicators
- ✅ AI-powered analysis
- ✅ RAG-enhanced recommendations
- ✅ Agent workflow visualization

**Happy Trading! 📈**

