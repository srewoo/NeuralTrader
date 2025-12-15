# NeuralTrader - Usage Guide

## Quick Start

### Starting the Application

Simply run:

```bash
./start.sh
```

This will:
- ✅ Start the backend server on `http://localhost:8000`
- ✅ Start the frontend server on `http://localhost:3000`
- ✅ Check for and install dependencies if needed
- ✅ Run both servers in the background
- ✅ Create log files for troubleshooting

### Stopping the Application

```bash
./stop.sh
```

This will:
- ✅ Stop both backend and frontend servers
- ✅ Clean up all running processes
- ✅ Verify successful shutdown

## Server Details

### Backend Server
- **URL**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Technology**: FastAPI with uvicorn
- **Log File**: `logs/backend.log`
- **PID File**: `logs/backend.pid`

### Frontend Server
- **URL**: http://localhost:3000
- **Technology**: React 19 with Vite/Create React App
- **Log File**: `logs/frontend.log`
- **PID File**: `logs/frontend.pid`

## Monitoring

### View Live Logs

**Backend logs:**
```bash
tail -f logs/backend.log
```

**Frontend logs:**
```bash
tail -f logs/frontend.log
```

**Both logs simultaneously:**
```bash
tail -f logs/*.log
```

### Check Server Status

```bash
# Check backend
curl http://localhost:8000/health

# Check frontend
curl http://localhost:3000
```

### Check Running Processes

```bash
# Check if backend is running
pgrep -f "uvicorn.*backend.server:app"

# Check if frontend is running
pgrep -f "react-scripts start"

# Check ports
lsof -ti:8000  # Backend port
lsof -ti:3000  # Frontend port
```

## Manual Server Management

### Starting Servers Manually

**Backend:**
```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm start
```

### Stopping Servers Manually

**Kill by process name:**
```bash
# Backend
pkill -f "uvicorn.*backend.server:app"

# Frontend
pkill -f "react-scripts start"
```

**Kill by port:**
```bash
# Backend (port 8000)
kill $(lsof -ti:8000)

# Frontend (port 3000)
kill $(lsof -ti:3000)
```

**Force kill (use with caution):**
```bash
pkill -9 -f "uvicorn.*backend.server:app"
pkill -9 -f "react-scripts start"
```

## Environment Setup

### First-Time Setup

**Backend:**
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Frontend:**
```bash
cd frontend
npm install
```

### Environment Variables

Create a `.env` file in the backend directory:

```bash
# API Keys (configure in UI or here)
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here

# MongoDB (optional - defaults to localhost)
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=neuraltrader_db

# Application Settings
DEBUG=True
LOG_LEVEL=INFO
```

Create a `.env` file in the frontend directory:

```bash
# Backend URL
REACT_APP_BACKEND_URL=http://localhost:8000

# Other settings
REACT_APP_ENV=development
```

## Troubleshooting

### Port Already in Use

**Problem:** Port 8000 or 3000 is already in use

**Solution:**
```bash
# Find and kill the process using the port
lsof -ti:8000 | xargs kill -9  # Backend
lsof -ti:3000 | xargs kill -9  # Frontend

# Or run the stop script
./stop.sh
```

### Dependencies Not Installed

**Problem:** Missing Python packages or npm modules

**Solution:**
```bash
# Backend
cd backend
source venv/bin/activate
pip install -r requirements.txt

# Frontend
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### Servers Won't Start

**Problem:** Scripts fail to start servers

**Solution:**
1. Check log files: `cat logs/backend.log` or `cat logs/frontend.log`
2. Ensure all dependencies are installed
3. Check if ports are available
4. Verify environment variables
5. Try manual startup to see detailed error messages

### MongoDB Connection Issues

**Problem:** Cannot connect to MongoDB

**Solution:**
```bash
# Check if MongoDB is running
brew services list | grep mongodb

# Start MongoDB
brew services start mongodb-community

# Or use Docker
docker run -d -p 27017:27017 --name mongodb mongo:latest
```

### Virtual Environment Issues

**Problem:** Cannot activate virtual environment

**Solution:**
```bash
cd backend
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Development Tips

### Hot Reloading

Both servers support hot reloading:
- **Backend**: Automatically reloads on Python file changes (uvicorn --reload)
- **Frontend**: Automatically reloads on JavaScript/CSS changes (React dev server)

### API Testing

Use the interactive API documentation:
```
http://localhost:8000/docs
```

Or use curl:
```bash
# Search for stocks
curl http://localhost:8000/api/stocks/search?q=RELIANCE

# Get stock data
curl http://localhost:8000/api/stocks/RELIANCE

# Run AI analysis
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "RELIANCE",
    "model": "gpt-4.1",
    "provider": "openai"
  }'
```

### Database Management

**View MongoDB data:**
```bash
mongosh
use neuraltrader_db
db.watchlist.find()
db.analysis_history.find()
db.settings.find()
```

**Clear database:**
```bash
mongosh neuraltrader_db --eval "db.dropDatabase()"
```

## Production Deployment

### Build Frontend for Production

```bash
cd frontend
npm run build
```

This creates an optimized production build in `frontend/build/`.

### Run Backend in Production

```bash
cd backend
source venv/bin/activate
uvicorn server:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using Process Managers

**PM2 (recommended):**
```bash
# Install PM2
npm install -g pm2

# Start backend
pm2 start "uvicorn server:app --host 0.0.0.0 --port 8000" --name neuraltrader-backend --cwd backend

# Start frontend (dev)
pm2 start "npm start" --name neuraltrader-frontend --cwd frontend

# Or serve production build
pm2 start "npx serve -s build -l 3000" --name neuraltrader-frontend --cwd frontend

# Manage processes
pm2 list
pm2 logs
pm2 restart all
pm2 stop all
pm2 delete all
```

**Supervisor:**
```bash
# Install supervisor
sudo apt-get install supervisor  # Ubuntu/Debian
brew install supervisor          # macOS

# Create config files in /etc/supervisor/conf.d/
# Then start services
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl start all
```

## Scripts Overview

### start.sh Features
- ✅ Checks if servers are already running
- ✅ Creates virtual environment if missing
- ✅ Installs dependencies if needed
- ✅ Starts both servers in background
- ✅ Creates PID files for tracking
- ✅ Displays server status and access points
- ✅ Color-coded output for clarity

### stop.sh Features
- ✅ Stops servers using PID files
- ✅ Falls back to process name matching
- ✅ Kills processes on specific ports
- ✅ Verifies successful shutdown
- ✅ Provides manual cleanup commands if needed
- ✅ Color-coded output for clarity

## Advanced Usage

### Custom Ports

Edit the scripts to use different ports:

**start.sh:**
```bash
# Change backend port (default 8000)
uvicorn server:app --reload --host 0.0.0.0 --port 8080

# Frontend port is configured in package.json or .env
PORT=3001 npm start
```

**stop.sh:**
```bash
# Update port numbers in lsof commands
lsof -ti:8080  # New backend port
lsof -ti:3001  # New frontend port
```

### Running Only Backend or Frontend

**Backend only:**
```bash
cd backend
source venv/bin/activate
uvicorn server:app --reload --host 0.0.0.0 --port 8000
```

**Frontend only:**
```bash
cd frontend
npm start
```

### Background vs Foreground

The start script runs servers in background. To run in foreground:

```bash
# Remove 'nohup' and '&' from start.sh
# Or run manually as shown above
```

## Support

For issues or questions:
1. Check log files first: `logs/backend.log` and `logs/frontend.log`
2. Review this guide for common solutions
3. Check the main documentation: `README.md`
4. View API docs: http://localhost:8000/docs

## Summary

| Command | Description |
|---------|-------------|
| `./start.sh` | Start all servers |
| `./stop.sh` | Stop all servers |
| `tail -f logs/backend.log` | View backend logs |
| `tail -f logs/frontend.log` | View frontend logs |
| `curl http://localhost:8000/health` | Check backend health |
| `lsof -ti:8000` | Check backend port |
| `lsof -ti:3000` | Check frontend port |

**That's it! You're ready to use NeuralTrader! 🚀**

