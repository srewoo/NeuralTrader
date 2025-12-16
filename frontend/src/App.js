import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import Dashboard from "@/pages/Dashboard";
import Settings from "@/pages/Settings";
import AnalysisHistory from "@/pages/AnalysisHistory";
import StockDetail from "@/pages/StockDetail";
import Backtesting from "@/pages/Backtesting";
import AIRecommends from "@/pages/AIRecommends";
import Help from "@/pages/Help";
import Layout from "@/components/Layout";

function App() {
  return (
    <div className="App min-h-screen bg-background">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="ai-recommends" element={<AIRecommends />} />
            <Route path="settings" element={<Settings />} />
            <Route path="history" element={<AnalysisHistory />} />
            <Route path="backtesting" element={<Backtesting />} />
            <Route path="help" element={<Help />} />
            <Route path="stock/:symbol" element={<StockDetail />} />
            <Route path="analysis/:analysisId" element={<StockDetail />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;
