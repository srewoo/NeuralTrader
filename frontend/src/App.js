import "@/App.css";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Toaster } from "@/components/ui/sonner";
import Dashboard from "@/pages/Dashboard";
import Settings from "@/pages/Settings";
import AIRecommends from "@/pages/AIRecommends";
import Help from "@/pages/Help";
import Alerts from "@/pages/Alerts";
import Layout from "@/components/Layout";

function App() {
  return (
    <div className="App min-h-screen bg-background">
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<Dashboard />} />
            <Route path="ai-picks" element={<AIRecommends />} />
            <Route path="alerts" element={<Alerts />} />
            <Route path="settings" element={<Settings />} />
            <Route path="help" element={<Help />} />
          </Route>
        </Routes>
      </BrowserRouter>
      <Toaster position="top-right" richColors />
    </div>
  );
}

export default App;
