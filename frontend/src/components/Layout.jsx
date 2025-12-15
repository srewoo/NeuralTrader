import { Outlet, Link, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { 
  TrendingUp, 
  Settings, 
  History, 
  BarChart3,
  Zap,
  LineChart
} from "lucide-react";

export const Layout = () => {
  const location = useLocation();

  const navItems = [
    { path: "/", icon: BarChart3, label: "Dashboard" },
    { path: "/history", icon: History, label: "History" },
    { path: "/backtesting", icon: LineChart, label: "Backtest" },
    { path: "/settings", icon: Settings, label: "Settings" },
  ];

  return (
    <div className="min-h-screen bg-background" data-testid="app-layout">
      {/* Header */}
      <header className="sticky top-0 z-50 glass-header">
        <div className="max-w-[1920px] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center gap-3" data-testid="logo-link">
              <div className="relative">
                <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                  <TrendingUp className="w-6 h-6 text-primary" />
                </div>
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-success rounded-full animate-pulse" />
              </div>
              <div>
                <h1 className="text-lg font-heading font-bold text-text-primary tracking-tight">
                  NeuralTrader
                </h1>
                <p className="text-xs text-text-secondary">NSE/BSE Trading Signals</p>
              </div>
            </Link>

            {/* Navigation */}
            <nav className="flex items-center gap-2">
              {navItems.map((item) => {
                const isActive = location.pathname === item.path;
                return (
                  <Link
                    key={item.path}
                    to={item.path}
                    data-testid={`nav-${item.label.toLowerCase()}`}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                      isActive
                        ? "bg-primary/10 text-primary"
                        : "text-text-secondary hover:text-text-primary hover:bg-white/5"
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    <span className="hidden sm:inline">{item.label}</span>
                  </Link>
                );
              })}
            </nav>

            {/* AI Status */}
            <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-ai-accent/10 border border-ai-accent/20">
              <Zap className="w-4 h-4 text-ai-accent" />
              <span className="text-xs font-medium text-ai-accent">AI Ready</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative">
        <div className="noise-overlay absolute inset-0 pointer-events-none" />
        <motion.div
          key={location.pathname}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="relative z-10"
        >
          <Outlet />
        </motion.div>
      </main>
    </div>
  );
};

export default Layout;
