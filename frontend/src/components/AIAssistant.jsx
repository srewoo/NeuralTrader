import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import {
  MessageSquare,
  Send,
  X,
  Loader2,
  Brain,
  Sparkles,
  TrendingUp,
  FileText,
  BarChart3,
  Maximize2,
  Minimize2
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { API_URL } from "@/config/api";

export default function AIAssistant() {
  const [isOpen, setIsOpen] = useState(false);
  const [isExpanded, setIsExpanded] = useState(false);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I'm your AI trading assistant. Ask me about stocks, market analysis, portfolio strategies, or any trading questions!",
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const quickActions = [
    { label: "Market Overview", icon: BarChart3, prompt: "Give me today's market overview and key movements" },
    { label: "Top Picks", icon: TrendingUp, prompt: "What are the top stock picks for today based on technical analysis?" },
    { label: "News Summary", icon: FileText, prompt: "Summarize the latest market news and their potential impact" },
    { label: "Portfolio Tips", icon: Sparkles, prompt: "Give me some portfolio diversification tips for Indian markets" }
  ];

  const handleSubmit = async (e, customPrompt = null) => {
    e?.preventDefault();
    const prompt = customPrompt || input;
    if (!prompt.trim() || isLoading) return;

    const userMessage = {
      role: "user",
      content: prompt,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_URL}/llm/ask`, {
        question: prompt,
        context: "Indian stock market trading, NSE, BSE"
      });

      const assistantMessage = {
        role: "assistant",
        content: response.data.answer || response.data.response || "I couldn't generate a response. Please try again.",
        timestamp: new Date(),
        model: response.data.model
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("AI Assistant error:", error);
      const errorMessage = {
        role: "assistant",
        content: "Sorry, I encountered an error. Please check your API keys in Settings and try again.",
        timestamp: new Date(),
        isError: true
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const formatTime = (date) => {
    return new Date(date).toLocaleTimeString('en-IN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  return (
    <>
      {/* Floating Button */}
      <AnimatePresence>
        {!isOpen && (
          <motion.button
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            exit={{ scale: 0 }}
            whileHover={{ scale: 1.1 }}
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full bg-ai-accent text-white shadow-lg shadow-ai-accent/30 flex items-center justify-center"
          >
            <MessageSquare className="w-6 h-6" />
          </motion.button>
        )}
      </AnimatePresence>

      {/* Chat Window */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 20, scale: 0.95 }}
            className={`fixed z-50 bg-surface border border-[#1F1F1F] rounded-2xl shadow-2xl flex flex-col ${
              isExpanded
                ? "inset-4 md:inset-8"
                : "bottom-6 right-6 w-[400px] h-[600px]"
            }`}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-[#1F1F1F]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-ai-accent/20 flex items-center justify-center">
                  <Brain className="w-5 h-5 text-ai-accent" />
                </div>
                <div>
                  <h3 className="font-heading font-bold text-text-primary">AI Assistant</h3>
                  <p className="text-xs text-text-secondary">Powered by Claude/GPT</p>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="h-8 w-8 p-0"
                >
                  {isExpanded ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setIsOpen(false)}
                  className="h-8 w-8 p-0"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            </div>

            {/* Messages */}
            <ScrollArea className="flex-1 p-4">
              <div className="space-y-4">
                {messages.map((message, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[85%] rounded-2xl p-3 ${
                        message.role === "user"
                          ? "bg-primary text-white rounded-br-md"
                          : message.isError
                          ? "bg-danger/10 border border-danger/20 rounded-bl-md"
                          : "bg-surface-highlight rounded-bl-md"
                      }`}
                    >
                      <p className={`text-sm whitespace-pre-wrap ${
                        message.role === "user" ? "text-white" : "text-text-primary"
                      }`}>
                        {message.content}
                      </p>
                      <div className={`flex items-center justify-between mt-2 ${
                        message.role === "user" ? "text-white/70" : "text-text-secondary"
                      }`}>
                        <span className="text-xs">{formatTime(message.timestamp)}</span>
                        {message.model && (
                          <Badge variant="outline" className="text-[10px] h-4">
                            {message.model}
                          </Badge>
                        )}
                      </div>
                    </div>
                  </motion.div>
                ))}

                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    className="flex justify-start"
                  >
                    <div className="bg-surface-highlight rounded-2xl rounded-bl-md p-3">
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-ai-accent" />
                        <span className="text-sm text-text-secondary">Thinking...</span>
                      </div>
                    </div>
                  </motion.div>
                )}

                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>

            {/* Quick Actions */}
            {messages.length <= 2 && (
              <div className="px-4 pb-2">
                <p className="text-xs text-text-secondary mb-2">Quick actions:</p>
                <div className="flex flex-wrap gap-2">
                  {quickActions.map((action, idx) => (
                    <Button
                      key={idx}
                      variant="outline"
                      size="sm"
                      onClick={(e) => handleSubmit(e, action.prompt)}
                      disabled={isLoading}
                      className="text-xs h-7 border-[#1F1F1F]"
                    >
                      <action.icon className="w-3 h-3 mr-1" />
                      {action.label}
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {/* Input */}
            <form onSubmit={handleSubmit} className="p-4 border-t border-[#1F1F1F]">
              <div className="flex items-center gap-2">
                <Input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about stocks, analysis, strategies..."
                  className="flex-1 bg-surface-highlight border-[#1F1F1F]"
                  disabled={isLoading}
                />
                <Button
                  type="submit"
                  disabled={!input.trim() || isLoading}
                  className="btn-primary h-10 w-10 p-0"
                >
                  {isLoading ? (
                    <Loader2 className="w-4 h-4 animate-spin" />
                  ) : (
                    <Send className="w-4 h-4" />
                  )}
                </Button>
              </div>
            </form>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
