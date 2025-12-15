import { motion } from "framer-motion";
import { Brain, ChevronRight, Lightbulb, AlertTriangle, TrendingUp } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

/**
 * ReasoningLog Component
 * Displays the AI's reasoning process and thought chain
 * @param {Object} props - Component props
 * @param {string|Array} props.reasoning - Reasoning text or array of reasoning steps
 */
export default function ReasoningLog({ reasoning }) {
  // Parse reasoning if it's a string
  const reasoningSteps = (() => {
    if (!reasoning) return [];

    if (Array.isArray(reasoning)) {
      return reasoning;
    }

    if (typeof reasoning === "string") {
      // Try to split by numbered patterns like "1)", "2.", "1:", or newlines
      // First try splitting by numbered patterns: "1)", "2)", etc.
      const numberedPattern = /(?:^|\s)(\d+)\)\s*/g;
      const hasNumberedPoints = numberedPattern.test(reasoning);

      let lines = [];

      if (hasNumberedPoints) {
        // Split by numbered points like "1) ", "2) ", etc.
        lines = reasoning
          .split(/\s*\d+\)\s*/)
          .map(line => line.trim())
          .filter(line => line.length > 0);
      } else {
        // Try splitting by newlines or other patterns
        lines = reasoning
          .split(/\n+|(?<=\.)\s+(?=[A-Z])|(?:^|\s)[-•]\s+/)
          .map(line => line.trim())
          .filter(line => line.length > 0);
      }

      // If still only one line, try to split on section headers
      if (lines.length === 1) {
        const sectionPattern = /([A-Z][a-zA-Z\s]+:)/g;
        if (sectionPattern.test(reasoning)) {
          lines = reasoning
            .split(/(?=[A-Z][a-zA-Z\s]+:)/)
            .map(line => line.trim())
            .filter(line => line.length > 0);
        }
      }

      return lines.map((line, index) => ({
        step: index + 1,
        content: line.replace(/^\d+[\.\)]\s*/, '').replace(/^[-•]\s*/, ''),
        type: detectStepType(line)
      }));
    }

    return [];
  })();

  // Detect step type from content
  function detectStepType(content) {
    const lowerContent = content.toLowerCase();

    // Check for specific section headers first
    if (lowerContent.startsWith("price action") || lowerContent.startsWith("trend")) {
      return "analysis";
    }
    if (lowerContent.startsWith("momentum") || lowerContent.startsWith("volatility")) {
      return "insight";
    }
    if (lowerContent.startsWith("pattern") || lowerContent.startsWith("synthesis") || lowerContent.startsWith("recommend")) {
      return "positive";
    }

    // Then check for keywords
    if (lowerContent.includes("risk") || lowerContent.includes("caution") || lowerContent.includes("warning") || lowerContent.includes("stop loss")) {
      return "warning";
    }
    if (lowerContent.includes("opportunity") || lowerContent.includes("bullish") || lowerContent.includes("buy") || lowerContent.includes("uptrend")) {
      return "positive";
    }
    if (lowerContent.includes("bearish") || lowerContent.includes("sell") || lowerContent.includes("downtrend") || lowerContent.includes("overbought")) {
      return "warning";
    }
    if (lowerContent.includes("analysis") || lowerContent.includes("indicator") || lowerContent.includes("technical") || lowerContent.includes("sma") || lowerContent.includes("rsi")) {
      return "analysis";
    }
    if (lowerContent.includes("insight") || lowerContent.includes("note") || lowerContent.includes("observe") || lowerContent.includes("momentum")) {
      return "insight";
    }

    return "default";
  }

  // Get icon based on step type
  const getStepIcon = (type) => {
    switch (type) {
      case "warning":
        return <AlertTriangle className="w-4 h-4 text-danger" />;
      case "positive":
        return <TrendingUp className="w-4 h-4 text-success" />;
      case "insight":
        return <Lightbulb className="w-4 h-4 text-yellow-500" />;
      case "analysis":
        return <Brain className="w-4 h-4 text-ai-accent" />;
      default:
        return <ChevronRight className="w-4 h-4 text-text-secondary" />;
    }
  };

  // Get badge for step type
  const getStepBadge = (type) => {
    switch (type) {
      case "warning":
        return <Badge variant="outline" className="text-xs border-danger/30 text-danger">Risk</Badge>;
      case "positive":
        return <Badge variant="outline" className="text-xs border-success/30 text-success">Opportunity</Badge>;
      case "insight":
        return <Badge variant="outline" className="text-xs border-yellow-500/30 text-yellow-500">Insight</Badge>;
      case "analysis":
        return <Badge variant="outline" className="text-xs border-ai-accent/30 text-ai-accent">Analysis</Badge>;
      default:
        return null;
    }
  };

  if (!reasoning || reasoningSteps.length === 0) {
    return (
      <div className="p-6 rounded-lg bg-surface-highlight border border-[#1F1F1F] text-center">
        <Brain className="w-8 h-8 mx-auto text-text-secondary/50 mb-2" />
        <p className="text-sm text-text-secondary">No reasoning data available</p>
      </div>
    );
  }

  return (
    <div className="space-y-3" data-testid="reasoning-log">
      <div className="flex items-center gap-2 mb-4">
        <Brain className="w-5 h-5 text-ai-accent" />
        <h4 className="text-sm font-heading font-medium text-text-primary">
          AI Reasoning Chain
        </h4>
        <Badge className="bg-ai-accent/10 text-ai-accent border-0 text-xs ml-auto">
          {reasoningSteps.length} Step{reasoningSteps.length !== 1 ? 's' : ''}
        </Badge>
      </div>

      <ScrollArea className="h-auto max-h-[400px]">
        <div className="space-y-2 pr-4">
          {reasoningSteps.map((step, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.05 }}
              className={cn(
                "relative flex gap-3 p-4 rounded-lg border transition-all hover:bg-surface-highlight",
                step.type === "warning" && "bg-danger-dim/20 border-danger/20",
                step.type === "positive" && "bg-success-dim/20 border-success/20",
                step.type === "insight" && "bg-yellow-500/5 border-yellow-500/20",
                step.type === "analysis" && "bg-ai-accent/5 border-ai-accent/20",
                step.type === "default" && "bg-surface border-[#1F1F1F]"
              )}
              data-testid={`reasoning-step-${index}`}
            >
              {/* Step indicator line */}
              {index < reasoningSteps.length - 1 && (
                <div className="absolute left-[22px] top-[52px] w-0.5 h-6 bg-[#2F2F2F]" />
              )}

              {/* Icon */}
              <div className={cn(
                "flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center",
                step.type === "warning" && "bg-danger-dim",
                step.type === "positive" && "bg-success-dim",
                step.type === "insight" && "bg-yellow-500/10",
                step.type === "analysis" && "bg-ai-accent/10",
                step.type === "default" && "bg-surface-highlight"
              )}>
                {getStepIcon(step.type)}
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-start justify-between gap-2 mb-1.5">
                  <span className="text-xs font-data text-text-secondary">
                    Step {step.step || index + 1}
                  </span>
                  {getStepBadge(step.type)}
                </div>

                {/* Extract and display section title if present */}
                {step.content.includes(':') ? (
                  <>
                    <h5 className="text-sm font-medium text-text-primary mb-1">
                      {step.content.split(':')[0].trim()}
                    </h5>
                    <p className="text-sm text-text-secondary leading-relaxed">
                      {step.content.split(':').slice(1).join(':').trim()}
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-text-primary leading-relaxed">
                    {step.content}
                  </p>
                )}

                {/* Additional metadata if available */}
                {step.confidence && (
                  <div className="mt-2">
                    <Badge variant="outline" className="text-xs">
                      Confidence: {step.confidence}%
                    </Badge>
                  </div>
                )}
              </div>
            </motion.div>
          ))}
        </div>
      </ScrollArea>

      {/* Summary footer */}
      <div className="flex items-center gap-2 pt-3 border-t border-[#1F1F1F] text-xs text-text-secondary">
        <Brain className="w-3.5 h-3.5" />
        <span>
          Generated using advanced chain-of-thought reasoning
        </span>
      </div>
    </div>
  );
}

