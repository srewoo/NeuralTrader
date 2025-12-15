import { motion } from "framer-motion";
import { Check, Loader2, Database, TrendingUp, Brain, Search, CheckCircle2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

/**
 * AgentWorkflow Component
 * Visualizes the multi-agent AI workflow process
 * @param {Object} props - Component props
 * @param {Array} props.steps - Array of agent step objects from analysis
 * @param {boolean} props.isRunning - Whether analysis is currently running
 */
export default function AgentWorkflow({ steps, isRunning = false }) {
  // Default steps when analysis is running but no steps yet
  const defaultSteps = [
    { 
      agent_name: "Data Collection Agent", 
      status: "running", 
      message: "Fetching market data and indicators...",
      icon: Database
    },
    { 
      agent_name: "Technical Analysis Agent", 
      status: "pending", 
      message: "Waiting to analyze patterns",
      icon: TrendingUp
    },
    { 
      agent_name: "RAG Knowledge Agent", 
      status: "pending", 
      message: "Waiting to retrieve historical insights",
      icon: Search
    },
    { 
      agent_name: "Deep Reasoning Agent", 
      status: "pending", 
      message: "Waiting to generate recommendation",
      icon: Brain
    },
    { 
      agent_name: "Validator Agent", 
      status: "pending", 
      message: "Waiting to validate analysis",
      icon: CheckCircle2
    }
  ];

  // Use provided steps or default steps
  const displaySteps = steps && steps.length > 0 ? steps : defaultSteps;

  // Map agent names to icons
  const getAgentIcon = (agentName) => {
    if (agentName.toLowerCase().includes("data") || agentName.toLowerCase().includes("collection")) {
      return Database;
    }
    if (agentName.toLowerCase().includes("technical") || agentName.toLowerCase().includes("analysis")) {
      return TrendingUp;
    }
    if (agentName.toLowerCase().includes("rag") || agentName.toLowerCase().includes("knowledge")) {
      return Search;
    }
    if (agentName.toLowerCase().includes("reasoning") || agentName.toLowerCase().includes("decision")) {
      return Brain;
    }
    if (agentName.toLowerCase().includes("validator") || agentName.toLowerCase().includes("critic")) {
      return CheckCircle2;
    }
    return Brain;
  };

  // Get status badge styling
  const getStatusBadge = (status) => {
    switch (status?.toLowerCase()) {
      case "completed":
      case "success":
        return (
          <Badge className="bg-success-dim text-success border-0 text-xs">
            <Check className="w-3 h-3 mr-1" />
            Completed
          </Badge>
        );
      case "running":
      case "in_progress":
        return (
          <Badge className="bg-primary/10 text-primary border-0 text-xs">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Running
          </Badge>
        );
      case "failed":
      case "error":
        return (
          <Badge className="bg-danger-dim text-danger border-0 text-xs">
            Error
          </Badge>
        );
      default:
        return (
          <Badge className="bg-[#1F1F1F] text-text-secondary border-0 text-xs">
            Pending
          </Badge>
        );
    }
  };

  // Get status icon
  const getStatusIcon = (status) => {
    switch (status?.toLowerCase()) {
      case "completed":
      case "success":
        return <Check className="w-5 h-5 text-success" />;
      case "running":
      case "in_progress":
        return <Loader2 className="w-5 h-5 text-primary animate-spin" />;
      case "failed":
      case "error":
        return <div className="w-5 h-5 rounded-full border-2 border-danger" />;
      default:
        return <div className="w-5 h-5 rounded-full border-2 border-[#2F2F2F]" />;
    }
  };

  return (
    <div className="space-y-4" data-testid="agent-workflow">
      <div className="flex items-center justify-between mb-4">
        <h4 className="text-sm font-heading font-medium text-text-primary">
          Agent Workflow
        </h4>
        {isRunning && (
          <Badge className="bg-ai-accent/10 text-ai-accent border-0 text-xs animate-pulse">
            <Loader2 className="w-3 h-3 mr-1 animate-spin" />
            Processing
          </Badge>
        )}
      </div>

      <div className="space-y-3">
        {displaySteps.map((step, index) => {
          const Icon = step.icon || getAgentIcon(step.agent_name);
          const isCompleted = step.status?.toLowerCase() === "completed" || step.status?.toLowerCase() === "success";
          const isRunning = step.status?.toLowerCase() === "running" || step.status?.toLowerCase() === "in_progress";
          const isFailed = step.status?.toLowerCase() === "failed" || step.status?.toLowerCase() === "error";

          return (
            <motion.div
              key={index}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.1 }}
              className={cn(
                "relative flex items-start gap-4 p-4 rounded-lg border transition-all",
                isCompleted && "bg-success-dim/30 border-success/20",
                isRunning && "bg-primary/5 border-primary/20 shadow-glow-primary",
                isFailed && "bg-danger-dim/30 border-danger/20",
                !isCompleted && !isRunning && !isFailed && "bg-surface-highlight border-[#1F1F1F]"
              )}
              data-testid={`agent-step-${index}`}
            >
              {/* Connector Line */}
              {index < displaySteps.length - 1 && (
                <div 
                  className={cn(
                    "absolute left-[30px] top-[60px] w-0.5 h-6",
                    isCompleted ? "bg-success/30" : "bg-[#2F2F2F]"
                  )}
                />
              )}

              {/* Status Icon */}
              <div className={cn(
                "flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center",
                isCompleted && "bg-success-dim",
                isRunning && "bg-primary/10",
                isFailed && "bg-danger-dim",
                !isCompleted && !isRunning && !isFailed && "bg-[#1F1F1F]"
              )}>
                {getStatusIcon(step.status)}
              </div>

              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between gap-2 mb-1">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4 text-ai-accent" />
                    <h5 className="text-sm font-medium text-text-primary">
                      {step.agent_name}
                    </h5>
                  </div>
                  {getStatusBadge(step.status)}
                </div>
                
                <p className="text-xs text-text-secondary leading-relaxed">
                  {step.message}
                </p>

                {/* Additional data if available */}
                {step.data && Object.keys(step.data).length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-2">
                    {Object.entries(step.data).slice(0, 3).map(([key, value]) => (
                      <div 
                        key={key} 
                        className="inline-flex items-center gap-1.5 px-2 py-1 rounded bg-black/20 text-xs"
                      >
                        <span className="text-text-secondary">{key}:</span>
                        <span className="text-text-primary font-data">
                          {typeof value === 'number' ? value.toLocaleString() : String(value)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {/* Summary if all completed */}
      {steps && steps.length > 0 && steps.every(s => s.status?.toLowerCase() === "completed") && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center gap-2 p-3 rounded-lg bg-success-dim/30 border border-success/20 mt-4"
        >
          <Check className="w-5 h-5 text-success" />
          <p className="text-sm text-success font-medium">
            All agents completed successfully
          </p>
        </motion.div>
      )}
    </div>
  );
}

