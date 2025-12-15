"""
Base Agent Class
Provides common functionality for all agents
"""

from typing import Dict, Any, Optional
from datetime import datetime, timezone
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """
    Abstract base class for all agents in the system
    """
    
    def __init__(self, name: str):
        """
        Initialize base agent
        
        Args:
            name: Name of the agent
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    @abstractmethod
    async def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the agent's task
        
        Args:
            state: Current state dictionary
            
        Returns:
            Updated state dictionary
        """
        pass
    
    def create_step_record(
        self,
        status: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a standardized agent step record
        
        Args:
            status: Status of the step (running, completed, failed)
            message: Human-readable message
            data: Optional data payload
            
        Returns:
            Formatted step record
        """
        return {
            "agent_name": self.name,
            "status": status,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {}
        }
    
    def log_execution(self, message: str, level: str = "info"):
        """
        Log agent execution with appropriate level
        
        Args:
            message: Log message
            level: Log level (info, warning, error)
        """
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(f"[{self.name}] {message}")
    
    async def handle_error(self, error: Exception, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle errors gracefully
        
        Args:
            error: The exception that occurred
            state: Current state
            
        Returns:
            Updated state with error information
        """
        error_message = f"Error in {self.name}: {str(error)}"
        self.log_execution(error_message, "error")
        
        # Add error step to agent steps
        if "agent_steps" not in state:
            state["agent_steps"] = []
        
        state["agent_steps"].append(
            self.create_step_record(
                status="failed",
                message=error_message,
                data={"error_type": type(error).__name__}
            )
        )
        
        # Mark state as having errors
        state["has_errors"] = True
        state["last_error"] = error_message
        
        return state

