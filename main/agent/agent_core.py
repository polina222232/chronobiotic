# chronobioticagent/main/agent/agent_core.py
"""Core agent base classes and interfaces"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from uuid import uuid4

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    DEGRADED = "degraded"


class AgentType(Enum):
    """Types of agents in the system"""
    ANALYSIS = "analysis"
    RESEARCH = "research"
    MULTILINGUAL = "multilingual"
    VOICE = "voice"
    MULTIMODAL = "multimodal"
    CITATION = "citation"
    ORCHESTRATOR = "orchestrator"


@dataclass
class AgentMessage:
    """Message exchanged between agents"""
    id: str = field(default_factory=lambda: str(uuid4()))
    sender: str = ""
    recipient: str = ""
    content: Any = None
    message_type: str = "text"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    correlation_id: Optional[str] = None


@dataclass
class AgentResult:
    """Result returned by an agent"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    confidence: float = 0.0
    citations: List[Dict] = field(default_factory=list)
    processing_time_ms: float = 0.0
    agent_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(
            self,
            name: str,
            agent_type: AgentType,
            config: Optional[Dict] = None
    ):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.status = AgentStatus.INITIALIZED
        self._message_handlers: Dict[str, Callable] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task = None
    
    @abstractmethod
    async def process(self, message: AgentMessage) -> AgentResult:
        """Process an incoming message and return result"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.status = AgentStatus.RUNNING
        self._running = True
        self._task = asyncio.create_task(self._run())
        logger.info(f"Agent {self.name} started")
    
    async def stop(self):
        """Stop the agent"""
        self._running = False
        if self._task:
            self._task.cancel()
        self.status = AgentStatus.STOPPED
        logger.info(f"Agent {self.name} stopped")
    
    async def _run(self):
        """Main agent loop"""
        while self._running:
            try:
                message = await self._task_queue.get()
                result = await self.process(message)
                if callback := message.metadata.get("callback"):
                    await callback(result)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Agent {self.name} error: {e}")
                self.status = AgentStatus.ERROR
    
    async def send_message(self, message: AgentMessage) -> AgentResult:
        """Send message to another agent via message bus"""
        pass
    
    def register_handler(self, message_type: str, handler: Callable):
        """Register a handler for specific message type"""
        self._message_handlers[message_type] = handler
