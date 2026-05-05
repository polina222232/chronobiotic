"""
Agent Core Module - Base classes for all agents
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional


class AgentType(Enum):
    """Agent types enumeration"""
    ASSISTANT = "assistant"
    ANALYST = "analyst"
    RESEARCHER = "researcher"
    CHEMICAL = "chemical"
    CITATION = "citation"
    VOICE = "voice"
    MULTIMODAL = "multimodal"


class AgentRole(Enum):
    """Agent roles"""
    CHAT = "chat"
    ANALYSIS = "analysis"
    SEARCH = "search"
    TRANSLATION = "translation"
    CITATION = "citation"
    VOICE = "voice"


@dataclass
class AgentContext:
    """Agent execution context"""
    conversation_id: str
    user_id: str
    language: str = "en"
    temperature: float = 0.7
    max_tokens: int = 2048
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Agent response structure"""
    success: bool
    content: str
    citations: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, agent_type: AgentType, config: Dict[str, Any] = None):
        self.name = name
        self.agent_type = agent_type
        self.config = config or {}
        self.is_running = False
    
    @abstractmethod
    async def process(self, message: str, context: AgentContext) -> AgentResponse:
        """Process a message and return response"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.is_running = True
    
    async def stop(self):
        """Stop the agent"""
        self.is_running = False
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "name": self.name,
            "type": self.agent_type.value,
            "is_running": self.is_running,
            "config": self.config
        }
