import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentType(Enum):
    ANALYSIS = "analysis"
    ASSISTANT = "assistant"
    CHRONOBIOTICS = "chronobiotics"
    CITATION = "citation"
    DATA = "data"
    MULTILINGUAL = "multilingual"
    RESEARCH = "research"
    VOICE = "voice"


class AgentPriority(Enum):
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class AgentContext:
    """Контекст выполнения агента"""
    session_id: str
    user_id: Optional[str] = None
    language: str = "ru"
    preferences: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentTask:
    """Задача для агента"""
    id: str
    type: str
    data: Dict[str, Any]
    priority: AgentPriority = AgentPriority.NORMAL
    context: Optional[AgentContext] = None
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResult:
    """Результат работы агента"""
    task_id: str
    success: bool
    data: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0


class BaseAgent(ABC):
    """Базовый класс для всех агентов"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = self.__class__.__name__
        self.agent_type = AgentType.ASSISTANT
        self.is_initialized = False
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Инициализация агента"""
        try:
            await self._setup_resources()
            self.is_initialized = True
            logger.info(f"Agent {self.name} initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize {self.name}: {e}")
            raise
    
    async def _setup_resources(self) -> None:
        """Настройка ресурсов (переопределяется в наследниках)"""
        pass
    
    @abstractmethod
    async def process(self, task: AgentTask) -> AgentResult:
        """Основной метод обработки задачи"""
        pass
    
    @abstractmethod
    async def validate(self, task: AgentTask) -> bool:
        """Валидация задачи"""
        pass
    
    async def shutdown(self) -> None:
        """Остановка агента"""
        self.is_initialized = False
        logger.info(f"Agent {self.name} shut down")
