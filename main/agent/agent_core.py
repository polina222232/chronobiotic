
import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import uuid


# ========== Вспомогательные классы ==========
class AgentStatus(Enum):
    CREATED = "created"
    INITIALIZING = "initializing"
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"


class AgentType(Enum):
    RESEARCH = "research_agent"
    DATA = "data_agent"
    ANALYSIS = "analysis_agent"
    CHRONOBIOTICS = "chronobiotics_agent"


@dataclass
class Message:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    sender: str = ""
    recipient: str = ""
    message_type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=lambda: asyncio.get_event_loop().time())
    correlation_id: Optional[str] = None


@dataclass
class AgentConfig:
    agent_id: str
    agent_type: AgentType
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    max_retries: int = 3
    retry_delay: float = 1.0


# ========== Основной класс агента ==========
class BaseAgent:
    """Базовый класс для всех агентов в системе."""
    
    def __init__(self, agent_id: str, agent_type: AgentType):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.status = AgentStatus.CREATED
        self.message_queue = asyncio.Queue()
        self.handlers: Dict[str, Callable] = {}
        self.logger = logging.getLogger(f"Agent.{agent_type}.{agent_id}")
    
    async def initialize(self, core: 'AgentCore'):
        """Инициализация агента с ссылкой на ядро."""
        self.core = core
        self.status = AgentStatus.INITIALIZING
        self.logger.info(f"Initializing {self.agent_type} agent")
        await self._setup_handlers()
        self.status = AgentStatus.IDLE
    
    async def _setup_handlers(self):
        """Настройка обработчиков сообщений. Должен быть переопределен."""
        pass
    
    async def process_message(self, message: Message):
        """Основной цикл обработки сообщений."""
        try:
            self.status = AgentStatus.PROCESSING
            handler = self.handlers.get(message.message_type)
            if handler:
                result = await handler(message)
                await self._send_result(message, result)
            else:
                self.logger.warning(f"No handler for message type: {message.message_type}")
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            await self._handle_error(message, e)
        finally:
            self.status = AgentStatus.IDLE
    
    async def _send_result(self, original_message: Message, result: Any):
        """Отправка результата обработки."""
        response = Message(
            sender=self.agent_id,
            recipient=original_message.sender,
            message_type=f"{original_message.message_type}_response",
            payload={"result": result, "status": "success"},
            correlation_id=original_message.id
        )
        await self.core.route_message(response)
    
    async def _handle_error(self, message: Message, error: Exception):
        """Обработка ошибок с политикой повторных попыток."""
        error_message = Message(
            sender=self.agent_id,
            recipient=message.sender,
            message_type="error",
            payload={"error": str(error), "original_message": message.payload},
            correlation_id=message.id
        )
        await self.core.route_message(error_message)


# ========== Ядро-оркестратор ==========
class AgentCore:
    """Центральный координатор всех агентов."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.message_bus = asyncio.Queue()
        self.running = False
        self.logger = logging.getLogger("AgentCore")
        
        # Регистрация фабрик агентов
        self.agent_factories: Dict[AgentType, Callable] = {}
    
    def register_agent_factory(self, agent_type: AgentType, factory: Callable):
        """Регистрация фабрики для создания агентов определенного типа."""
        self.agent_factories[agent_type] = factory
    
    async def initialize(self):
        """Инициализация ядра и создание агентов из конфигурации."""
        self.logger.info("Initializing Agent Core")
        
        # Создание агентов в порядке зависимостей (топологическая сортировка)
        for agent_id, config in self.agent_configs.items():
            if config.agent_type in self.agent_factories:
                agent = self.agent_factories[config.agent_type](agent_id)
                self.agents[agent_id] = agent
                await agent.initialize(self)
        
        # Запуск обработки сообщений
        self.running = True
        asyncio.create_task(self._message_dispatcher())
    
    async def _message_dispatcher(self):
        """Основной диспетчер сообщений."""
        while self.running:
            try:
                message = await self.message_bus.get()
                
                # Маршрутизация сообщения
                if message.recipient in self.agents:
                    agent = self.agents[message.recipient]
                    if agent.status != AgentStatus.ERROR:
                        asyncio.create_task(agent.process_message(message))
                    else:
                        self.logger.error(f"Agent {message.recipient} is in error state")
                else:
                    self.logger.warning(f"No agent found for recipient: {message.recipient}")
            
            except Exception as e:
                self.logger.error(f"Error in message dispatcher: {e}")
    
    async def route_message(self, message: Message):
        """Публикация сообщения в шину."""
        await self.message_bus.put(message)
    
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Получение статуса агента."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            return {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "status": agent.status.value,
                "queue_size": agent.message_queue.qsize() if hasattr(agent, 'message_queue') else 0
            }
        return {"error": "Agent not found"}
    
    async def shutdown(self):
        """Корректное завершение работы всех агентов."""
        self.logger.info("Shutting down Agent Core")
        self.running = False
        
        for agent in self.agents.values():
            agent.status = AgentStatus.TERMINATED