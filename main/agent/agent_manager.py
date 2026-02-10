import asyncio
from typing import Dict, List, Optional
from dataclasses import dataclass
from agent_core import AgentCore, BaseAgent, AgentType, AgentConfig, AgentStatus
import time
import statistics


@dataclass
class PoolConfig:
    min_instances: int = 1
    max_instances: int = 5
    target_load: float = 0.7  # Целевая загрузка (0-1)
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    idle_timeout: float = 300.0  # Время бездействия перед уничтожением (сек)


class AgentPool:
    """Пул однотипных агентов для горизонтального масштабирования."""
    
    def __init__(self, agent_type: AgentType, config: PoolConfig, factory: Callable):
        self.agent_type = agent_type
        self.config = config
        self.factory = factory
        self.instances: Dict[str, BaseAgent] = {}
        self.metrics: Dict[str, List[float]] = {}
        self.last_activity: Dict[str, float] = {}
    
    async def get_agent(self) -> Optional[BaseAgent]:
        """Получение доступного агента с балансировкой нагрузки."""
        available = [a for a in self.instances.values()
                     if a.status == AgentStatus.IDLE]
        
        if available:
            # Простейшая балансировка — выбираем наименее нагруженного
            return min(available, key=lambda x: x.message_queue.qsize())
        return None
    
    async def scale_up(self, core: AgentCore):
        """Масштабирование пула вверх при необходимости."""
        if len(self.instances) < self.config.max_instances:
            agent_id = f"{self.agent_type.value}_{len(self.instances) + 1}"
            agent = self.factory(agent_id)
            self.instances[agent_id] = agent
            await agent.initialize(core)
            self.logger.info(f"Scaled up {self.agent_type}: {agent_id}")
    
    async def scale_down(self):
        """Масштабирование пула вниз при низкой нагрузке."""
        if len(self.instances) > self.config.min_instances:
            # Находим самый неактивный агент
            idle_agents = [(agent_id, self.last_activity.get(agent_id, 0))
                           for agent_id, agent in self.instances.items()
                           if agent.status == AgentStatus.IDLE]
            
            if idle_agents:
                oldest_agent_id, last_active = max(idle_agents, key=lambda x: x[1])
                current_time = time.time()
                
                if current_time - last_active > self.config.idle_timeout:
                    agent = self.instances.pop(oldest_agent_id)
                    agent.status = AgentStatus.TERMINATED
                    self.logger.info(f"Scaled down {self.agent_type}: {oldest_agent_id}")


class AgentManager(AgentCore):
    """Расширенное ядро с динамическим управлением агентами."""
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        self.pools: Dict[AgentType, AgentPool] = {}
        self.version_map: Dict[str, str] = {}  # agent_id -> version
        self.a_b_tests: Dict[str, Dict] = {}
    
    def create_pool(self, agent_type: AgentType, pool_config: PoolConfig):
        """Создание пула агентов определенного типа."""
        if agent_type in self.agent_factories:
            factory = self.agent_factories[agent_type]
            pool = AgentPool(agent_type, pool_config, factory)
            self.pools[agent_type] = pool
            return pool
    
    async def deploy_new_version(self, agent_type: AgentType, new_factory: Callable,
                                 rollout_percentage: float = 10.0):
        """Постепенный rollout новой версии агента."""
        self.logger.info(f"Starting rollout of new {agent_type} version")
        
        # A/B тестирование: создаем новые экземпляры с новой версией
        new_version_id = f"v{int(time.time())}"
        
        for agent_id in list(self.agents.keys()):
            if self.agents[agent_id].agent_type == agent_type:
                # Процент rollout определяет, какие агенты обновляются
                if hash(agent_id) % 100 < rollout_percentage:
                    await self._upgrade_agent(agent_id, new_factory, new_version_id)
    
    async def _upgrade_agent(self, agent_id: str, new_factory: Callable, version: str):
        """Обновление конкретного агента."""
        old_agent = self.agents[agent_id]
        
        # Создаем нового агента
        new_agent = new_factory(agent_id)
        await new_agent.initialize(self)
        
        # Заменяем в словаре
        self.agents[agent_id] = new_agent
        self.version_map[agent_id] = version
        
        # Корректно завершаем старого
        old_agent.status = AgentStatus.TERMINATED
    
    def get_system_health(self) -> Dict[str, Any]:
        """Полная картина здоровья системы."""
        health = {
            "total_agents": len(self.agents),
            "active_agents": sum(1 for a in self.agents.values()
                                 if a.status in [AgentStatus.IDLE, AgentStatus.PROCESSING]),
            "error_agents": sum(1 for a in self.agents.values()
                                if a.status == AgentStatus.ERROR),
            "pools": {}
        }
        
        for pool_type, pool in self.pools.items():
            health["pools"][pool_type.value] = {
                "instances": len(pool.instances),
                "idle": sum(1 for a in pool.instances.values()
                            if a.status == AgentStatus.IDLE),
                "processing": sum(1 for a in pool.instances.values()
                                  if a.status == AgentStatus.PROCESSING)
            }
        
        return health

