import logging
from typing import Dict, Any, Optional, Type

from .agent_core import BaseAgent, AgentContext, AgentResponse

logger = logging.getLogger(__name__)


class AgentManager:
    """Central manager for all agents"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._agents: Dict[str, BaseAgent] = {}
        self._agent_types: Dict[str, Type[BaseAgent]] = {}
        self._initialized = True
        logger.info("AgentManager initialized")
    
    def register_agent_type(self, name: str, agent_class: Type[BaseAgent]):
        self._agent_types[name] = agent_class
        logger.info(f"Registered agent type: {name}")
    
    async def create_agent(self, name: str, agent_type: str, config: Optional[Dict] = None) -> BaseAgent:
        if name in self._agents:
            raise ValueError(f"Agent {name} already exists")
        
        if agent_type not in self._agent_types:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = self._agent_types[agent_type]
        agent = agent_class(name, config or {})
        
        success = await agent.initialize()
        if not success:
            raise RuntimeError(f"Failed to initialize agent {name}")
        
        self._agents[name] = agent
        logger.info(f"Created agent: {name}")
        return agent
    
    async def get_agent(self, name: str) -> Optional[BaseAgent]:
        return self._agents.get(name)
    
    async def send_request(self, agent_name: str, request: Any,
                           context: Optional[AgentContext] = None) -> AgentResponse:
        agent = self._agents.get(agent_name)
        if not agent:
            return AgentResponse(
                success=False,
                error=f"Agent {agent_name} not found",
                agent_name="AgentManager"
            )
        return await agent.process(request, context)
    
    async def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        return {name: agent.get_status() for name, agent in self._agents.items()}
    
    async def shutdown_all(self):
        logger.info("Shutting down all agents...")
        self._agents.clear()
        logger.info("All agents shutdown complete")
