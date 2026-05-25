i  # chronobioticagent/main/agent/agent_manager.py
"""Agent Manager - Coordinates all agents"""

import asyncio
import logging
from collections import defaultdict
from typing import Dict, List, Optional

from .agent_core import BaseAgent, AgentStatus, AgentType, AgentMessage, AgentResult

logger = logging.getLogger(__name__)


class AgentManager:
    """Central manager for all agents"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_registry: Dict[AgentType, List[str]] = defaultdict(list)
        self.message_bus: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._worker_task = None
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the manager"""
        self.agents[agent.name] = agent
        self.agent_registry[agent.agent_type].append(agent.name)
        logger.info(f"Registered agent: {agent.name} ({agent.agent_type.value})")
    
    async def start_all(self):
        """Start all registered agents"""
        self._running = True
        self._worker_task = asyncio.create_task(self._message_worker())
        
        for agent in self.agents.values():
            await agent.start()
        
        logger.info(f"AgentManager started with {len(self.agents)} agents")
    
    async def stop_all(self):
        """Stop all agents"""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
        
        for agent in self.agents.values():
            await agent.stop()
        
        logger.info("AgentManager stopped")
    
    async def send_to_agent(
            self,
            agent_name: str,
            message: AgentMessage,
            timeout_seconds: float = 30.0
    ) -> Optional[AgentResult]:
        """Send message to specific agent"""
        if agent_name not in self.agents:
            raise ValueError(f"Agent {agent_name} not found")
        
        agent = self.agents[agent_name]
        if agent.status != AgentStatus.RUNNING:
            raise RuntimeError(f"Agent {agent_name} is not running")
        
        # Create future for response
        response_future = asyncio.Future()
        
        # Add to agent's queue
        message.metadata["response_future"] = response_future
        await agent._task_queue.put(message)
        
        # Wait for response with timeout
        try:
            result = await asyncio.wait_for(response_future, timeout_seconds)
            return result
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for response from {agent_name}")
            return None
    
    async def broadcast(
            self,
            message: AgentMessage,
            agent_type: Optional[AgentType] = None
    ):
        """Broadcast message to all agents or specific type"""
        targets = []
        if agent_type:
            targets = self.agent_registry.get(agent_type, [])
        else:
            targets = list(self.agents.keys())
        
        for agent_name in targets:
            await self.send_to_agent(agent_name, message)
    
    async def _message_worker(self):
        """Process messages on the message bus"""
        while self._running:
            try:
                message = await self.message_bus.get()
                # Route message to appropriate agent(s)
                if recipient := message.recipient:
                    await self.send_to_agent(recipient, message)
                else:
                    await self.broadcast(message)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Message worker error: {e}")
    
    def get_agent_status(self) -> Dict[str, str]:
        """Get status of all agents"""
        return {name: agent.status.value for name, agent in self.agents.items()}
    
    async def execute_pipeline(
            self,
            agents: List[str],
            initial_message: AgentMessage
    ) -> List[AgentResult]:
        """Execute agents in sequence (pipeline)"""
        results = []
        current_message = initial_message
        
        for agent_name in agents:
            result = await self.send_to_agent(agent_name, current_message)
            results.append(result)
            
            if not result or not result.success:
                break
            
            # Pass result as next message
            current_message = AgentMessage(
                sender=agent_name,
                content=result.data,
                correlation_id=initial_message.correlation_id
            )
        
        return results
    
    async def execute_parallel(
            self,
            agent_messages: List[tuple],
            max_concurrent: int = 5
    ) -> List[AgentResult]:
        """Execute multiple agent calls in parallel"""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_send(agent_name: str, message: AgentMessage):
            async with semaphore:
                return await self.send_to_agent(agent_name, message)
        
        tasks = [
            bounded_send(agent_name, message)
            for agent_name, message in agent_messages
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to AgentResult
        processed_results = []
        for r in results:
            if isinstance(r, Exception):
                processed_results.append(AgentResult(
                    success=False,
                    error=str(r)
                ))
            else:
                processed_results.append(r)
        
        return processed_results
