"""
Extended Base Agent with common functionality for all agents
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class AgentCapability(Enum):
    """Agent capabilities for capability-based routing"""
    CHEMICAL_ANALYSIS = "chemical_analysis"
    PROPERTY_PREDICTION = "property_prediction"
    TOXICITY_ESTIMATION = "toxicity_estimation"
    INTERACTION_ANALYSIS = "interaction_analysis"
    SIMILARITY_SEARCH = "similarity_search"
    LITERATURE_SEARCH = "literature_search"
    CITATION_MANAGEMENT = "citation_management"
    TRANSLATION = "translation"
    SPEECH_RECOGNITION = "speech_recognition"
    SPEECH_SYNTHESIS = "speech_synthesis"
    KNOWLEDGE_GRAPH = "knowledge_graph"
    CLINICAL_DATA = "clinical_data"
    MECHANISM_ANALYSIS = "mechanism_analysis"


@dataclass
class ProcessingContext:
    """Context for agent processing"""
    request_id: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    language: str = "en"
    confidence_threshold: float = 0.7
    max_tokens: int = 2000
    temperature: float = 0.7
    
    def to_dict(self) -> Dict:
        return {
            "request_id": self.request_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "language": self.language,
            "confidence_threshold": self.confidence_threshold,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }


class BaseAgentImplementation(BaseAgent):
    """
    Extended base agent with common utilities
    """
    
    def __init__(self, name: str, role: AgentRole, config: Dict[str, Any] = None):
        super().__init__(name, role, config)
        self.capabilities = set()
        self._cache = {}
        self._rate_limiter = None
    
    def add_capability(self, capability: AgentCapability):
        """Add a capability to this agent"""
        self.capabilities.add(capability)
    
    def has_capability(self, capability: AgentCapability) -> bool:
        """Check if agent has a specific capability"""
        return capability in self.capabilities
    
    async def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get from cache with TTL"""
        if key in self._cache:
            data, expiry = self._cache[key]
            if datetime.now() < expiry:
                return data
            else:
                del self._cache[key]
        return None
    
    async def _set_in_cache(self, key: str, value: Any, ttl_seconds: int = 3600):
        """Set cache with TTL"""
        expiry = datetime.now().replace(microsecond=0) + __import__('datetime').timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expiry)
    
    def _generate_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _call_llm(self, prompt: str, context: ProcessingContext) -> str:
        """Call LLM with prompt - to be implemented with actual LLM integration"""
        # This will be integrated with your LLM service
        # For now, return a placeholder
        return f"LLM Response to: {prompt[:100]}..."
