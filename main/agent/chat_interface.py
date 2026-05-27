"""
Chat Interface for Chronobiotics Agent System
Provides multi-modal chat capabilities with agent orchestration
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"
    ASSISTANT = "assistant"


class ChatStatus(Enum):
    ACTIVE = "active"
    PROCESSING = "processing"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    language: str = "en"
    confidence: float = 1.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "citations": self.citations,
            "sources": self.sources,
            "language": self.language,
            "confidence": self.confidence
        }


@dataclass
class ChatSession:
    """Manages a chat session"""
    session_id: str
    user_id: Optional[str]
    messages: List[ChatMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: ChatStatus = ChatStatus.ACTIVE
    context: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    
    def add_message(self, message: ChatMessage):
        self.messages.append(message)
        self.updated_at = datetime.now()
    
    def get_recent_messages(self, limit: int = 10) -> List[ChatMessage]:
        return self.messages[-limit:]
    
    def get_conversation_context(self, max_tokens: int = 4000) -> str:
        """Build conversation context for LLM"""
        context_parts = []
        total_length = 0
        
        for msg in self.messages[-10:]:  # Last 10 messages
            role_prefix = "User:" if msg.role == MessageRole.USER else "Assistant:"
            msg_text = f"{role_prefix} {msg.content}\n"
            
            if total_length + len(msg_text) > max_tokens:
                break
            
            context_parts.append(msg_text)
            total_length += len(msg_text)
        
        return "\n".join(context_parts)


class ChatInterface:
    """
    Main chat interface for Chronobiotics Agent System
    Handles message processing, agent orchestration, and response generation
    """
    
    def __init__(self, agent_manager=None, llm_service=None):
        self.agent_manager = agent_manager
        self.llm_service = llm_service
        self.sessions: Dict[str, ChatSession] = {}
        self.message_handlers = []
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup message handlers"""
        self.message_handlers = [
            self._handle_chemical_query,
            self._handle_literature_search,
            self._handle_mechanism_query,
            self._handle_clinical_data,
            self._handle_general_chat,
        ]
    
    async def process_message(
            self,
            message: str,
            session_id: str,
            user_id: Optional[str] = None,
            attachments: Optional[List[Dict]] = None,
            language: str = "en"
    ) -> ChatMessage:
        """
        Process incoming message and generate response

        Args:
            message: User message text
            session_id: Chat session identifier
            user_id: Optional user identifier
            attachments: Optional file attachments
            language: Response language preference

        Returns:
            ChatMessage: Agent response
        """
        try:
            # Get or create session
            session = self._get_or_create_session(session_id, user_id, language)
            
            # Create user message
            user_message = ChatMessage(
                id=self._generate_message_id(),
                role=MessageRole.USER,
                content=message,
                timestamp=datetime.now(),
                metadata={"attachments": attachments} if attachments else {},
                language=language
            )
            session.add_message(user_message)
            
            # Update session status
            session.status = ChatStatus.PROCESSING
            
            # Process through agent system
            response_content, metadata = await self._orchestrate_response(
                message, session, attachments
            )
            
            # Create agent response
            agent_message = ChatMessage(
                id=self._generate_message_id(),
                role=MessageRole.AGENT,
                content=response_content,
                timestamp=datetime.now(),
                metadata=metadata.get("metadata", {}),
                citations=metadata.get("citations", []),
                sources=metadata.get("sources", []),
                language=language,
                confidence=metadata.get("confidence", 0.95)
            )
            session.add_message(agent_message)
            
            # Reset session status
            session.status = ChatStatus.ACTIVE
            
            return agent_message
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            session.status = ChatStatus.ERROR
            return self._create_error_message(str(e), language)
    
    async def _orchestrate_response(
            self,
            message: str,
            session: ChatSession,
            attachments: Optional[List[Dict]] = None
    ) -> tuple[str, Dict]:
        """
        Orchestrate response using multiple agents

        Returns:
            tuple: (response_content, metadata)
        """
        # Detect intent and route to appropriate handlers
        intent = await self._detect_intent(message)
        
        # Parallel execution of relevant agents
        tasks = []
        
        if intent == "chemical":
            tasks.append(self._query_chemical_agents(message))
        if intent == "literature":
            tasks.append(self._query_literature_agents(message))
        if intent == "mechanism":
            tasks.append(self._query_mechanism_agents(message))
        if intent == "clinical":
            tasks.append(self._query_clinical_agents(message))
        
        # Always include general chat
        tasks.append(self._query_chat_agent(message, session))
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Aggregate and format response
        return await self._aggregate_responses(results, intent, session.language)
    
    async def _detect_intent(self, message: str) -> str:
        """Detect user intent from message"""
        message_lower = message.lower()
        
        # Chemical intent
        chemical_keywords = ["smiles", "molecule", "compound", "chemical",
                             "structure", "property", "toxicity", "admet"]
        if any(kw in message_lower for kw in chemical_keywords):
            return "chemical"
        
        # Literature intent
        literature_keywords = ["paper", "article", "study", "research",
                               "publication", "cite", "reference"]
        if any(kw in message_lower for kw in literature_keywords):
            return "literature"
        
        # Mechanism intent
        mechanism_keywords = ["mechanism", "pathway", "how does", "why does",
                              "target", "receptor", "signaling"]
        if any(kw in message_lower for kw in mechanism_keywords):
            return "mechanism"
        
        # Clinical intent
        clinical_keywords = ["trial", "clinical", "patient", "dose",
                             "treatment", "therapy", "efficacy"]
        if any(kw in message_lower for kw in clinical_keywords):
            return "clinical"
        
        return "general"
    
    async def _query_chemical_agents(self, message: str) -> Dict:
        """Query chemical analysis agents"""
        if not self.agent_manager:
            return {"type": "chemical", "content": "Chemical analysis not available"}
        
        try:
            # Parallel chemical queries
            tasks = [
                self.agent_manager.run_agent("chemical_analyzer", {"query": message}),
                self.agent_manager.run_agent("property_predictor", {"query": message}),
                self.agent_manager.run_agent("toxicity_estimator", {"query": message}),
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            return {
                "type": "chemical",
                "analysis": results[0] if not isinstance(results[0], Exception) else None,
                "properties": results[1] if not isinstance(results[1], Exception) else None,
                "toxicity": results[2] if not isinstance(results[2], Exception) else None,
            }
        except Exception as e:
            logger.error(f"Chemical query failed: {e}")
            return {"type": "chemical", "error": str(e)}
    
    async def _query_literature_agents(self, message: str) -> Dict:
        """Query literature mining agents"""
        if not self.agent_manager:
            return {"type": "literature", "content": "Literature search not available"}
        
        try:
            results = await self.agent_manager.run_agent(
                "literature_miner",
                {"query": message, "max_sources": 5}
            )
            return {"type": "literature", "results": results}
        except Exception as e:
            logger.error(f"Literature query failed: {e}")
            return {"type": "literature", "error": str(e)}
    
    async def _query_mechanism_agents(self, message: str) -> Dict:
        """Query mechanism research agents"""
        if not self.agent_manager:
            return {"type": "mechanism", "content": "Mechanism analysis not available"}
        
        try:
            results = await self.agent_manager.run_agent(
                "mechanism_researcher",
                {"query": message}
            )
            return {"type": "mechanism", "results": results}
        except Exception as e:
            logger.error(f"Mechanism query failed: {e}")
            return {"type": "mechanism", "error": str(e)}
    
    async def _query_clinical_agents(self, message: str) -> Dict:
        """Query clinical data agents"""
        if not self.agent_manager:
            return {"type": "clinical", "content": "Clinical data not available"}
        
        try:
            results = await self.agent_manager.run_agent(
                "clinical_data_finder",
                {"query": message}
            )
            return {"type": "clinical", "results": results}
        except Exception as e:
            logger.error(f"Clinical query failed: {e}")
            return {"type": "clinical", "error": str(e)}
    
    async def _query_chat_agent(self, message: str, session: ChatSession) -> Dict:
        """Query general chat agent with context"""
        if self.llm_service:
            context = session.get_conversation_context()
            response = await self.llm_service.generate(
                prompt=message,
                context=context,
                temperature=0.7,
                max_tokens=500
            )
            return {"type": "chat", "response": response}
        elif self.agent_manager:
            response = await self.agent_manager.run_agent(
                "chat_agent",
                {"message": message, "context": session.get_conversation_context()}
            )
            return {"type": "chat", "response": response}
        else:
            return {"type": "chat", "response": "I'm here to help with chronobiotics research!"}
    
    async def _aggregate_responses(
            self,
            results: List[Any],
            intent: str,
            language: str
    ) -> tuple[str, Dict]:
        """Aggregate and format responses from multiple agents"""
        
        # Filter out exceptions and None values
        valid_results = [r for r in results if r and not isinstance(r, Exception)]
        
        # Build response content
        response_parts = []
        metadata = {
            "sources": [],
            "citations": [],
            "confidence": 0.0,
            "intent": intent
        }
        
        for result in valid_results:
            if result.get("type") == "chat":
                response_parts.append(result.get("response", ""))
            
            elif result.get("type") == "chemical":
                if result.get("analysis"):
                    response_parts.append(self._format_chemical_analysis(result["analysis"]))
                    metadata["confidence"] += 0.3
            
            elif result.get("type") == "literature":
                if result.get("results"):
                    response_parts.append(self._format_literature_results(result["results"]))
                    metadata["sources"].extend(result["results"].get("sources", []))
                    metadata["citations"].extend(result["results"].get("citations", []))
            
            elif result.get("type") == "mechanism":
                if result.get("results"):
                    response_parts.append(self._format_mechanism_results(result["results"]))
            
            elif result.get("type") == "clinical":
                if result.get("results"):
                    response_parts.append(self._format_clinical_results(result["results"]))
        
        # Combine response parts
        if not response_parts:
            response_parts = [
                "I'm processing your request. Please provide more specific information about chronobiotics research."]
        
        response_text = "\n\n".join(response_parts)
        metadata["confidence"] = min(metadata["confidence"] / len(valid_results) if valid_results else 0.5, 1.0)
        
        return response_text, metadata
    
    def _format_chemical_analysis(self, analysis: Dict) -> str:
        """Format chemical analysis results"""
        if not analysis:
            return ""
        
        parts = ["**Chemical Analysis:**"]
        
        if "properties" in analysis:
            parts.append(f"- Properties: {analysis['properties']}")
        if "toxicity" in analysis:
            parts.append(f"- Toxicity Profile: {analysis['toxicity']}")
        if "recommendations" in analysis:
            parts.append(f"- Recommendations: {analysis['recommendations']}")
        
        return "\n".join(parts)
    
    def _format_literature_results(self, results: Dict) -> str:
        """Format literature search results"""
        if not results:
            return ""
        
        parts = ["**Relevant Literature:**"]
        
        sources = results.get("sources", [])
        for i, source in enumerate(sources[:3], 1):
            parts.append(
                f"{i}. {source.get('title', 'Untitled')} - {source.get('authors', 'Unknown')} ({source.get('year', 'n.d.')})")
        
        return "\n".join(parts)
    
    def _format_mechanism_results(self, results: Dict) -> str:
        """Format mechanism research results"""
        if not results:
            return ""
        
        parts = ["**Mechanism Insights:**"]
        
        mechanisms = results.get("mechanisms", [])
        for mechanism in mechanisms[:3]:
            parts.append(f"- {mechanism}")
        
        return "\n".join(parts)
    
    def _format_clinical_results(self, results: Dict) -> str:
        """Format clinical data results"""
        if not results:
            return ""
        
        parts = ["**Clinical Information:**"]
        
        trials = results.get("clinical_trials", [])
        for trial in trials[:2]:
            parts.append(
                f"- {trial.get('name', 'Study')}: {trial.get('phase', 'N/A')} - {trial.get('status', 'Unknown')}")
        
        return "\n".join(parts)
    
    def _get_or_create_session(
            self,
            session_id: str,
            user_id: Optional[str],
            language: str
    ) -> ChatSession:
        """Get existing session or create new one"""
        if session_id in self.sessions:
            return self.sessions[session_id]
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            language=language
        )
        self.sessions[session_id] = session
        return session
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _create_error_message(self, error_msg: str, language: str) -> ChatMessage:
        """Create error response message"""
        return ChatMessage(
            id=self._generate_message_id(),
            role=MessageRole.SYSTEM,
            content=f"⚠️ Error: {error_msg}\nPlease try again or rephrase your question.",
            timestamp=datetime.now(),
            language=language,
            confidence=0.0
        )
    
    def get_session_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get chat session history"""
        session = self.sessions.get(session_id)
        if not session:
            return []
        
        return [msg.to_dict() for msg in session.messages[-limit:]]
    
    def clear_session(self, session_id: str):
        """Clear a chat session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    async def stream_response(
            self,
            message: str,
            session_id: str,
            user_id: Optional[str] = None
    ):
        """
        Stream response token by token for real-time chat
        """
        session = self._get_or_create_session(session_id, user_id, "en")
        session.status = ChatStatus.PROCESSING
        
        try:
            # Process through LLM with streaming
            if self.llm_service:
                async for token in self.llm_service.stream_generate(
                        prompt=message,
                        context=session.get_conversation_context()
                ):
                    yield token
            else:
                # Fallback to batch response
                response = await self.process_message(message, session_id, user_id)
                yield response.content
        
        finally:
            session.status = ChatStatus.ACTIVE


# WebSocket consumer for real-time chat
class ChatConsumer:
    """WebSocket consumer for real-time chat"""
    
    def __init__(self, chat_interface: ChatInterface):
        self.chat_interface = chat_interface
        self.active_sessions = {}
    
    async def connect(self, session_id: str, websocket):
        """Handle WebSocket connection"""
        self.active_sessions[session_id] = websocket
    
    async def disconnect(self, session_id: str):
        """Handle WebSocket disconnection"""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
    
    async def handle_message(self, session_id: str, message_data: Dict):
        """Handle incoming WebSocket message"""
        websocket = self.active_sessions.get(session_id)
        if not websocket:
            return
        
        try:
            # Process message
            response = await self.chat_interface.process_message(
                message=message_data.get("text", ""),
                session_id=session_id,
                user_id=message_data.get("user_id"),
                attachments=message_data.get("attachments"),
                language=message_data.get("language", "en")
            )
            
            # Send response
            await websocket.send_json({
                "type": "message",
                "data": response.to_dict()
            })
        
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
    
    async def stream_message(self, session_id: str, message: str):
        """Handle streaming message"""
        websocket = self.active_sessions.get(session_id)
        if not websocket:
            return
        
        try:
            # Send start signal
            await websocket.send_json({"type": "stream_start"})
            
            # Stream response
            async for chunk in self.chat_interface.stream_response(message, session_id):
                await websocket.send_json({
                    "type": "stream_chunk",
                    "content": chunk
                })
            
            # Send end signal
            await websocket.send_json({"type": "stream_end"})
        
        except Exception as e:
            await websocket.send_json({
                "type": "error",
                "error": str(e)
            })
