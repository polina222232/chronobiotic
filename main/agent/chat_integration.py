"""
Integration module for chat system with Chronobiotics Agent
"""

import asyncio
from typing import Optional, Dict, Any
from datetime import datetime

from .chat.chat_engine import ChatEngine
from .chat.conversation_manager import ConversationManager
from .chat.message_handler import MessageHandler
from .chat.response_builder import ResponseBuilder
from .chat.multilingual_chat_engine import MultilingualChatEngine
from .chat.voice_chat_engine import VoiceChatEngine
from .chat.session_manager import SessionManager


class ChronobioticsChatSystem:
    """
    Complete chat system integration for Chronobiotics Agent
    """
    
    def __init__(
            self,
            agent_manager=None,
            llm_service=None,
            enable_multilingual: bool = True,
            enable_voice: bool = True,
            enable_persistence: bool = True
    ):
        self.agent_manager = agent_manager
        self.llm_service = llm_service
        
        # Initialize components
        self.session_manager = SessionManager() if enable_persistence else None
        self.conversation_manager = ConversationManager()
        self.message_handler = MessageHandler()
        self.response_builder = ResponseBuilder()
        
        # Initialize chat engine
        self.chat_engine = ChatEngine(
            max_session_messages=100,
            cache_enabled=True,
            rate_limit_enabled=True
        )
        
        # Add middleware for agent integration
        self.chat_engine.add_middleware(self._agent_middleware)
        self.chat_engine.add_middleware(self._citation_middleware)
        
        # Initialize multilingual engine
        if enable_multilingual:
            self.chat_engine = MultilingualChatEngine(
                base_engine=self.chat_engine,
                auto_detect=True,
                auto_translate=False,
                default_language="en"
            )
        
        # Initialize voice engine
        if enable_voice:
            self.voice_engine = VoiceChatEngine(
                base_engine=self.chat_engine,
                default_language="en"
            )
        else:
            self.voice_engine = None
        
        # Metrics
        self.metrics = {
            "messages_processed": 0,
            "sessions_created": 0,
            "avg_response_time": 0,
            "total_response_time": 0
        }
    
    async def _agent_middleware(self, message: str, session) -> str:
        """
        Middleware to process message through agent system
        """
        if not self.agent_manager:
            return message
        
        try:
            # Route to appropriate agent based on message type
            intent = await self._detect_intent(message)
            
            if intent == "chemical":
                result = await self.agent_manager.run_agent(
                    "chemical_analyzer",
                    {"query": message}
                )
                if result:
                    message = f"[Chemical Analysis]\n{result}"
            
            elif intent == "literature":
                result = await self.agent_manager.run_agent(
                    "literature_miner",
                    {"query": message}
                )
                if result:
                    message = f"[Literature Search]\n{result}"
            
            elif intent == "mechanism":
                result = await self.agent_manager.run_agent(
                    "mechanism_researcher",
                    {"query": message}
                )
                if result:
                    message = f"[Mechanism Analysis]\n{result}"
            
            elif intent == "clinical":
                result = await self.agent_manager.run_agent(
                    "clinical_data_finder",
                    {"query": message}
                )
                if result:
                    message = f"[Clinical Data]\n{result}"
        
        except Exception as e:
            logger.error(f"Agent middleware error: {e}")
        
        return message
    
    async def _citation_middleware(self, message: str, session) -> str:
        """
        Middleware to add citations to responses
        """
        # Extract potential citations from message
        citations = self._extract_citations(message)
        
        if citations:
            # Store citations in session context
            session.context["citations"] = citations
        
        return message
    
    async def _detect_intent(self, message: str) -> str:
        """Detect user intent"""
        message_lower = message.lower()
        
        if any(kw in message_lower for kw in ["smiles", "molecule", "structure", "property"]):
            return "chemical"
        elif any(kw in message_lower for kw in ["paper", "article", "study", "literature"]):
            return "literature"
        elif any(kw in message_lower for kw in ["mechanism", "pathway", "how does"]):
            return "mechanism"
        elif any(kw in message_lower for kw in ["trial", "clinical", "patient"]):
            return "clinical"
        
        return "general"
    
    def _extract_citations(self, text: str) -> list:
        """Extract citations from text"""
        import re
        citations = []
        
        # Look for DOI patterns
        doi_pattern = r'10\.\d{4,9}/[-._;()/:A-Z0-9]+'
        dois = re.findall(doi_pattern, text, re.IGNORECASE)
        
        for doi in dois:
            citations.append({
                "type": "doi",
                "identifier": doi,
                "url": f"https://doi.org/{doi}"
            })
        
        return citations
    
    async def send_message(
            self,
            session_id: str,
            message: str,
            user_id: Optional[str] = None,
            user_name: Optional[str] = None,
            language: str = "en",
            **kwargs
    ) -> Dict[str, Any]:
        """
        Send a message to the chat system

        Returns:
            Dict with response and metadata
        """
        start_time = datetime.now()
        
        # Create or get session
        session = self.session_manager.get_session(session_id) if self.session_manager else None
        if not session and self.session_manager:
            session = self.session_manager.create_session(
                session_id=session_id,
                user_id=user_id,
                user_name=user_name,
                language=language
            )
        
        # Process message
        response = await self.chat_engine.process_message(
            session_id=session_id,
            message=message,
            user_id=user_id,
            metadata=kwargs
        )
        
        # Update metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.metrics["messages_processed"] += 1
        self.metrics["total_response_time"] += processing_time
        self.metrics["avg_response_time"] = (
                self.metrics["total_response_time"] / self.metrics["messages_processed"]
        )
        
        # Format response
        formatted = self.response_builder.build_response(
            content=response.content,
            citations=getattr(response, "citations", []),
            sources=getattr(response, "sources", [])
        )
        
        return {
            "success": True,
            "response": formatted,
            "processing_time": processing_time,
            "session_id": session_id,
            "language": language
        }
    
    async def stream_message(
            self,
            session_id: str,
            message: str,
            user_id: Optional[str] = None
    ):
        """
        Stream a message response
        """
        async for chunk in self.chat_engine.stream_message(
                session_id=session_id,
                message=message,
                user_id=user_id
        ):
            yield chunk
    
    async def process_voice(
            self,
            session_id: str,
            audio_data: bytes,
            user_id: Optional[str] = None,
            language: str = "en"
    ) -> Dict[str, Any]:
        """
        Process voice message
        """
        if not self.voice_engine:
            return {
                "success": False,
                "error": "Voice engine not enabled"
            }
        
        result = await self.voice_engine.process_voice_message(
            session_id=session_id,
            audio_data=audio_data,
            user_id=user_id,
            language=language
        )
        
        return result
    
    def get_session_history(
            self,
            session_id: str,
            limit: int = 50
    ) -> Dict[str, Any]:
        """
        Get session history
        """
        if self.session_manager:
            messages = self.session_manager.get_messages(session_id, limit=limit)
            stats = self.session_manager.get_session_stats(session_id)
            
            return {
                "session_id": session_id,
                "messages": messages,
                "stats": stats
            }
        
        # Fallback to conversation manager
        history = self.conversation_manager.get_conversation_history(session_id, limit)
        
        return {
            "session_id": session_id,
            "messages": history,
            "stats": {}
        }
    
    def clear_session(self, session_id: str) -> bool:
        """
        Clear session data
        """
        if self.session_manager:
            return self.session_manager.delete_session(session_id)
        
        self.conversation_manager.clear_conversation(session_id)
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system metrics
        """
        metrics = dict(self.metrics)
        
        if self.session_manager:
            metrics["sessions_count"] = len(self.session_manager.get_active_sessions())
        
        return metrics
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check
        """
        return {
            "status": "healthy",
            "chat_engine": "online",
            "multilingual": isinstance(self.chat_engine, MultilingualChatEngine),
            "voice_enabled": self.voice_engine is not None,
            "persistence_enabled": self.session_manager is not None,
            "metrics": self.get_metrics(),
            "timestamp": datetime.now().isoformat()
        }
