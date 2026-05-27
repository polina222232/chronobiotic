# main/agent/chat/chat_engine.py
"""
Core Chat Engine with advanced features
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Union
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


class ChatState(Enum):
    """Chat session states"""
    ACTIVE = "active"
    PROCESSING = "processing"
    PAUSED = "paused"
    IDLE = "idle"
    ERROR = "error"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class ChatMessage:
    """Represents a single chat message"""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    priority: MessagePriority = MessagePriority.NORMAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict] = field(default_factory=list)
    sources: List[Dict] = field(default_factory=list)
    language: str = "en"
    confidence: float = 1.0
    tokens_used: int = 0
    processing_time: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "metadata": self.metadata,
            "citations": self.citations,
            "sources": self.sources,
            "language": self.language,
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "processing_time": self.processing_time
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ChatMessage':
        return cls(
            id=data["id"],
            role=data["role"],
            content=data["content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=MessagePriority(data.get("priority", 1)),
            metadata=data.get("metadata", {}),
            citations=data.get("citations", []),
            sources=data.get("sources", []),
            language=data.get("language", "en"),
            confidence=data.get("confidence", 1.0),
            tokens_used=data.get("tokens_used", 0),
            processing_time=data.get("processing_time", 0.0)
        )


@dataclass
class ChatSession:
    """Manages a chat session"""
    session_id: str
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    state: ChatState = ChatState.ACTIVE
    messages: List[ChatMessage] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: ChatMessage):
        """Add message to session"""
        self.messages.append(message)
        self.last_active = datetime.now()
    
    def get_last_n_messages(self, n: int) -> List[ChatMessage]:
        """Get last N messages"""
        return self.messages[-n:] if n > 0 else []
    
    def get_conversation_length(self) -> int:
        """Get total conversation length in characters"""
        return sum(len(msg.content) for msg in self.messages)
    
    def get_message_count(self) -> int:
        """Get total message count"""
        return len(self.messages)
    
    def clear(self):
        """Clear all messages"""
        self.messages.clear()
        self.context.clear()
    
    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "state": self.state.value,
            "message_count": len(self.messages),
            "language": self.language,
            "metadata": self.metadata
        }


class ChatCache:
    """Cache system for chat responses"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: Dict[str, tuple[Any, datetime]] = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            value, expires = self.cache[key]
            if datetime.now() < expires:
                self.hits += 1
                return value
            else:
                del self.cache[key]
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest items
            oldest = min(self.cache.items(), key=lambda x: x[1][1])
            del self.cache[oldest[0]]
        
        self.cache[key] = (value, datetime.now() + timedelta(seconds=self.ttl))
    
    def clear(self):
        """Clear all cache"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total > 0 else 0
        }


class RateLimiter:
    """Rate limiting for chat requests"""
    
    def __init__(self):
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
    
    def is_allowed(self, user_id: str, limit: int = 60, window: int = 60) -> bool:
        """
        Check if request is allowed
        
        Args:
            user_id: User identifier
            limit: Maximum requests per window
            window: Time window in seconds
        """
        now = datetime.now()
        cutoff = now - timedelta(seconds=window)
        
        # Clean old requests
        self.requests[user_id] = [
            req_time for req_time in self.requests[user_id]
            if req_time > cutoff
        ]
        
        # Check limit
        if len(self.requests[user_id]) >= limit:
            return False
        
        # Add current request
        self.requests[user_id].append(now)
        return True
    
    def get_remaining(self, user_id: str, limit: int = 60, window: int = 60) -> int:
        """Get remaining requests allowed"""
        now = datetime.now()
        cutoff = now - timedelta(seconds=window)
        
        recent = [
            req_time for req_time in self.requests[user_id]
            if req_time > cutoff
        ]
        
        return max(0, limit - len(recent))


class ChatEngine:
    """
    Core Chat Engine for Chronobiotics Agent System
    Handles message processing, caching, rate limiting, and session management
    """
    
    def __init__(
            self,
            max_session_messages: int = 100,
            cache_enabled: bool = True,
            rate_limit_enabled: bool = True,
            default_language: str = "en"
    ):
        self.max_session_messages = max_session_messages
        self.cache_enabled = cache_enabled
        self.rate_limit_enabled = rate_limit_enabled
        self.default_language = default_language
        
        self.sessions: Dict[str, ChatSession] = {}
        self.cache = ChatCache() if cache_enabled else None
        self.rate_limiter = RateLimiter() if rate_limit_enabled else None
        
        self.metrics = defaultdict(int)
        self.message_handlers = []
        self.middleware = []
        
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup default message handlers"""
        self.message_handlers = [
            self._handle_text_message,
            self._handle_voice_message,
            self._handle_file_message,
            self._handle_command_message
        ]
    
    def add_middleware(self, middleware: Callable):
        """Add middleware to chat engine"""
        self.middleware.append(middleware)
    
    def get_or_create_session(
            self,
            session_id: str,
            user_id: Optional[str] = None,
            user_name: Optional[str] = None,
            language: Optional[str] = None
    ) -> ChatSession:
        """Get existing session or create new one"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_active = datetime.now()
            return session
        
        session = ChatSession(
            session_id=session_id,
            user_id=user_id,
            user_name=user_name,
            language=language or self.default_language
        )
        self.sessions[session_id] = session
        self.metrics["sessions_created"] += 1
        return session
    
    async def process_message(
            self,
            session_id: str,
            message: str,
            user_id: Optional[str] = None,
            message_type: str = "text",
            metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """
        Process a chat message
        
        Args:
            session_id: Session identifier
            message: Message content
            user_id: User identifier
            message_type: Type of message (text, voice, file, command)
            metadata: Additional metadata
        
        Returns:
            ChatMessage: Processed response
        """
        start_time = time.time()
        
        # Rate limiting check
        if self.rate_limit_enabled and user_id:
            if not self.rate_limiter.is_allowed(user_id):
                return self._create_error_message(
                    "Rate limit exceeded. Please wait before sending more messages."
                )
        
        # Get or create session
        session = self.get_or_create_session(session_id, user_id)
        
        # Create user message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            role="user",
            content=message,
            timestamp=datetime.now(),
            metadata=metadata or {},
            language=session.language
        )
        session.add_message(user_message)
        
        # Update session state
        previous_state = session.state
        session.state = ChatState.PROCESSING
        
        try:
            # Process through middleware
            processed_message = message
            for middleware in self.middleware:
                processed_message = await middleware(processed_message, session)
            
            # Handle based on message type
            response_message = None
            for handler in self.message_handlers:
                if await self._can_handle(handler, message_type, processed_message):
                    response_message = await handler(processed_message, session, metadata)
                    if response_message:
                        break
            
            if not response_message:
                response_message = await self._handle_default(processed_message, session)
            
            # Add response to session
            response_message.processing_time = time.time() - start_time
            session.add_message(response_message)
            
            # Trim session if needed
            if len(session.messages) > self.max_session_messages:
                session.messages = session.messages[-self.max_session_messages:]
            
            # Update metrics
            self.metrics["messages_processed"] += 1
            self.metrics[f"message_type_{message_type}"] += 1
            
            return response_message
        
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            session.state = ChatState.ERROR
            return self._create_error_message(f"Error: {str(e)}")
        
        finally:
            session.state = previous_state
    
    async def _can_handle(self, handler: Callable, message_type: str, message: str) -> bool:
        """Check if handler can process this message"""
        try:
            if hasattr(handler, "can_handle"):
                return await handler.can_handle(message_type, message)
            return True
        except:
            return True
    
    async def _handle_text_message(
            self,
            message: str,
            session: ChatSession,
            metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Handle regular text message"""
        # Check cache
        if self.cache_enabled:
            cache_key = self._generate_cache_key(message, session.language)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                return ChatMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content=cached_response["content"],
                    timestamp=datetime.now(),
                    citations=cached_response.get("citations", []),
                    sources=cached_response.get("sources", []),
                    language=session.language
                )
        
        # Process message - this would call your agent system
        response_content = await self._generate_response(message, session)
        
        # Cache response
        if self.cache_enabled:
            self.cache.set(cache_key, {
                "content": response_content,
                "citations": [],
                "sources": []
            })
        
        return ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response_content,
            timestamp=datetime.now(),
            language=session.language
        )
    
    async def _handle_voice_message(
            self,
            message: str,
            session: ChatSession,
            metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Handle voice message"""
        # Voice messages are transcribed to text first
        # Then processed as text message
        return await self._handle_text_message(message, session, metadata)
    
    async def _handle_file_message(
            self,
            message: str,
            session: ChatSession,
            metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Handle file message with attachments"""
        file_info = metadata.get("file_info", {}) if metadata else {}
        
        response = f"📎 **File received:** {file_info.get('filename', 'Unknown')}\n\n"
        response += "Processing file content..."
        
        # Extract text from file based on type
        if file_info.get("content"):
            response += f"\n\nExtracted content:\n{file_info['content'][:500]}..."
        
        return ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            language=session.language
        )
    
    async def _handle_command_message(
            self,
            message: str,
            session: ChatSession,
            metadata: Optional[Dict] = None
    ) -> ChatMessage:
        """Handle command messages (starting with /)"""
        command = message.lower().strip()
        
        if command == "/clear":
            session.clear()
            response = "✅ Conversation history cleared."
        elif command == "/help":
            response = self._get_help_text()
        elif command == "/stats":
            response = self._get_session_stats(session)
        elif command == "/export":
            response = self._export_session(session)
        else:
            response = f"❌ Unknown command: {command}\nType /help for available commands."
        
        return ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            language=session.language
        )
    
    async def _handle_default(
            self,
            message: str,
            session: ChatSession
    ) -> ChatMessage:
        """Default message handler"""
        response = await self._generate_response(message, session)
        
        return ChatMessage(
            id=str(uuid.uuid4()),
            role="assistant",
            content=response,
            timestamp=datetime.now(),
            language=session.language
        )
    
    async def _generate_response(self, message: str, session: ChatSession) -> str:
        """
        Generate response using agent system
        This is where you integrate with your actual agent implementation
        """
        # Placeholder - replace with actual agent integration
        return f"""I understand you're asking about: "{message}"

This is a response from the Chronobiotics Agent System. The system is designed to help with:
- Chemical structure analysis and property prediction
- Chronobiotics mechanism research
- Clinical trial data retrieval
- Literature mining and citation management

How can I assist you with your chronobiotics research today?
"""
    
    def _generate_cache_key(self, message: str, language: str) -> str:
        """Generate cache key from message"""
        key_data = f"{message}:{language}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _create_error_message(self, error: str) -> ChatMessage:
        """Create error message"""
        return ChatMessage(
            id=str(uuid.uuid4()),
            role="system",
            content=f"⚠️ {error}",
            timestamp=datetime.now(),
            confidence=0.0
        )
    
    def _get_help_text(self) -> str:
        """Get help text for commands"""
        return """**Available Commands:**

/clear - Clear conversation history
/help - Show this help message
/stats - Show session statistics
/export - Export conversation

**Tips:**
- Ask about chemical structures using SMILES notation
- Request literature searches for specific topics
- Inquire about chronobiotics mechanisms
- Ask for clinical trial information

**Examples:**
- "What is the mechanism of melatonin?"
- "Find clinical trials for circadian rhythm disorders"
- "Analyze SMILES: CC(=O)NC1=CC=C(O)C=C1"
"""
    
    def _get_session_stats(self, session: ChatSession) -> str:
        """Get session statistics"""
        return f"""**Session Statistics:**

- Session ID: {session.session_id}
- Created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')}
- Last active: {session.last_active.strftime('%Y-%m-%d %H:%M:%S')}
- Messages: {len(session.messages)}
- Conversation length: {session.get_conversation_length()} characters
- Language: {session.language}
"""
    
    def _export_session(self, session: ChatSession) -> str:
        """Export session as JSON"""
        export_data = {
            "session": session.to_dict(),
            "messages": [msg.to_dict() for msg in session.messages]
        }
        return f"```json\n{json.dumps(export_data, indent=2)}\n```"
    
    def get_session(self, session_id: str) -> Optional[ChatSession]:
        """Get session by ID"""
        return self.sessions.get(session_id)
    
    def get_all_sessions(self) -> List[ChatSession]:
        """Get all active sessions"""
        return list(self.sessions.values())
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            self.metrics["sessions_deleted"] += 1
            return True
        return False
    
    def get_metrics(self) -> Dict:
        """Get engine metrics"""
        metrics = dict(self.metrics)
        metrics["active_sessions"] = len(self.sessions)
        
        if self.cache_enabled:
            metrics["cache"] = self.cache.get_stats()
        
        return metrics
    
    async def stream_message(
            self,
            session_id: str,
            message: str,
            user_id: Optional[str] = None
    ):
        """
        Stream message response token by token
        """
        session = self.get_or_create_session(session_id, user_id)
        session.state = ChatState.PROCESSING
        
        try:
            # Generate response in chunks
            response = await self._generate_response(message, session)
            
            # Stream in chunks
            chunk_size = 50
            words = response.split()
            
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                yield chunk
                await asyncio.sleep(0.05)
        
        finally:
            session.state = ChatState.ACTIVE
