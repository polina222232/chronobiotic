# main/agent/chat/websocket_handlers.py
"""
WebSocket handlers for real-time chat
"""

import json
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class WebSocketChatHandler:
    """
    WebSocket handler for chat messages with streaming support
    """
    
    def __init__(self, chat_engine, session_manager):
        self.chat_engine = chat_engine
        self.session_manager = session_manager
        self.active_connections = {}
        self.streaming_handlers = {}
    
    async def handle_connection(
            self,
            websocket,
            session_id: str,
            user_id: Optional[str] = None
    ):
        """
        Handle new WebSocket connection
        """
        # Store connection
        self.active_connections[session_id] = {
            "websocket": websocket,
            "user_id": user_id,
            "connected_at": datetime.now()
        }
        
        try:
            # Send welcome message
            await self._send_json(websocket, {
                "type": "connection",
                "status": "connected",
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            })
            
            # Send conversation history
            await self._send_history(websocket, session_id)
            
            # Listen for messages
            async for message in websocket:
                await self._handle_message(websocket, session_id, user_id, message)
        
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        
        finally:
            # Clean up
            self.active_connections.pop(session_id, None)
            self.streaming_handlers.pop(session_id, None)
    
    async def _handle_message(
            self,
            websocket,
            session_id: str,
            user_id: Optional[str],
            message: str
    ):
        """
        Handle incoming WebSocket message
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type", "message")
            
            if msg_type == "message":
                await self._handle_chat_message(websocket, session_id, user_id, data)
            
            elif msg_type == "stream":
                await self._handle_streaming_message(websocket, session_id, user_id, data)
            
            elif msg_type == "stop":
                await self._handle_stop_stream(session_id)
            
            elif msg_type == "typing":
                await self._handle_typing_indicator(session_id, data)
            
            elif msg_type == "command":
                await self._handle_command(websocket, session_id, user_id, data)
            
            else:
                await self._send_error(websocket, f"Unknown message type: {msg_type}")
        
        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON format")
        except Exception as e:
            logger.error(f"Message handling error: {e}")
            await self._send_error(websocket, str(e))
    
    async def _handle_chat_message(
            self,
            websocket,
            session_id: str,
            user_id: Optional[str],
            data: Dict
    ):
        """
        Handle regular chat message
        """
        message = data.get("message", "")
        language = data.get("language", "en")
        
        if not message:
            await self._send_error(websocket, "Empty message")
            return
        
        # Send typing indicator
        await self._broadcast_typing(session_id, "agent", True)
        
        try:
            # Process message
            response = await self.chat_engine.process_message(
                session_id=session_id,
                message=message,
                user_id=user_id,
                metadata={"language": language}
            )
            
            # Send response
            await self._send_json(websocket, {
                "type": "message",
                "data": response.to_dict()
            })
            
            # Save to session manager
            await self._save_messages(session_id, message, response.content)
        
        finally:
            await self._broadcast_typing(session_id, "agent", False)
    
    async def _handle_streaming_message(
            self,
            websocket,
            session_id: str,
            user_id: Optional[str],
            data: Dict
    ):
        """
        Handle streaming chat message
        """
        message = data.get("message", "")
        
        if not message:
            await self._send_error(websocket, "Empty message")
            return
        
        # Send stream start
        await self._send_json(websocket, {
            "type": "stream_start",
            "stream_id": session_id
        })
        
        # Store stream handler
        self.streaming_handlers[session_id] = True
        
        try:
            # Stream response
            async for chunk in self.chat_engine.stream_message(
                    session_id=session_id,
                    message=message,
                    user_id=user_id
            ):
                if not self.streaming_handlers.get(session_id, True):
                    break
                
                await self._send_json(websocket, {
                    "type": "stream_chunk",
                    "content": chunk
                })
                await asyncio.sleep(0.05)
            
            # Send stream end
            await self._send_json(websocket, {
                "type": "stream_end",
                "stream_id": session_id
            })
        
        finally:
            self.streaming_handlers.pop(session_id, None)
    
    async def _handle_stop_stream(self, session_id: str):
        """Handle stop streaming request"""
        if session_id in self.streaming_handlers:
            self.streaming_handlers[session_id] = False
    
    async def _handle_typing_indicator(self, session_id: str, data: Dict):
        """Handle typing indicator"""
        is_typing = data.get("is_typing", False)
        user = data.get("user", "user")
        
        await self._broadcast_typing(session_id, user, is_typing)
    
    async def _handle_command(
            self,
            websocket,
            session_id: str,
            user_id: Optional[str],
            data: Dict
    ):
        """Handle special commands"""
        command = data.get("command", "")
        
        if command == "/clear":
            # Clear conversation
            if self.session_manager:
                self.session_manager.clear_session(session_id)
            
            await self._send_json(websocket, {
                "type": "command_result",
                "command": command,
                "result": "Conversation cleared"
            })
        
        elif command == "/history":
            # Get history
            if self.session_manager:
                messages = self.session_manager.get_messages(session_id, limit=50)
                await self._send_json(websocket, {
                    "type": "command_result",
                    "command": command,
                    "result": messages
                })
        
        elif command == "/export":
            # Export conversation
            if self.session_manager:
                export = self.session_manager.export_session(session_id)
                await self._send_json(websocket, {
                    "type": "command_result",
                    "command": command,
                    "result": export
                })
        
        else:
            await self._send_error(websocket, f"Unknown command: {command}")
    
    async def _send_history(self, websocket, session_id: str):
        """Send conversation history"""
        if self.session_manager:
            messages = self.session_manager.get_messages(session_id, limit=50)
            
            await self._send_json(websocket, {
                "type": "history",
                "messages": messages
            })
    
    async def _save_messages(self, session_id: str, user_msg: str, assistant_msg: str):
        """Save messages to session manager"""
        if self.session_manager:
            self.session_manager.add_message(
                session_id=session_id,
                role="user",
                content=user_msg
            )
            self.session_manager.add_message(
                session_id=session_id,
                role="assistant",
                content=assistant_msg
            )
    
    async def _broadcast_typing(self, session_id: str, user: str, is_typing: bool):
        """Broadcast typing indicator to session"""
        connection = self.active_connections.get(session_id)
        if connection:
            await self._send_json(connection["websocket"], {
                "type": "typing",
                "user": user,
                "is_typing": is_typing
            })
    
    async def _send_json(self, websocket, data: Dict):
        """Send JSON data over WebSocket"""
        await websocket.send(json.dumps(data))
    
    async def _send_error(self, websocket, error: str):
        """Send error message"""
        await self._send_json(websocket, {
            "type": "error",
            "error": error,
            "timestamp": datetime.now().isoformat()
        })
