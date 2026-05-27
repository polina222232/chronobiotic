"""
Streaming handler for real-time response streaming
"""

import asyncio
import random
from typing import AsyncGenerator, Callable, Optional, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class StreamingHandler:
    """
    Handles streaming of responses for real-time chat
    """
    
    def __init__(self, chunk_size: int = 50, delay: float = 0.05):
        self.chunk_size = chunk_size
        self.delay = delay
        self.active_streams = {}
    
    async def stream_text(
            self,
            text: str,
            callback: Callable,
            stream_id: str = None,
            simulate_typing: bool = False
    ) -> None:
        """
        Stream text in chunks
        
        Args:
            text: Text to stream
            callback: Async callback for each chunk
            stream_id: Optional stream identifier
            simulate_typing: Whether to simulate typing delay
        """
        if stream_id:
            self.active_streams[stream_id] = True
        
        try:
            # Split into words for natural streaming
            words = text.split()
            chunks = []
            
            for i in range(0, len(words), self.chunk_size):
                chunk = " ".join(words[i:i + self.chunk_size])
                chunks.append(chunk)
            
            for i, chunk in enumerate(chunks):
                # Check if stream was cancelled
                if stream_id and not self.active_streams.get(stream_id, True):
                    break
                
                await callback(chunk)
                
                # Add delay between chunks
                if simulate_typing and i < len(chunks) - 1:
                    await asyncio.sleep(self.delay)
        
        finally:
            if stream_id:
                self.active_streams.pop(stream_id, None)
    
    async def stream_characters(
            self,
            text: str,
            callback: Callable,
            stream_id: str = None,
            variable_speed: bool = True
    ) -> None:
        """
        Stream text character by character
        
        Args:
            text: Text to stream
            callback: Async callback for each character
            stream_id: Optional stream identifier
            variable_speed: Whether to add random speed variations
        """
        if stream_id:
            self.active_streams[stream_id] = True
        
        try:
            for char in text:
                if stream_id and not self.active_streams.get(stream_id, True):
                    break
                
                await callback(char)
                
                # Calculate delay
                delay = 0.05  # Base delay
                
                if variable_speed:
                    # Natural typing variations
                    if char in '.!?;:':
                        delay = 0.3  # Pause at punctuation
                    elif char == ' ':
                        delay = 0.02  # Faster on spaces
                    elif char == '\n':
                        delay = 0.5  # Pause at line breaks
                    else:
                        delay = random.uniform(0.03, 0.08)
                
                await asyncio.sleep(delay)
        
        finally:
            if stream_id:
                self.active_streams.pop(stream_id, None)
    
    async def stream_with_think(
            self,
            text: str,
            callback: Callable,
            think_callback: Optional[Callable] = None
    ) -> None:
        """
        Stream with thinking indicator before response
        """
        if think_callback:
            await think_callback("🤔 Thinking...")
            await asyncio.sleep(1)
            await think_callback("💭 Analyzing...")
            await asyncio.sleep(0.5)
            await think_callback("✨ Generating response...")
            await asyncio.sleep(0.5)
            await think_callback(None)  # Clear thinking indicator
        
        await self.stream_text(text, callback, simulate_typing=True)
    
    def cancel_stream(self, stream_id: str) -> bool:
        """Cancel an active stream"""
        if stream_id in self.active_streams:
            self.active_streams[stream_id] = False
            return True
        return False
    
    def is_streaming(self, stream_id: str) -> bool:
        """Check if stream is active"""
        return stream_id in self.active_streams
    
    def get_active_streams(self) -> List[str]:
        """Get list of active stream IDs"""
        return list(self.active_streams.keys())


class TypingSimulator:
    """
    Simulates human-like typing for chat responses
    """
    
    def __init__(self, wpm: int = 200):
        self.wpm = wpm
        # Average word is 5 characters
        self.char_delay = 60 / (wpm * 5)
        
        # Common typing patterns
        self.punctuation_delay = {
            '.': 0.3,
            '!': 0.3,
            '?': 0.3,
            ',': 0.1,
            ';': 0.1,
            ':': 0.1,
            '\n': 0.5
        }
    
    async def simulate(
            self,
            text: str,
            callback: Callable,
            include_errors: bool = False
    ) -> None:
        """
        Simulate human typing
        
        Args:
            text: Text to type
            callback: Callback for each character
            include_errors: Whether to simulate typing errors
        """
        for i, char in enumerate(text):
            await callback(char)
            
            # Base delay
            delay = self.char_delay
            
            # Add natural variations
            delay *= random.uniform(0.7, 1.3)
            
            # Add punctuation pauses
            if char in self.punctuation_delay:
                delay += self.punctuation_delay[char]
            
            # Simulate typing errors (rare)
            if include_errors and random.random() < 0.01:
                # Type wrong character
                wrong_char = random.choice('abcdefghijklmnopqrstuvwxyz')
                await callback(wrong_char)
                await asyncio.sleep(0.1)
                # Backspace
                await callback('\b')
                await asyncio.sleep(0.05)
                # Type correct character
                await callback(char)
                delay = 0.2
            
            await asyncio.sleep(delay)
    
    @classmethod
    def calculate_typing_time(cls, text: str, wpm: int = 200) -> float:
        """Calculate estimated typing time in seconds"""
        char_count = len(text)
        words = char_count / 5
        minutes = words / wpm
        return minutes * 60


class StreamBuffer:
    """
    Buffer for collecting streaming chunks
    """
    
    def __init__(self):
        self.buffer = []
        self.start_time = None
        self.end_time = None
    
    def add_chunk(self, chunk: str):
        """Add chunk to buffer"""
        self.buffer.append(chunk)
        
        if not self.start_time:
            self.start_time = datetime.now()
    
    def complete(self):
        """Mark buffer as complete"""
        self.end_time = datetime.now()
    
    def get_full_text(self) -> str:
        """Get complete text"""
        return ''.join(self.buffer)
    
    def get_length(self) -> int:
        """Get total length"""
        return len(self.get_full_text())
    
    def get_duration(self) -> float:
        """Get streaming duration in seconds"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()
        self.start_time = None
        self.end_time = None
