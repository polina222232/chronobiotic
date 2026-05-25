# chronobioticagent/main/agent/llm/llm_service.py
"""LLM Service for managing language model interactions"""

import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    LOCAL = "local"


@dataclass
class LLMRequest:
    prompt: str
    temperature: float = 0.3
    max_tokens: int = 2000
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    stream: bool = False


@dataclass
class LLMResponse:
    text: str
    model: str
    tokens_used: int
    finish_reason: str
    processing_time_ms: float


class LLMService:
    """Service for interacting with various LLM providers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = LLMProvider(config.get("provider", "openai"))
        self.api_key = config.get("api_key")
        self.model = config.get("model", "gpt-4")
        self.cache = {}
        
        # Rate limiting
        self.requests_per_minute = config.get("rate_limit", 60)
        self._request_times = []
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate response from LLM"""
        import time
        start_time = time.time()
        
        # Check cache
        cache_key = self._get_cache_key(request)
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Apply rate limiting
        await self._check_rate_limit()
        
        # Call appropriate provider
        if self.provider == LLMProvider.OPENAI:
            response = await self._call_openai(request)
        elif self.provider == LLMProvider.ANTHROPIC:
            response = await self._call_anthropic(request)
        elif self.provider == LLMProvider.GEMINI:
            response = await self._call_gemini(request)
        else:
            response = await self._call_local(request)
        
        # Cache response
        self.cache[cache_key] = response
        
        return response
    
    async def _call_openai(self, request: LLMRequest) -> LLMResponse:
        """Call OpenAI API"""
        import aiohttp
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": request.stream
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
            ) as resp:
                data = await resp.json()
                
                return LLMResponse(
                    text=data["choices"][0]["message"]["content"],
                    model=data["model"],
                    tokens_used=data["usage"]["total_tokens"],
                    finish_reason=data["choices"][0]["finish_reason"],
                    processing_time_ms=0  # Will be set by caller
                )
    
    async def _call_anthropic(self, request: LLMRequest) -> LLMResponse:
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stop_sequences": request.stop_sequences or []
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.anthropic.com/v1/messages",
                    headers=headers,
                    json=payload
            ) as resp:
                data = await resp.json()
                
                return LLMResponse(
                    text=data["content"][0]["text"],
                    model=data["model"],
                    tokens_used=data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                    finish_reason=data["stop_reason"],
                    processing_time_ms=0
                )
    
    async def _call_gemini(self, request: LLMRequest) -> LLMResponse:
        """Call Google Gemini API"""
        # Implementation for Gemini
        pass
    
    async def _call_local(self, request: LLMRequest) -> LLMResponse:
        """Call local LLM (Llama, Mistral, etc.)"""
        # Implementation for local models
        pass
    
    async def stream_generate(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream generation from LLM"""
        if self.provider == LLMProvider.OPENAI:
            async for chunk in self._stream_openai(request):
                yield chunk
    
    async def _stream_openai(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Stream from OpenAI"""
        request.stream = True
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=payload
            ) as resp:
                async for line in resp.content:
                    if line:
                        line = line.decode('utf-8').strip()
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if delta := chunk.get("choices", [{}])[0].get("delta", {}):
                                    if content := delta.get("content"):
                                        yield content
                            except json.JSONDecodeError:
                                continue
    
    def _get_cache_key(self, request: LLMRequest) -> str:
        """Generate cache key for request"""
        import hashlib
        key_string = f"{request.prompt}_{request.temperature}_{self.model}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        import time
        
        now = time.time()
        self._request_times = [t for t in self._request_times if now - t < 60]
        
        if len(self._request_times) >= self.requests_per_minute:
            wait_time = 60 - (now - self._request_times[0])
            if wait_time > 0:
                await asyncio.sleep(wait_time)
        
        self._request_times.append(now)
