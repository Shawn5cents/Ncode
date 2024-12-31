from mcp import ClientSession as Client, Tool, Resource, stdio_client
import logging
import asyncio
from typing import Dict, Optional, Any
from pathlib import Path
import sys
from functools import lru_cache
import time

logger = logging.getLogger(__name__)

class MCPIntegration:
    def __init__(self):
        self.client = stdio_client()
        self.initialized = False
        self.init_lock = asyncio.Lock()
        self.tools_registered = False
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        self.background_tasks = set()
        
    @lru_cache(maxsize=1000)
    async def _cached_tool_call(self, tool_name: str, args_key: str) -> Dict:
        """Cached tool calls for faster repeated operations"""
        current_time = time.time()
        cache_key = f"{tool_name}:{args_key}"
        
        # Check cache
        if cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                return result
        
        # Make actual tool call
        result = await self.client.call_tool(tool_name, eval(args_key))
        
        # Update cache
        self.cache[cache_key] = (result, current_time)
        return result
        
    async def initialize(self):
        """Fast initialization with background tool registration"""
        async with self.init_lock:
            if self.initialized:
                return
                
            try:
                # Initialize client connection
                await self.client.connect()
                
                # Start tool registration in background
                if not self.tools_registered:
                    task = asyncio.create_task(self._register_tools())
                    self.background_tasks.add(task)
                    task.add_done_callback(self.background_tasks.discard)
                
                self.initialized = True
                logger.info("MCP integration initialized")
            except Exception as e:
                logger.error(f"MCP initialization error: {str(e)}")
                raise
                
    async def _register_tools(self):
        """Register tools with optimized schemas"""
        try:
            tools = [
                Tool(
                    name="enhance_generation",
                    description="Enhance generation with context and examples",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string"},
                            "mode": {"type": "string", "enum": ["planning", "coding"]},
                            "context_type": {"type": "string", "enum": ["fast", "comprehensive"]}
                        },
                        "required": ["prompt", "mode"]
                    }
                ),
                Tool(
                    name="get_context",
                    description="Get relevant context for generation",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "mode": {"type": "string", "enum": ["planning", "coding"]},
                            "keywords": {"type": "array", "items": {"type": "string"}}
                        },
                        "required": ["mode"]
                    }
                ),
                Tool(
                    name="manage_connections",
                    description="Manage MCP server connections",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["connect", "disconnect", "status"]},
                            "server": {"type": "string"}
                        },
                        "required": ["action"]
                    }
                ),
                Tool(
                    name="monitor_usage",
                    description="Monitor MCP tool usage statistics",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "time_range": {"type": "string", "enum": ["day", "week", "month"]},
                            "tool_name": {"type": "string"}
                        }
                    }
                ),
                Tool(
                    name="validate_responses",
                    description="Validate MCP tool responses",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "tool_name": {"type": "string"},
                            "response": {"type": "object"}
                        },
                        "required": ["tool_name", "response"]
                    }
                ),
                Tool(
                    name="manage_configurations",
                    description="Manage MCP tool configurations",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["get", "set"]},
                            "tool_name": {"type": "string"},
                            "config": {"type": "object"}
                        },
                        "required": ["action"]
                    }
                )
            ]
            
            # Register all tools concurrently
            await asyncio.gather(*[
                self.client.register_tool(tool) for tool in tools
            ])
            
            self.tools_registered = True
            logger.info("MCP tools registered")
            
        except Exception as e:
            logger.error(f"Tool registration error: {str(e)}")
            raise
            
    async def enhance_generation(self, prompt: str, mode: str, fast: bool = True) -> Dict:
        """Enhance generation with context, optimized for speed"""
        if not self.initialized:
            await self.initialize()
            
        try:
            # Use cached results when possible
            context_type = "fast" if fast else "comprehensive"
            args_key = str({"prompt": prompt, "mode": mode, "context_type": context_type})
            
            return await self._cached_tool_call("enhance_generation", args_key)
            
        except Exception as e:
            logger.error(f"Enhancement error: {str(e)}")
            # Return None on error to allow generation to proceed without enhancement
            return None
            
    async def get_context(self, mode: str, keywords: Optional[list] = None) -> Dict:
        """Get generation context with caching"""
        if not self.initialized:
            await self.initialize()
            
        try:
            args_key = str({"mode": mode, "keywords": keywords or []})
            return await self._cached_tool_call("get_context", args_key)
            
        except Exception as e:
            logger.error(f"Context error: {str(e)}")
            return None
            
    def _cleanup_cache(self):
        """Clean expired cache entries"""
        current_time = time.time()
        self.cache = {
            k: (v, t) for k, (v, t) in self.cache.items()
            if current_time - t < self.cache_ttl
        }
            
    async def cleanup(self):
        """Cleanup resources"""
        try:
            if self.initialized:
                # Cancel background tasks
                for task in self.background_tasks:
                    task.cancel()
                    
                # Clean cache
                self._cleanup_cache()
                
                # Disconnect client
                await self.client.disconnect()
                self.initialized = False
                logger.info("MCP cleanup complete")
        except Exception as e:
            logger.error(f"Cleanup error: {str(e)}")
            raise

    async def manage_connections(self, action: str, server: Optional[str] = None) -> Dict:
        """Manage MCP server connections"""
        if not self.initialized:
            await self.initialize()
            
        try:
            args_key = str({"action": action, "server": server})
            return await self._cached_tool_call("manage_connections", args_key)
        except Exception as e:
            logger.error(f"Connection management error: {str(e)}")
            return None

    async def monitor_usage(self, time_range: Optional[str] = None, tool_name: Optional[str] = None) -> Dict:
        """Monitor MCP tool usage statistics"""
        if not self.initialized:
            await self.initialize()
            
        try:
            args_key = str({"time_range": time_range, "tool_name": tool_name})
            return await self._cached_tool_call("monitor_usage", args_key)
        except Exception as e:
            logger.error(f"Usage monitoring error: {str(e)}")
            return None

    async def validate_responses(self, tool_name: str, response: Dict) -> Dict:
        """Validate MCP tool responses"""
        if not self.initialized:
            await self.initialize()
            
        try:
            args_key = str({"tool_name": tool_name, "response": response})
            return await self._cached_tool_call("validate_responses", args_key)
        except Exception as e:
            logger.error(f"Response validation error: {str(e)}")
            return None

    async def manage_configurations(self, action: str, tool_name: Optional[str] = None, config: Optional[Dict] = None) -> Dict:
        """Manage MCP tool configurations"""
        if not self.initialized:
            await self.initialize()
            
        try:
            args_key = str({"action": action, "tool_name": tool_name, "config": config})
            return await self._cached_tool_call("manage_configurations", args_key)
        except Exception as e:
            logger.error(f"Configuration management error: {str(e)}")
            return None
