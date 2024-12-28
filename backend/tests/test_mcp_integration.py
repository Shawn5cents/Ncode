import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from ..mcp_integration import MCPIntegration

@pytest.fixture
async def mcp_client():
    client = MCPIntegration()
    yield client
    await client.cleanup()

@pytest.mark.asyncio
async def test_initialization():
    """Test MCP client initialization"""
    client = MCPIntegration()
    try:
        await client.initialize()
        assert client.initialized == True
        assert client.tools_registered == True
    finally:
        await client.cleanup()

@pytest.mark.asyncio
async def test_model_management():
    """Test model management operations"""
    client = MCPIntegration()
    try:
        await client.initialize()
        result = await client.manage_models(
            action="list",
            model_type="planning"
        )
        assert isinstance(result, dict)
    finally:
        await client.cleanup()

@pytest.mark.asyncio
async def test_training_data_management():
    """Test training data operations"""
    client = MCPIntegration()
    try:
        await client.initialize()
        result = await client.manage_training_data(
            action="list",
            data_type="coding"
        )
        assert isinstance(result, dict)
    finally:
        await client.cleanup()

@pytest.mark.asyncio
async def test_performance_monitoring():
    """Test performance monitoring"""
    client = MCPIntegration()
    try:
        await client.initialize()
        result = await client.monitor_performance(
            metric="cpu",
            duration=5
        )
        assert isinstance(result, dict)
        assert "cpu_usage" in result
    finally:
        await client.cleanup()

@pytest.mark.asyncio
async def test_cleanup():
    """Test proper cleanup of MCP resources"""
    client = MCPIntegration()
    await client.initialize()
    await client.cleanup()
    assert client.initialized == False
