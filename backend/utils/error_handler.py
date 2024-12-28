from typing import Dict, Any
import logging
from fastapi import HTTPException
from websockets import WebSocketException
from starlette import status

logger = logging.getLogger(__name__)

class APIError(Exception):
    """Base class for API errors"""
    def __init__(self, message: str, code: str, status_code: int = 500):
        self.message = message
        self.code = code
        self.status_code = status_code
        super().__init__(message)

class ModelError(APIError):
    """Errors related to model operations"""
    def __init__(self, message: str):
        super().__init__(message, "MODEL_ERROR", 503)

class ConnectionError(APIError):
    """Errors related to connection issues"""
    def __init__(self, message: str):
        super().__init__(message, "CONNECTION_ERROR", 502)

class ValidationError(APIError):
    """Errors related to input validation"""
    def __init__(self, message: str):
        super().__init__(message, "VALIDATION_ERROR", 400)

def handle_error(error: Exception) -> Dict[str, Any]:
    """Handle errors and return appropriate response"""
    if isinstance(error, APIError):
        logger.error(f"API Error: {error.message}")
        return {
            "type": "error",
            "message": error.message,
            "code": error.code,
            "status_code": error.status_code
        }
    elif isinstance(error, WebSocketException):
        logger.error(f"WebSocket Error: {str(error)}")
        return {
            "type": "error",
            "message": "WebSocket connection error",
            "code": "WEBSOCKET_ERROR",
            "status_code": 502
        }
    elif isinstance(error, HTTPException):
        logger.error(f"HTTP Error: {str(error)}")
        return {
            "type": "error",
            "message": error.detail,
            "code": "HTTP_ERROR",
            "status_code": error.status_code
        }
    else:
        logger.error(f"Unexpected Error: {str(error)}")
        return {
            "type": "error",
            "message": "An unexpected error occurred",
            "code": "UNEXPECTED_ERROR",
            "status_code": 500
        }

def create_error_response(error: Exception):
    """Create a standardized error response"""
    error_info = handle_error(error)
    return {
        "error": error_info["message"],
        "code": error_info["code"],
        "status_code": error_info["status_code"]
    }
