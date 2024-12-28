import json
import logging
import asyncio
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import io

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    name: str
    base_model: str
    created_at: str
    training_samples: int
    metrics: Dict
    status: str

class VersionManager:
    def __init__(self):
        self.versions_dir = Path("model_versions")
        self.versions_dir.mkdir(exist_ok=True)
        
        # Separate tracking for each model type
        self.planning_versions_file = self.versions_dir / "planning_versions.json"
        self.coding_versions_file = self.versions_dir / "coding_versions.json"
        
        # Initialize thread pool with optimal workers
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
        # Enhanced caching
        self._version_cache: Dict[str, Tuple[float, Dict]] = {}
        self._cache_ttl = 300  # Cache TTL increased to 5 minutes
        self._validation_cache: Set[str] = set()  # Cache for validated version names
        
        # Batch processing configuration
        self.batch_size = 1000  # Process versions in larger batches
        
        # Initialize version tracking
        asyncio.create_task(self._init_version_tracking())
        
    async def _init_version_tracking(self):
        """Initialize or load version tracking files asynchronously"""
        initial_data = {
            "versions": [],
            "current_version": None,
            "last_validated": datetime.now().isoformat()
        }
        
        # Initialize both files concurrently
        await asyncio.gather(*[
            self._init_version_file(file_path, initial_data)
            for file_path in [self.planning_versions_file, self.coding_versions_file]
        ])
        
    async def _init_version_file(self, file_path: Path, initial_data: Dict):
        """Initialize a single version file if it doesn't exist"""
        if not file_path.exists():
            await self._save_file(file_path, initial_data)
                    
    async def _load_file(self, file_path: Path) -> Dict:
        """Load file content with optimized I/O"""
        try:
            content = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: file_path.read_text()
            )
            return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {str(e)}")
            return {"versions": [], "current_version": None, "last_validated": datetime.now().isoformat()}

    async def _save_file(self, file_path: Path, data: Dict):
        """Save file content with optimized I/O"""
        content = json.dumps(data, indent=2)
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: file_path.write_text(content)
        )

    async def _get_versions(self, model_type: str) -> Dict:
        """Get version data with enhanced caching"""
        # Check cache first
        now = datetime.now().timestamp()
        if model_type in self._version_cache:
            timestamp, data = self._version_cache[model_type]
            if now - timestamp < self._cache_ttl:
                return data.copy()  # Return a copy to prevent cache corruption
        
        # Load data
        file_path = self.planning_versions_file if model_type == "planning" else self.coding_versions_file
        data = await self._load_file(file_path)
        
        # Update cache
        self._version_cache[model_type] = (now, data.copy())
        return data
            
    async def _save_versions(self, model_type: str, data: Dict):
        """Save version data with cache update"""
        file_path = self.planning_versions_file if model_type == "planning" else self.coding_versions_file
        await self._save_file(file_path, data)
        self._version_cache[model_type] = (datetime.now().timestamp(), data.copy())
            
    def _validate_version_data(self, data: Dict) -> bool:
        """Validate version data structure with caching"""
        # Quick validation for cached versions
        if any(version["name"] in self._validation_cache for version in data.get("versions", [])):
            return True
            
        try:
            # Basic structure validation
            if not isinstance(data.get("versions"), list):
                return False
                
            if "current_version" not in data or "last_validated" not in data:
                return False
            
            # Validate each version
            required_keys = {"name", "base_model", "created_at", "training_samples", "metrics", "status"}
            for version in data["versions"]:
                if not all(key in version for key in required_keys):
                    return False
                self._validation_cache.add(version["name"])
                
            return True
            
        except Exception:
            return False

    async def register_version(self, model_type: str, version: ModelVersion) -> bool:
        """Register a new model version with optimized processing"""
        try:
            versions_data = await self._get_versions(model_type)
            
            if not self._validate_version_data(versions_data):
                logger.error("Invalid version data structure")
                return False
            
            # Add new version
            version_dict = version.__dict__
            versions_data["versions"].append(version_dict)
            versions_data["last_validated"] = datetime.now().isoformat()
            
            # Update current version if first or successful
            if not versions_data["current_version"] or version.status == "success":
                versions_data["current_version"] = version.name
                
            # Save changes
            await self._save_versions(model_type, versions_data)
            
            # Update validation cache
            self._validation_cache.add(version.name)
            
            logger.info(f"Registered new {model_type} model version: {version.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register version: {str(e)}")
            return False
            
    @lru_cache(maxsize=10)
    async def get_current_version(self, model_type: str) -> Optional[ModelVersion]:
        """Get current active version with caching"""
        try:
            versions_data = await self._get_versions(model_type)
            current = versions_data["current_version"]
            
            if not current:
                return None
                
            # Find current version data
            for version in versions_data["versions"]:
                if version["name"] == current:
                    return ModelVersion(**version)
                    
            return None
            
        except Exception as e:
            logger.error(f"Failed to get current version: {str(e)}")
            return None
            
    async def rollback_version(self, model_type: str) -> Optional[ModelVersion]:
        """Rollback to previous successful version with optimized lookup"""
        try:
            versions_data = await self._get_versions(model_type)
            
            # Find successful versions in chronological order
            successful_versions = [
                v for v in versions_data["versions"]
                if v["status"] == "success"
            ]
            
            if len(successful_versions) < 2:
                logger.warning("No previous successful version to rollback to")
                return None
            
            # Get the most recent successful version before the current one
            current_version = versions_data["current_version"]
            previous_version = None
            
            for i in range(len(successful_versions) - 1, -1, -1):
                if successful_versions[i]["name"] == current_version and i > 0:
                    previous_version = successful_versions[i - 1]
                    break
            
            if not previous_version:
                logger.warning("No previous successful version found")
                return None
            versions_data["current_version"] = previous_version["name"]
            
            await self._save_versions(model_type, versions_data)
            logger.info(f"Rolled back to version: {previous_version['name']}")
            
            return ModelVersion(**previous_version)
            
        except Exception as e:
            logger.error(f"Rollback failed: {str(e)}")
            return None
            
    async def cleanup_old_versions(self, model_type: str, keep_versions: int = 5):
        """Remove old model versions with optimized batch processing"""
        try:
            versions_data = await self._get_versions(model_type)
            versions = versions_data["versions"]
            
            if len(versions) <= keep_versions:
                return
            
            # Calculate versions to remove
            to_remove = len(versions) - keep_versions
            
            if to_remove > 0:
                # Keep the most recent versions
                versions_data["versions"] = versions[-keep_versions:]
                
                # Update current version if needed
                if versions_data["current_version"] not in {v["name"] for v in versions_data["versions"]}:
                    versions_data["current_version"] = versions_data["versions"][-1]["name"]
                
                # Update cache and save
                await self._save_versions(model_type, versions_data)
                
                # Update validation cache
                self._validation_cache = {v["name"] for v in versions_data["versions"]}
                
            logger.info(f"Cleaned up {to_remove} old versions for {model_type} model")
            
        except Exception as e:
            logger.error(f"Version cleanup failed: {str(e)}")
            
    @lru_cache(maxsize=10)
    async def get_version_history(self, model_type: str) -> List[ModelVersion]:
        """Get version history with caching"""
        try:
            versions_data = await self._get_versions(model_type)
            return [ModelVersion(**v) for v in versions_data["versions"]]
        except Exception as e:
            logger.error(f"Failed to get version history: {str(e)}")
            return []
