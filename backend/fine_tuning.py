import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

class ValidationError(Exception):
    """Raised when validation of input data fails"""
    pass

class ResourceError(Exception):
    """Raised when a resource operation fails"""
    pass

logger = logging.getLogger(__name__)

class VersionManager:
    def __init__(self):
        self.version_dir = Path("model_versions")
        self.version_files = {
            "planning": self.version_dir / "planning_versions.json",
            "coding": self.version_dir / "coding_versions.json"
        }
        self.initialized = False
        self.init_lock = asyncio.Lock()

    async def _init_version_tracking(self):
        """Initialize version tracking files"""
        async with self.init_lock:
            if self.initialized:
                return

            try:
                # Create version directory if it doesn't exist
                os.makedirs(self.version_dir, exist_ok=True)

                # Initialize version files if they don't exist
                for file_path in self.version_files.values():
                    if not file_path.exists():
                        async with asyncio.Lock():
                            with open(file_path, 'w') as f:
                                json.dump({
                                    "versions": [],
                                    "current": None,
                                    "last_updated": datetime.now().isoformat()
                                }, f, indent=2)

                self.initialized = True
                logger.info("Version tracking initialized")
            except Exception as e:
                logger.error(f"Failed to initialize version tracking: {str(e)}")
                raise

class FineTuningManager:
    def __init__(self):
        self.version_manager = VersionManager()
        self.training_dir = Path("training_data")
        self.initialized = False
        self.init_lock = asyncio.Lock()

    async def initialize(self):
        """Initialize fine-tuning manager"""
        async with self.init_lock:
            if self.initialized:
                return

            try:
                # Create training data directories
                os.makedirs(self.training_dir / "planning", exist_ok=True)
                os.makedirs(self.training_dir / "coding", exist_ok=True)

                # Initialize version tracking
                await self.version_manager._init_version_tracking()

                self.initialized = True
                logger.info("Fine-tuning manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize fine-tuning manager: {str(e)}")
                raise

    async def save_generation(
        self,
        model_type: str,
        prompt: str,
        completion: str,
        metadata: Optional[Dict] = None
    ):
        """Save generation for fine-tuning"""
        if not self.initialized:
            await self.initialize()

        try:
            # Create timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{model_type}.json"
            filepath = self.training_dir / model_type / filename

            # Save generation data
            data = {
                "prompt": prompt,
                "completion": completion,
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat()
            }

            async with asyncio.Lock():
                with open(filepath, 'w') as f:
                    json.dump(data, f, indent=2)

            logger.info(f"Saved {model_type} generation to {filename}")
        except Exception as e:
            logger.error(f"Failed to save generation: {str(e)}")
            raise

    async def get_training_stats(self) -> Dict:
        """Get training data statistics"""
        if not self.initialized:
            await self.initialize()

        try:
            stats = {
                "planning": {"total_samples": 0, "last_updated": None},
                "coding": {"total_samples": 0, "last_updated": None}
            }

            for model_type in ["planning", "coding"]:
                dir_path = self.training_dir / model_type
                files = list(dir_path.glob("*.json"))
                stats[model_type]["total_samples"] = len(files)
                if files:
                    latest_file = max(files, key=lambda x: x.stat().st_mtime)
                    stats[model_type]["last_updated"] = datetime.fromtimestamp(
                        latest_file.stat().st_mtime
                    ).isoformat()

            return stats
        except Exception as e:
            logger.error(f"Failed to get training stats: {str(e)}")
            raise

    async def fine_tune_model(
        self,
        model_type: str,
        base_model: str
    ) -> Optional[str]:
        """Fine-tune model with collected data"""
        if not self.initialized:
            await self.initialize()

        try:
            # Get training files
            training_path = self.training_dir / model_type
            training_files = list(training_path.glob("*.json"))

            if not training_files:
                logger.info(f"No training data available for {model_type}")
                return None

            # Create new version name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_version = f"{base_model}-ft-{timestamp}"

            # Record new version
            version_file = self.version_manager.version_files[model_type]
            async with asyncio.Lock():
                with open(version_file, 'r') as f:
                    versions = json.load(f)

                versions["versions"].append({
                    "name": new_version,
                    "base_model": base_model,
                    "created": datetime.now().isoformat(),
                    "training_samples": len(training_files)
                })
                versions["current"] = new_version
                versions["last_updated"] = datetime.now().isoformat()

                with open(version_file, 'w') as f:
                    json.dump(versions, f, indent=2)

            logger.info(f"Created new {model_type} model version: {new_version}")
            return new_version

        except Exception as e:
            logger.error(f"Failed to fine-tune model: {str(e)}")
            raise
