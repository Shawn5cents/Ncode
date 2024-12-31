import os
import json
import shutil
import gzip
import logging
import psutil
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import io

logger = logging.getLogger(__name__)

@dataclass
class StorageMetrics:
    total_space: int
    used_space: int
    free_space: int
    usage_percent: float

class StorageManager:
    def __init__(self, base_dir: str = "storage"):
        self.base_dir = Path(base_dir)
        self.backup_dir = self.base_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Configure backup retention
        self.backup_retention_days = 30
        self.compression_threshold_days = 7
        
        # Configure storage thresholds
        self.storage_warning_threshold = 80  # Percentage
        self.storage_critical_threshold = 90  # Percentage
        
        # Initialize metrics tracking
        self.metrics: Dict[str, StorageMetrics] = {}
        
        # Initialize thread pool with optimal workers
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count() or 4)
        
        # Enhanced caching
        self.compressed_files: Set[Path] = set()
        self.verified_files: Set[Path] = set()
        self._last_metrics_check = 0
        self._metrics_cache_duration = 300  # Cache metrics for 5 minutes
        self._metrics_cache: Optional[StorageMetrics] = None
        
        # Batch processing configuration
        self.batch_size = 1000  # Process files in larger batches
        self.compression_batch_size = 100  # Compress files in batches
        
    def get_storage_metrics(self) -> StorageMetrics:
        """Get current storage metrics with efficient caching"""
        current_time = datetime.now().timestamp()
        
        # Return cached metrics if still valid
        if (self._metrics_cache is not None and 
            current_time - self._last_metrics_check < self._metrics_cache_duration):
            return self._metrics_cache
            
        # Update metrics
        disk = psutil.disk_usage(self.base_dir)
        self._metrics_cache = StorageMetrics(
            total_space=disk.total,
            used_space=disk.used,
            free_space=disk.free,
            usage_percent=disk.percent
        )
        self._last_metrics_check = current_time
        
        return self._metrics_cache
        
    def check_storage_health(self) -> bool:
        """Check if storage usage is within acceptable limits"""
        metrics = self.get_storage_metrics()
        
        if metrics.usage_percent >= self.storage_critical_threshold:
            logger.critical(f"Storage usage critical: {metrics.usage_percent}%")
            asyncio.create_task(self.cleanup_old_data())  # Non-blocking cleanup
            return False
            
        if metrics.usage_percent >= self.storage_warning_threshold:
            logger.warning(f"Storage usage high: {metrics.usage_percent}%")
            
        return True
        
    def _get_code_directory(self, filename: str) -> Path:
        """Get the appropriate directory for storing generated code."""
        # Create base directory structure
        today = datetime.now().strftime("%Y-%m-%d")
        project = "default"
        
        # Extract project name from filename if specified
        if '_' in filename:
            project = filename.split('_')[0]
            
        # Create directory path
        code_dir = self.generated_code_dir / today / project
        code_dir.mkdir(parents=True, exist_ok=True)
        return code_dir

    async def save_generated_code(self, filename: str, content: str) -> str:
        """Save generated code with proper organization."""
        try:
            # Ensure filename has .py extension
            if not filename.endswith('.py'):
                filename += '.py'
                
            # Get appropriate directory
            code_dir = self._get_code_directory(filename)
            file_path = code_dir / filename
            
            # Save the content
            await self._write_file(file_path, content)
            
            # Update metrics
            self.metrics['generated_code'] = self.metrics.get('generated_code', 0) + 1
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"Error saving generated code: {str(e)}")
            raise

    async def create_backup(self) -> str:
        """Create a backup of all data asynchronously"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        try:
            # Create backup directory
            backup_path.mkdir(exist_ok=True)
            
            # Backup conversations and knowledge base concurrently
            await asyncio.gather(
                self._copy_directory(
                    self.base_dir / "conversations",
                    backup_path / "conversations"
                ),
                self._copy_directory(
                    self.base_dir / "knowledge_base",
                    backup_path / "knowledge_base"
                )
            )
            
            # Create backup manifest
            manifest = {
                "timestamp": timestamp,
                "files": [str(p) for p in backup_path.rglob("*")],
                "metrics": self.get_storage_metrics().__dict__
            }
            
            await self._write_json(backup_path / "manifest.json", manifest)
            
            logger.info(f"Backup created successfully: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Backup creation failed: {str(e)}")
            if backup_path.exists():
                await self._remove_directory(backup_path)
            raise
            
    async def compress_old_data(self):
        """Compress data older than threshold with batched processing"""
        threshold_date = datetime.now() - timedelta(days=self.compression_threshold_days)
        
        # Get all files needing compression
        files_to_compress = [
            f for f in self.base_dir.rglob("*.json")
            if f.stat().st_mtime < threshold_date.timestamp() and 
               f not in self.compressed_files
        ]
        
        # Process in batches
        for i in range(0, len(files_to_compress), self.compression_batch_size):
            batch = files_to_compress[i:i + self.compression_batch_size]
            await asyncio.gather(*[self._compress_file(f) for f in batch])
    
    async def _compress_file(self, file_path: Path):
        """Compress a single file with optimized I/O"""
        try:
            # Read file content
            content = await self._read_file(file_path)
            
            # Compress in memory
            gz_buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=gz_buffer, mode='wb') as gz:
                gz.write(content.encode())
            
            # Write compressed file
            gz_path = file_path.with_suffix(".json.gz")
            await self._write_bytes(gz_path, gz_buffer.getvalue())
            
            # Remove original file
            await self._remove_file(file_path)
            
            self.compressed_files.add(gz_path)
            logger.info(f"Compressed {file_path} to {gz_path}")
            
        except Exception as e:
            logger.error(f"Compression failed for {file_path}: {str(e)}")
                    
    async def cleanup_old_data(self):
        """Remove old backups and compress old data asynchronously"""
        retention_date = datetime.now() - timedelta(days=self.backup_retention_days)
        
        # Get old backups
        old_backups = [
            d for d in self.backup_dir.iterdir()
            if d.stat().st_mtime < retention_date.timestamp()
        ]
        
        # Remove old backups concurrently
        await asyncio.gather(*[
            self._remove_directory(d) for d in old_backups
        ])
        
        # Compress old data
        await self.compress_old_data()
        
    async def verify_data_integrity(self) -> bool:
        """Verify integrity of all data files with batched processing"""
        # Get all files needing verification
        unverified_files = [
            (f, f.suffix == '.gz')
            for f in self.base_dir.rglob("*.json*")
            if f not in self.verified_files
        ]
        
        # Process in batches
        results = []
        for i in range(0, len(unverified_files), self.batch_size):
            batch = unverified_files[i:i + self.batch_size]
            batch_results = await asyncio.gather(*[
                self._verify_file(f, compressed) 
                for f, compressed in batch
            ])
            results.extend(batch_results)
        
        return all(results)
    
    async def _verify_file(self, file_path: Path, compressed: bool) -> bool:
        """Verify integrity of a single file"""
        try:
            if compressed:
                content = await self._read_gzip(file_path)
            else:
                content = await self._read_file(file_path)
            
            # Verify JSON structure
            json.loads(content)
            
            self.verified_files.add(file_path)
            return True
            
        except Exception as e:
            logger.error(f"Corrupted file {file_path}: {str(e)}")
            return False
        
    async def restore_backup(self, backup_path: str):
        """Restore data from a backup with optimized I/O"""
        backup_dir = Path(backup_path)
        
        try:
            # Verify backup integrity
            manifest_file = backup_dir / "manifest.json"
            if not manifest_file.exists():
                raise ValueError("Invalid backup: missing manifest")
                
            # Create backup of current state before restore
            await self.create_backup()
            
            # Restore data concurrently
            await asyncio.gather(
                self._copy_directory(
                    backup_dir / "conversations",
                    self.base_dir / "conversations"
                ),
                self._copy_directory(
                    backup_dir / "knowledge_base",
                    self.base_dir / "knowledge_base"
                )
            )
            
            # Rebuild indices after restore
            await self.rebuild_indices()
            
            logger.info(f"Successfully restored from backup: {backup_path}")
            
        except Exception as e:
            logger.error(f"Restore failed: {str(e)}")
            raise

    # Optimized I/O helper methods
    async def _read_file(self, path: Path) -> str:
        """Read file content asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: path.read_text()
        )

    async def _write_file(self, path: Path, content: str):
        """Write file content asynchronously"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: path.write_text(content)
        )

    async def _write_bytes(self, path: Path, content: bytes):
        """Write binary content asynchronously"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: path.write_bytes(content)
        )

    async def _read_gzip(self, path: Path) -> str:
        """Read gzipped file content asynchronously"""
        return await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: gzip.open(path, 'rt').read()
        )

    async def _write_json(self, path: Path, data: dict):
        """Write JSON content asynchronously"""
        await self._write_file(path, json.dumps(data, indent=2, default=str))

    async def _remove_file(self, path: Path):
        """Remove file asynchronously"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            path.unlink
        )

    async def _remove_directory(self, path: Path):
        """Remove directory asynchronously"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: shutil.rmtree(path)
        )

    async def _copy_directory(self, src: Path, dst: Path):
        """Copy directory asynchronously"""
        await asyncio.get_event_loop().run_in_executor(
            self.executor,
            lambda: shutil.copytree(src, dst, dirs_exist_ok=True)
        )
