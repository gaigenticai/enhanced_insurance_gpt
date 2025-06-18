"""
Insurance AI Agent System - Database Connection and Configuration
Production-ready database utilities with connection pooling, health checks, and monitoring
"""

import os
import asyncio
import logging
from typing import Optional, Dict, Any, AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as redis
from sqlalchemy import create_engine, event, pool, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker, declarative_base
import structlog
from prometheus_client import Counter, Histogram, Gauge
import time
from contextlib import asynccontextmanager

# Setup structured logging
logger = structlog.get_logger(__name__)

# Prometheus metrics
db_connections_total = Counter('db_connections_total', 'Total database connections', ['status'])
db_query_duration = Histogram('db_query_duration_seconds', 'Database query duration')
db_active_connections = Gauge('db_active_connections', 'Active database connections')
redis_operations_total = Counter('redis_operations_total', 'Total Redis operations', ['operation', 'status'])

# SQLAlchemy Base
Base = declarative_base()

class DatabaseConfig:
    """Database configuration with environment variable support"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL', 'postgresql+asyncpg://user:password@localhost:5432/insurance_ai')
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        
        # Connection pool settings
        self.pool_size = int(os.getenv('DB_POOL_SIZE', '20'))
        self.max_overflow = int(os.getenv('DB_MAX_OVERFLOW', '30'))
        self.pool_timeout = int(os.getenv('DB_POOL_TIMEOUT', '30'))
        self.pool_recycle = int(os.getenv('DB_POOL_RECYCLE', '3600'))
        
        # Redis settings
        self.redis_max_connections = int(os.getenv('REDIS_MAX_CONNECTIONS', '50'))
        self.redis_retry_on_timeout = os.getenv('REDIS_RETRY_ON_TIMEOUT', 'true').lower() == 'true'
        
        # Health check settings
        self.health_check_interval = int(os.getenv('DB_HEALTH_CHECK_INTERVAL', '30'))
        
        # Query timeout settings
        self.query_timeout = int(os.getenv('DB_QUERY_TIMEOUT', '30'))

class DatabaseManager:
    """Production-ready database manager with connection pooling and monitoring"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self._engine = None
        self._async_session_factory = None
        self._redis_pool = None
        self._health_check_task = None
        self._is_healthy = True
        
    async def initialize(self):
        """Initialize database connections and start health checks"""
        try:
            # Create async engine with connection pooling
            self._engine = create_async_engine(
                self.config.database_url,
                pool_size=self.config.pool_size,
                max_overflow=self.config.max_overflow,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                pool_pre_ping=True,
                echo=False,
                future=True
            )
            
            # Create session factory
            self._async_session_factory = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Initialize Redis connection pool
            self._redis_pool = redis.ConnectionPool.from_url(
                self.config.redis_url,
                max_connections=self.config.redis_max_connections,
                retry_on_timeout=self.config.redis_retry_on_timeout,
                decode_responses=True
            )
            
            # Test connections
            await self._test_database_connection()
            await self._test_redis_connection()
            
            # Start health check task
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info("Database manager initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize database manager", error=str(e))
            raise
    
    async def _test_database_connection(self):
        """Test database connectivity"""
        try:
            async with self._engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            db_connections_total.labels(status='success').inc()
            logger.info("Database connection test successful")
        except Exception as e:
            db_connections_total.labels(status='error').inc()
            logger.error("Database connection test failed", error=str(e))
            raise
    
    async def _test_redis_connection(self):
        """Test Redis connectivity"""
        try:
            redis_client = redis.Redis(connection_pool=self._redis_pool)
            await redis_client.ping()
            redis_operations_total.labels(operation='ping', status='success').inc()
            logger.info("Redis connection test successful")
        except Exception as e:
            redis_operations_total.labels(operation='ping', status='error').inc()
            logger.error("Redis connection test failed", error=str(e))
            raise
    
    async def _health_check_loop(self):
        """Continuous health check loop"""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                
                # Check database health
                start_time = time.time()
                async with self._engine.connect() as conn:
                    await conn.execute(text("SELECT 1"))
                db_query_duration.observe(time.time() - start_time)
                
                # Check Redis health
                redis_client = redis.Redis(connection_pool=self._redis_pool)
                await redis_client.ping()
                
                # Update metrics
                pool = self._engine.pool
                db_active_connections.set(pool.checkedout())
                
                if not self._is_healthy:
                    self._is_healthy = True
                    logger.info("Database health check: All systems healthy")
                    
            except Exception as e:
                if self._is_healthy:
                    self._is_healthy = False
                    logger.error("Database health check failed", error=str(e))
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup"""
        if not self._async_session_factory:
            raise RuntimeError("Database manager not initialized")
        
        session = self._async_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
    
    @asynccontextmanager
    async def get_redis(self) -> AsyncGenerator[redis.Redis, None]:
        """Get Redis client with automatic cleanup"""
        if not self._redis_pool:
            raise RuntimeError("Redis pool not initialized")
        
        client = redis.Redis(connection_pool=self._redis_pool)
        try:
            yield client
        finally:
            await client.close()
    
    async def execute_query(self, query: str, params: Dict[str, Any] = None) -> Any:
        """Execute raw SQL query with monitoring"""
        start_time = time.time()
        try:
            async with self._engine.connect() as conn:
                result = await conn.execute(query, params or {})
                db_query_duration.observe(time.time() - start_time)
                return result
        except Exception as e:
            logger.error("Query execution failed", query=query, error=str(e))
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            async with self._engine.connect() as conn:
                # Get connection pool stats
                pool = self._engine.pool
                pool_stats = {
                    'pool_size': pool.size(),
                    'checked_out': pool.checkedout(),
                    'overflow': pool.overflow(),
                    'checked_in': pool.checkedin()
                }
                
                # Get database stats
                db_stats_query = """
                SELECT 
                    pg_database_size(current_database()) as db_size,
                    (SELECT count(*) FROM pg_stat_activity WHERE state = 'active') as active_connections,
                    (SELECT count(*) FROM pg_stat_activity) as total_connections
                """
                result = await conn.execute(db_stats_query)
                row = result.fetchone()
                
                db_stats = {
                    'database_size_bytes': row[0],
                    'active_connections': row[1],
                    'total_connections': row[2]
                }
                
                return {
                    'pool_stats': pool_stats,
                    'database_stats': db_stats,
                    'is_healthy': self._is_healthy
                }
                
        except Exception as e:
            logger.error("Failed to get database stats", error=str(e))
            return {'error': str(e), 'is_healthy': False}
    
    async def get_redis_stats(self) -> Dict[str, Any]:
        """Get Redis statistics"""
        try:
            async with self.get_redis() as redis_client:
                info = await redis_client.info()
                return {
                    'connected_clients': info.get('connected_clients', 0),
                    'used_memory': info.get('used_memory', 0),
                    'used_memory_human': info.get('used_memory_human', '0B'),
                    'keyspace_hits': info.get('keyspace_hits', 0),
                    'keyspace_misses': info.get('keyspace_misses', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
        except Exception as e:
            logger.error("Failed to get Redis stats", error=str(e))
            return {'error': str(e)}
    
    async def cleanup(self):
        """Cleanup database connections and tasks"""
        try:
            if self._health_check_task:
                self._health_check_task.cancel()
                try:
                    await self._health_check_task
                except asyncio.CancelledError:
                    pass
            
            if self._engine:
                await self._engine.dispose()
            
            if self._redis_pool:
                await self._redis_pool.disconnect()
            
            logger.info("Database manager cleanup completed")
            
        except Exception as e:
            logger.error("Error during database cleanup", error=str(e))

class CacheManager:
    """Redis cache manager with TTL and namespace support"""
    
    def __init__(self, redis_client: redis.Redis, default_ttl: int = 3600):
        self.redis_client = redis_client
        self.default_ttl = default_ttl
    
    async def get(self, key: str, namespace: str = None) -> Optional[str]:
        """Get value from cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        try:
            value = await self.redis_client.get(full_key)
            redis_operations_total.labels(operation='get', status='success').inc()
            return value
        except Exception as e:
            redis_operations_total.labels(operation='get', status='error').inc()
            logger.error("Cache get failed", key=full_key, error=str(e))
            return None
    
    async def set(self, key: str, value: str, ttl: int = None, namespace: str = None) -> bool:
        """Set value in cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        ttl = ttl or self.default_ttl
        try:
            await self.redis_client.setex(full_key, ttl, value)
            redis_operations_total.labels(operation='set', status='success').inc()
            return True
        except Exception as e:
            redis_operations_total.labels(operation='set', status='error').inc()
            logger.error("Cache set failed", key=full_key, error=str(e))
            return False
    
    async def delete(self, key: str, namespace: str = None) -> bool:
        """Delete value from cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        try:
            result = await self.redis_client.delete(full_key)
            redis_operations_total.labels(operation='delete', status='success').inc()
            return result > 0
        except Exception as e:
            redis_operations_total.labels(operation='delete', status='error').inc()
            logger.error("Cache delete failed", key=full_key, error=str(e))
            return False
    
    async def exists(self, key: str, namespace: str = None) -> bool:
        """Check if key exists in cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        try:
            result = await self.redis_client.exists(full_key)
            redis_operations_total.labels(operation='exists', status='success').inc()
            return result > 0
        except Exception as e:
            redis_operations_total.labels(operation='exists', status='error').inc()
            logger.error("Cache exists check failed", key=full_key, error=str(e))
            return False
    
    async def increment(self, key: str, amount: int = 1, namespace: str = None) -> Optional[int]:
        """Increment counter in cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        try:
            result = await self.redis_client.incrby(full_key, amount)
            redis_operations_total.labels(operation='increment', status='success').inc()
            return result
        except Exception as e:
            redis_operations_total.labels(operation='increment', status='error').inc()
            logger.error("Cache increment failed", key=full_key, error=str(e))
            return None
    
    async def set_hash(self, key: str, mapping: Dict[str, str], ttl: int = None, namespace: str = None) -> bool:
        """Set hash in cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        ttl = ttl or self.default_ttl
        try:
            await self.redis_client.hset(full_key, mapping=mapping)
            await self.redis_client.expire(full_key, ttl)
            redis_operations_total.labels(operation='hset', status='success').inc()
            return True
        except Exception as e:
            redis_operations_total.labels(operation='hset', status='error').inc()
            logger.error("Cache hash set failed", key=full_key, error=str(e))
            return False
    
    async def get_hash(self, key: str, namespace: str = None) -> Optional[Dict[str, str]]:
        """Get hash from cache"""
        full_key = f"{namespace}:{key}" if namespace else key
        try:
            result = await self.redis_client.hgetall(full_key)
            redis_operations_total.labels(operation='hgetall', status='success').inc()
            return result if result else None
        except Exception as e:
            redis_operations_total.labels(operation='hgetall', status='error').inc()
            logger.error("Cache hash get failed", key=full_key, error=str(e))
            return None

# Global database manager instance
db_manager: Optional[DatabaseManager] = None

async def init_database() -> DatabaseManager:
    """Initialize the global database manager if needed.

    This provides backward compatibility with older modules that
    expected an ``init_database`` function.
    """
    return await get_database_manager()

async def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    global db_manager
    if not db_manager:
        db_manager = DatabaseManager()
        await db_manager.initialize()
    return db_manager

@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for getting database session"""
    manager = await get_database_manager()
    async with manager.get_session() as session:
        yield session

@asynccontextmanager
async def get_redis_client() -> AsyncGenerator[redis.Redis, None]:
    """Dependency for getting Redis client"""
    manager = await get_database_manager()
    async with manager.get_redis() as client:
        yield client

@asynccontextmanager
async def get_cache_manager() -> AsyncGenerator[CacheManager, None]:
    """Dependency for getting cache manager"""
    manager = await get_database_manager()
    async with manager.get_redis() as client:
        yield CacheManager(client)

# Health check functions
async def check_database_health() -> Dict[str, Any]:
    """Check database health"""
    try:
        manager = await get_database_manager()
        return await manager.get_database_stats()
    except Exception as e:
        return {'error': str(e), 'is_healthy': False}

async def check_redis_health() -> Dict[str, Any]:
    """Check Redis health"""
    try:
        manager = await get_database_manager()
        return await manager.get_redis_stats()
    except Exception as e:
        return {'error': str(e), 'is_healthy': False}

# Cleanup function for graceful shutdown
async def cleanup_database():
    """Cleanup database connections on shutdown"""
    global db_manager
    if db_manager:
        await db_manager.cleanup()
        db_manager = None

