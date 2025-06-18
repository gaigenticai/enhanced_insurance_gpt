#!/usr/bin/env python3
"""
Insurance AI Agent System - Database Migration Manager
Production-ready database migration and initialization system
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncpg
import redis
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import click
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

class DatabaseManager:
    """Production-ready database manager for migrations and initialization"""
    
    def __init__(self, database_url: str, redis_url: str = None):
        self.database_url = database_url
        self.redis_url = redis_url
        self.engine = create_async_engine(database_url, echo=False)
        self.sync_engine = create_engine(database_url.replace('postgresql+asyncpg://', 'postgresql://'))
        self.redis_client = redis.from_url(redis_url) if redis_url else None
        
    async def check_connection(self) -> bool:
        """Check database connectivity"""
        try:
            async with self.engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            console.print("✅ Database connection successful", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Database connection failed: {e}", style="red")
            return False
    
    async def check_redis_connection(self) -> bool:
        """Check Redis connectivity"""
        if not self.redis_client:
            console.print("⚠️ Redis not configured", style="yellow")
            return False
            
        try:
            self.redis_client.ping()
            console.print("✅ Redis connection successful", style="green")
            return True
        except Exception as e:
            console.print(f"❌ Redis connection failed: {e}", style="red")
            return False
    
    async def create_database_if_not_exists(self, db_name: str):
        """Create database if it doesn't exist"""
        # Connect to postgres database to create target database
        postgres_url = self.database_url.rsplit('/', 1)[0] + '/postgres'
        engine = create_async_engine(postgres_url)
        
        try:
            async with engine.connect() as conn:
                # Check if database exists
                result = await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                    {"db_name": db_name}
                )
                
                if not result.fetchone():
                    # Create database
                    await conn.execute(text("COMMIT"))  # End transaction
                    await conn.execute(text(f"CREATE DATABASE {db_name}"))
                    console.print(f"✅ Created database: {db_name}", style="green")
                else:
                    console.print(f"ℹ️ Database already exists: {db_name}", style="blue")
                    
        except Exception as e:
            console.print(f"❌ Failed to create database: {e}", style="red")
            raise
        finally:
            await engine.dispose()
    
    async def run_schema_migration(self, schema_file: Path):
        """Run schema migration from SQL file"""
        if not schema_file.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_file}")
        
        console.print(f"🔄 Running schema migration: {schema_file.name}")
        
        with open(schema_file, 'r') as f:
            schema_sql = f.read()
        
        # Split SQL into individual statements
        statements = [stmt.strip() for stmt in schema_sql.split(';') if stmt.strip()]
        
        async with self.engine.connect() as conn:
            async with conn.begin():
                for i, statement in enumerate(statements):
                    if statement:
                        try:
                            await conn.execute(text(statement))
                            logger.debug(f"Executed statement {i+1}/{len(statements)}")
                        except Exception as e:
                            logger.error(f"Failed to execute statement {i+1}: {e}")
                            logger.error(f"Statement: {statement[:100]}...")
                            raise
        
        console.print("✅ Schema migration completed successfully", style="green")
    
    async def create_migration_table(self):
        """Create migrations tracking table"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id SERIAL PRIMARY KEY,
            version VARCHAR(255) UNIQUE NOT NULL,
            name VARCHAR(255) NOT NULL,
            applied_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            checksum VARCHAR(64)
        );
        """
        
        async with self.engine.connect() as conn:
            async with conn.begin():
                await conn.execute(text(create_table_sql))
        
        console.print("✅ Migration tracking table ready", style="green")
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migrations"""
        try:
            async with self.engine.connect() as conn:
                result = await conn.execute(
                    text("SELECT version FROM schema_migrations ORDER BY applied_at")
                )
                return [row[0] for row in result.fetchall()]
        except Exception:
            return []
    
    async def record_migration(self, version: str, name: str, checksum: str):
        """Record applied migration"""
        async with self.engine.connect() as conn:
            async with conn.begin():
                await conn.execute(
                    text("""
                        INSERT INTO schema_migrations (version, name, checksum)
                        VALUES (:version, :name, :checksum)
                        ON CONFLICT (version) DO NOTHING
                    """),
                    {"version": version, "name": name, "checksum": checksum}
                )
    
    async def seed_initial_data(self):
        """Seed database with initial data"""
        console.print("🌱 Seeding initial data...")
        
        # This is handled in the schema.sql file, but we can add additional seeding here
        seed_queries = [
            # Additional communication templates
            """
            INSERT INTO communication_templates (name, template_type, subject_template, content_template, organization_id)
            VALUES 
                ('claim_acknowledgment', 'email', 'Claim Acknowledgment - {{claim_number}}', 
                 'Dear {{customer_name}},\n\nWe have received your claim {{claim_number}} and are processing it. We will contact you within 24 hours with an update.\n\nBest regards,\nClaims Team', 
                 '00000000-0000-0000-0000-000000000001'),
                ('underwriting_decision', 'email', 'Underwriting Decision - {{submission_number}}', 
                 'Dear {{broker_name}},\n\nWe have completed the underwriting review for submission {{submission_number}}. Decision: {{decision}}.\n\nPlease contact us for any questions.\n\nBest regards,\nUnderwriting Team', 
                 '00000000-0000-0000-0000-000000000001')
            ON CONFLICT DO NOTHING;
            """,
            
            # Default prompts for AI agents
            """
            INSERT INTO prompts (name, version, prompt_text, agent_name, organization_id)
            VALUES 
                ('document_analysis_prompt', 1, 
                 'Analyze the following insurance document and extract key information including policy details, coverage amounts, exclusions, and any risk factors. Provide a structured response with confidence scores.',
                 'document-analysis-agent', '00000000-0000-0000-0000-000000000001'),
                ('claim_decision_prompt', 1,
                 'Based on the claim information and evidence provided, make a recommendation on claim approval. Consider policy coverage, exclusions, liability assessment, and fraud indicators. Provide reasoning for your decision.',
                 'decision-engine-agent', '00000000-0000-0000-0000-000000000001'),
                ('underwriting_risk_prompt', 1,
                 'Assess the underwriting risk for this submission based on the provided information. Consider industry risk factors, coverage amounts, loss history, and other relevant factors. Provide a risk score and recommendation.',
                 'decision-engine-agent', '00000000-0000-0000-0000-000000000001')
            ON CONFLICT (name, version) DO NOTHING;
            """
        ]
        
        async with self.engine.connect() as conn:
            async with conn.begin():
                for query in seed_queries:
                    await conn.execute(text(query))
        
        console.print("✅ Initial data seeded successfully", style="green")
    
    async def setup_redis_cache(self):
        """Setup Redis cache structure"""
        if not self.redis_client:
            console.print("⚠️ Skipping Redis setup - not configured", style="yellow")
            return
        
        console.print("🔄 Setting up Redis cache structure...")
        
        # Set up cache namespaces and default values
        cache_setup = {
            "config:max_file_size_mb": "50",
            "config:session_timeout_minutes": "1440",
            "config:rate_limit_per_minute": "1000",
            "config:workflow_timeout_seconds": "300",
            "health:api_gateway": "healthy",
            "health:database": "healthy",
            "health:redis": "healthy"
        }
        
        for key, value in cache_setup.items():
            self.redis_client.setex(key, 3600, value)  # 1 hour TTL
        
        console.print("✅ Redis cache structure setup complete", style="green")
    
    async def verify_installation(self) -> bool:
        """Verify database installation"""
        console.print("🔍 Verifying database installation...")
        
        verification_queries = [
            ("Organizations table", "SELECT COUNT(*) FROM organizations"),
            ("Users table", "SELECT COUNT(*) FROM users"),
            ("Workflows table", "SELECT COUNT(*) FROM workflows"),
            ("Claims table", "SELECT COUNT(*) FROM claims"),
            ("Knowledge base table", "SELECT COUNT(*) FROM knowledge_base"),
            ("Prompts table", "SELECT COUNT(*) FROM prompts"),
            ("System config", "SELECT COUNT(*) FROM system_config"),
            ("Feature flags", "SELECT COUNT(*) FROM feature_flags"),
        ]
        
        table = Table(title="Database Verification")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Count", style="yellow")
        
        all_passed = True
        
        async with self.engine.connect() as conn:
            for name, query in verification_queries:
                try:
                    result = await conn.execute(text(query))
                    count = result.scalar()
                    table.add_row(name, "✅ OK", str(count))
                except Exception as e:
                    table.add_row(name, f"❌ FAILED: {e}", "N/A")
                    all_passed = False
        
        console.print(table)
        
        if all_passed:
            console.print("✅ Database verification passed", style="green")
        else:
            console.print("❌ Database verification failed", style="red")
        
        return all_passed
    
    async def cleanup(self):
        """Cleanup database connections"""
        await self.engine.dispose()
        if self.redis_client:
            self.redis_client.close()

# CLI Commands
@click.group()
def cli():
    """Insurance AI Database Management CLI"""
    pass

@cli.command()
@click.option('--database-url', envvar='DATABASE_URL', required=True, help='Database URL')
@click.option('--redis-url', envvar='REDIS_URL', help='Redis URL')
@click.option('--force', is_flag=True, help='Force recreation of database')
def init(database_url: str, redis_url: str, force: bool):
    """Initialize the database with schema and initial data"""
    async def _init():
        db_manager = DatabaseManager(database_url, redis_url)
        
        try:
            # Extract database name from URL
            db_name = database_url.split('/')[-1]
            
            # Check connections
            if not await db_manager.check_connection():
                if force:
                    await db_manager.create_database_if_not_exists(db_name)
                else:
                    console.print("❌ Cannot connect to database. Use --force to create.", style="red")
                    return
            
            await db_manager.check_redis_connection()
            
            # Create migration tracking table
            await db_manager.create_migration_table()
            
            # Run schema migration
            schema_file = Path(__file__).parent / "schema.sql"
            await db_manager.run_schema_migration(schema_file)
            
            # Record migration
            await db_manager.record_migration(
                version="001",
                name="initial_schema",
                checksum="initial"
            )
            
            # Seed initial data
            await db_manager.seed_initial_data()
            
            # Setup Redis cache
            await db_manager.setup_redis_cache()
            
            # Verify installation
            if await db_manager.verify_installation():
                console.print("🎉 Database initialization completed successfully!", style="green bold")
            else:
                console.print("⚠️ Database initialization completed with warnings", style="yellow")
                
        except Exception as e:
            console.print(f"❌ Database initialization failed: {e}", style="red")
            logger.exception("Database initialization error")
            sys.exit(1)
        finally:
            await db_manager.cleanup()
    
    asyncio.run(_init())

@cli.command()
@click.option('--database-url', envvar='DATABASE_URL', required=True, help='Database URL')
def verify(database_url: str):
    """Verify database installation"""
    async def _verify():
        db_manager = DatabaseManager(database_url)
        
        try:
            if not await db_manager.check_connection():
                sys.exit(1)
            
            if not await db_manager.verify_installation():
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Verification failed: {e}", style="red")
            sys.exit(1)
        finally:
            await db_manager.cleanup()
    
    asyncio.run(_verify())

@cli.command()
@click.option('--database-url', envvar='DATABASE_URL', required=True, help='Database URL')
def status(database_url: str):
    """Show database status and applied migrations"""
    async def _status():
        db_manager = DatabaseManager(database_url)
        
        try:
            if not await db_manager.check_connection():
                sys.exit(1)
            
            migrations = await db_manager.get_applied_migrations()
            
            table = Table(title="Applied Migrations")
            table.add_column("Version", style="cyan")
            table.add_column("Applied At", style="green")
            
            if migrations:
                for migration in migrations:
                    table.add_row(migration, "✅ Applied")
            else:
                table.add_row("No migrations", "❌ Not applied")
            
            console.print(table)
            
        except Exception as e:
            console.print(f"❌ Status check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await db_manager.cleanup()
    
    asyncio.run(_status())

@cli.command()
@click.option('--database-url', envvar='DATABASE_URL', required=True, help='Database URL')
@click.option('--redis-url', envvar='REDIS_URL', help='Redis URL')
def health(database_url: str, redis_url: str):
    """Check health of database and cache systems"""
    async def _health():
        db_manager = DatabaseManager(database_url, redis_url)
        
        try:
            db_healthy = await db_manager.check_connection()
            redis_healthy = await db_manager.check_redis_connection()
            
            if db_healthy and (redis_healthy or not redis_url):
                console.print("🟢 All systems healthy", style="green bold")
                sys.exit(0)
            else:
                console.print("🔴 Some systems unhealthy", style="red bold")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"❌ Health check failed: {e}", style="red")
            sys.exit(1)
        finally:
            await db_manager.cleanup()
    
    asyncio.run(_health())

if __name__ == "__main__":
    cli()

