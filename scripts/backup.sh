#!/bin/bash

# Insurance AI Agent System - Backup Script
# Production-ready backup automation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"
BACKUP_DIR="$PROJECT_DIR/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_NAME="insurance_ai_backup_$TIMESTAMP"

# Load environment variables
if [[ -f "$ENV_FILE" ]]; then
    source "$ENV_FILE"
fi

# Default values
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_DB=${POSTGRES_DB:-insurance_ai}
BACKUP_RETENTION_DAYS=${BACKUP_RETENTION_DAYS:-30}

# Functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] INFO: $1${NC}"
}

# Create backup directory
create_backup_dir() {
    log "Creating backup directory..."
    mkdir -p "$BACKUP_DIR/$BACKUP_NAME"
}

# Backup PostgreSQL database
backup_database() {
    log "Backing up PostgreSQL database..."
    
    cd "$PROJECT_DIR"
    
    # Create database dump
    docker-compose exec -T postgres pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --clean --if-exists > "$BACKUP_DIR/$BACKUP_NAME/database.sql"
    
    if [[ $? -eq 0 ]]; then
        log "Database backup completed successfully."
    else
        error "Database backup failed."
    fi
}

# Backup Redis data
backup_redis() {
    log "Backing up Redis data..."
    
    cd "$PROJECT_DIR"
    
    # Create Redis dump
    docker-compose exec -T redis redis-cli --rdb - > "$BACKUP_DIR/$BACKUP_NAME/redis.rdb"
    
    if [[ $? -eq 0 ]]; then
        log "Redis backup completed successfully."
    else
        warn "Redis backup failed, continuing..."
    fi
}

# Backup application files
backup_files() {
    log "Backing up application files..."
    
    # Backup uploads directory
    if [[ -d "$PROJECT_DIR/uploads" ]]; then
        cp -r "$PROJECT_DIR/uploads" "$BACKUP_DIR/$BACKUP_NAME/"
        log "Uploads directory backed up."
    fi
    
    # Backup data directory
    if [[ -d "$PROJECT_DIR/data" ]]; then
        cp -r "$PROJECT_DIR/data" "$BACKUP_DIR/$BACKUP_NAME/"
        log "Data directory backed up."
    fi
    
    # Backup logs directory
    if [[ -d "$PROJECT_DIR/logs" ]]; then
        cp -r "$PROJECT_DIR/logs" "$BACKUP_DIR/$BACKUP_NAME/"
        log "Logs directory backed up."
    fi
    
    # Backup configuration files
    cp "$PROJECT_DIR/.env" "$BACKUP_DIR/$BACKUP_NAME/" 2>/dev/null || warn ".env file not found"
    cp "$PROJECT_DIR/docker-compose.yml" "$BACKUP_DIR/$BACKUP_NAME/"
    
    log "Application files backup completed."
}

# Backup Docker volumes
backup_volumes() {
    log "Backing up Docker volumes..."
    
    cd "$PROJECT_DIR"
    
    # Get list of volumes
    volumes=$(docker-compose config --volumes)
    
    for volume in $volumes; do
        info "Backing up volume: $volume"
        
        # Create volume backup
        docker run --rm \
            -v "${PROJECT_DIR}_${volume}:/data" \
            -v "$BACKUP_DIR/$BACKUP_NAME:/backup" \
            alpine tar czf "/backup/${volume}.tar.gz" -C /data .
        
        if [[ $? -eq 0 ]]; then
            info "Volume $volume backed up successfully."
        else
            warn "Failed to backup volume $volume."
        fi
    done
    
    log "Docker volumes backup completed."
}

# Create backup metadata
create_metadata() {
    log "Creating backup metadata..."
    
    cat > "$BACKUP_DIR/$BACKUP_NAME/metadata.json" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "timestamp": "$TIMESTAMP",
    "date": "$(date -Iseconds)",
    "version": "1.0.0",
    "components": {
        "database": true,
        "redis": true,
        "files": true,
        "volumes": true
    },
    "database_info": {
        "type": "postgresql",
        "database": "$POSTGRES_DB",
        "user": "$POSTGRES_USER"
    },
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "architecture": "$(uname -m)"
    }
}
EOF
    
    log "Backup metadata created."
}

# Compress backup
compress_backup() {
    log "Compressing backup..."
    
    cd "$BACKUP_DIR"
    tar czf "${BACKUP_NAME}.tar.gz" "$BACKUP_NAME"
    
    if [[ $? -eq 0 ]]; then
        # Remove uncompressed directory
        rm -rf "$BACKUP_NAME"
        log "Backup compressed successfully: ${BACKUP_NAME}.tar.gz"
    else
        error "Failed to compress backup."
    fi
}

# Upload to S3 (if configured)
upload_to_s3() {
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && [[ -n "${BACKUP_S3_BUCKET:-}" ]]; then
        log "Uploading backup to S3..."
        
        # Check if AWS CLI is available
        if command -v aws &> /dev/null; then
            aws s3 cp "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" "s3://${BACKUP_S3_BUCKET}/backups/${BACKUP_NAME}.tar.gz"
            
            if [[ $? -eq 0 ]]; then
                log "Backup uploaded to S3 successfully."
            else
                warn "Failed to upload backup to S3."
            fi
        else
            warn "AWS CLI not found. Skipping S3 upload."
        fi
    else
        info "S3 configuration not found. Skipping S3 upload."
    fi
}

# Clean old backups
clean_old_backups() {
    log "Cleaning old backups..."
    
    # Remove local backups older than retention period
    find "$BACKUP_DIR" -name "insurance_ai_backup_*.tar.gz" -mtime +$BACKUP_RETENTION_DAYS -delete
    
    # Clean S3 backups if configured
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && [[ -n "${BACKUP_S3_BUCKET:-}" ]]; then
        if command -v aws &> /dev/null; then
            # List and delete old S3 backups
            aws s3 ls "s3://${BACKUP_S3_BUCKET}/backups/" | while read -r line; do
                backup_date=$(echo "$line" | awk '{print $1}')
                backup_file=$(echo "$line" | awk '{print $4}')
                
                if [[ -n "$backup_date" ]] && [[ -n "$backup_file" ]]; then
                    # Calculate age in days
                    backup_timestamp=$(date -d "$backup_date" +%s)
                    current_timestamp=$(date +%s)
                    age_days=$(( (current_timestamp - backup_timestamp) / 86400 ))
                    
                    if [[ $age_days -gt $BACKUP_RETENTION_DAYS ]]; then
                        info "Deleting old S3 backup: $backup_file"
                        aws s3 rm "s3://${BACKUP_S3_BUCKET}/backups/$backup_file"
                    fi
                fi
            done
        fi
    fi
    
    log "Old backups cleaned up."
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    # Test if backup file exists and is readable
    if [[ -f "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" ]]; then
        # Test if tar file is valid
        if tar tzf "$BACKUP_DIR/${BACKUP_NAME}.tar.gz" > /dev/null 2>&1; then
            log "Backup integrity verification passed."
        else
            error "Backup integrity verification failed - corrupted archive."
        fi
    else
        error "Backup file not found."
    fi
}

# Send notification (if configured)
send_notification() {
    local status=$1
    local message=$2
    
    # Email notification (if SendGrid is configured)
    if [[ -n "${SENDGRID_API_KEY:-}" ]] && [[ -n "${SENDGRID_FROM_EMAIL:-}" ]]; then
        # This would integrate with your notification system
        info "Backup notification: $status - $message"
    fi
    
    # Slack notification (if webhook is configured)
    if [[ -n "${SLACK_WEBHOOK_URL:-}" ]]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"Insurance AI Backup $status: $message\"}" \
            "$SLACK_WEBHOOK_URL" > /dev/null 2>&1 || true
    fi
}

# Main backup function
backup() {
    log "Starting backup process..."
    
    local start_time=$(date +%s)
    
    # Check if services are running
    cd "$PROJECT_DIR"
    if ! docker-compose ps | grep -q "Up"; then
        warn "Some services are not running. Backup may be incomplete."
    fi
    
    # Create backup
    create_backup_dir
    backup_database
    backup_redis
    backup_files
    backup_volumes
    create_metadata
    compress_backup
    upload_to_s3
    verify_backup
    clean_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Backup completed successfully in ${duration} seconds."
    log "Backup location: $BACKUP_DIR/${BACKUP_NAME}.tar.gz"
    
    # Send success notification
    send_notification "SUCCESS" "Backup completed in ${duration} seconds"
}

# Restore function
restore() {
    local backup_file=$1
    
    if [[ -z "$backup_file" ]]; then
        error "Please specify backup file to restore."
    fi
    
    if [[ ! -f "$backup_file" ]]; then
        error "Backup file not found: $backup_file"
    fi
    
    log "Starting restore process from: $backup_file"
    
    # Extract backup
    local restore_dir="/tmp/restore_$(date +%s)"
    mkdir -p "$restore_dir"
    tar xzf "$backup_file" -C "$restore_dir"
    
    # Find backup directory
    local backup_content_dir=$(find "$restore_dir" -name "insurance_ai_backup_*" -type d | head -1)
    
    if [[ -z "$backup_content_dir" ]]; then
        error "Invalid backup file structure."
    fi
    
    # Stop services
    cd "$PROJECT_DIR"
    docker-compose down
    
    # Restore database
    if [[ -f "$backup_content_dir/database.sql" ]]; then
        log "Restoring database..."
        docker-compose up -d postgres
        sleep 10
        docker-compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" < "$backup_content_dir/database.sql"
    fi
    
    # Restore Redis
    if [[ -f "$backup_content_dir/redis.rdb" ]]; then
        log "Restoring Redis data..."

        # Stop the Redis container so we can replace the dump file safely
        docker-compose stop redis

        # Copy the backed up RDB file into the Redis data directory
        docker cp "$backup_content_dir/redis.rdb" insurance-redis:/data/dump.rdb

        # Restart Redis to load the new dump
        docker-compose start redis
        sleep 5

        # Verify Redis started correctly and loaded the dump
        if docker-compose exec -T redis redis-cli ping | grep -q PONG; then
            log "Redis restore completed successfully."
        else
            warn "Redis restore verification failed."
        fi
    fi
    
    # Restore files
    if [[ -d "$backup_content_dir/uploads" ]]; then
        log "Restoring uploads..."
        rm -rf "$PROJECT_DIR/uploads"
        cp -r "$backup_content_dir/uploads" "$PROJECT_DIR/"
    fi
    
    if [[ -d "$backup_content_dir/data" ]]; then
        log "Restoring data..."
        rm -rf "$PROJECT_DIR/data"
        cp -r "$backup_content_dir/data" "$PROJECT_DIR/"
    fi
    
    # Start services
    docker-compose up -d
    
    # Cleanup
    rm -rf "$restore_dir"
    
    log "Restore completed successfully."
    
    # Send notification
    send_notification "RESTORE" "System restored from backup"
}

# List available backups
list_backups() {
    log "Available local backups:"
    ls -la "$BACKUP_DIR"/insurance_ai_backup_*.tar.gz 2>/dev/null || info "No local backups found."
    
    # List S3 backups if configured
    if [[ -n "${AWS_ACCESS_KEY_ID:-}" ]] && [[ -n "${AWS_SECRET_ACCESS_KEY:-}" ]] && [[ -n "${BACKUP_S3_BUCKET:-}" ]]; then
        if command -v aws &> /dev/null; then
            log "Available S3 backups:"
            aws s3 ls "s3://${BACKUP_S3_BUCKET}/backups/" || info "No S3 backups found."
        fi
    fi
}

# Command line options
case "${1:-backup}" in
    "backup")
        backup
        ;;
    "restore")
        restore "${2:-}"
        ;;
    "list")
        list_backups
        ;;
    "clean")
        clean_old_backups
        ;;
    "help")
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  backup           - Create backup (default)"
        echo "  restore <file>   - Restore from backup file"
        echo "  list             - List available backups"
        echo "  clean            - Clean old backups"
        echo "  help             - Show this help"
        ;;
    *)
        error "Unknown command: $1. Use 'help' for available commands."
        ;;
esac

