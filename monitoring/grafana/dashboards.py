"""
Grafana Dashboard Configuration - Production Ready
Comprehensive monitoring dashboards for Insurance AI Agent System
"""

import json
from typing import Dict, List, Any

# Main System Overview Dashboard
SYSTEM_OVERVIEW_DASHBOARD = {
    "dashboard": {
        "id": None,
        "title": "Insurance AI System - Overview",
        "tags": ["insurance", "ai", "overview"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "System Health",
                "type": "stat",
                "targets": [
                    {
                        "expr": "up{job=\"insurance-ai-backend\"}",
                        "legendFormat": "Backend Status"
                    },
                    {
                        "expr": "up{job=\"insurance-ai-frontend\"}",
                        "legendFormat": "Frontend Status"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "thresholds"
                        },
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "green", "value": 1}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Request Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total{job=\"insurance-ai-backend\"}[5m])",
                        "legendFormat": "{{method}} {{endpoint}}"
                    }
                ],
                "yAxes": [
                    {"label": "Requests/sec", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "id": 3,
                "title": "Response Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job=\"insurance-ai-backend\"}[5m]))",
                        "legendFormat": "95th percentile"
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(http_request_duration_seconds_bucket{job=\"insurance-ai-backend\"}[5m]))",
                        "legendFormat": "50th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 4,
                "title": "Error Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(http_requests_total{job=\"insurance-ai-backend\",status=~\"4..|5..\"}[5m]) / rate(http_requests_total{job=\"insurance-ai-backend\"}[5m])",
                        "legendFormat": "Error Rate"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 1},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": 5,
                "title": "Active Policies",
                "type": "stat",
                "targets": [
                    {
                        "expr": "active_policies_current",
                        "legendFormat": "Active Policies"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
            },
            {
                "id": 6,
                "title": "Claims Processed Today",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(claims_processed_total[24h])",
                        "legendFormat": "Claims Today"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
            },
            {
                "id": 7,
                "title": "Underwriting Decisions Today",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(underwriting_decisions_total[24h])",
                        "legendFormat": "Decisions Today"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16}
            },
            {
                "id": 8,
                "title": "AI Agent Status",
                "type": "stat",
                "targets": [
                    {
                        "expr": "ai_agent_health_status",
                        "legendFormat": "{{agent_name}}"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {
                            "mode": "thresholds"
                        },
                        "thresholds": {
                            "steps": [
                                {"color": "red", "value": 0},
                                {"color": "green", "value": 1}
                            ]
                        }
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16}
            }
        ],
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "refresh": "30s"
    }
}

# AI Agents Performance Dashboard
AI_AGENTS_DASHBOARD = {
    "dashboard": {
        "id": None,
        "title": "Insurance AI Agents - Performance",
        "tags": ["insurance", "ai", "agents"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "Agent Processing Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(agent_processing_duration_seconds_bucket[5m]))",
                        "legendFormat": "{{agent_name}} - 95th percentile"
                    }
                ],
                "yAxes": [
                    {"label": "Seconds", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Agent Success Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(agent_operations_total{status=\"success\"}[5m]) / rate(agent_operations_total[5m])",
                        "legendFormat": "{{agent_name}} Success Rate"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 1},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "id": 3,
                "title": "Document Analysis Performance",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(documents_processed_total[5m])",
                        "legendFormat": "Documents/sec"
                    },
                    {
                        "expr": "histogram_quantile(0.95, rate(document_processing_duration_seconds_bucket[5m]))",
                        "legendFormat": "Processing Time (95th)"
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 4,
                "title": "Evidence Processing Queue",
                "type": "graph",
                "targets": [
                    {
                        "expr": "evidence_processing_queue_size",
                        "legendFormat": "Queue Size"
                    },
                    {
                        "expr": "rate(evidence_processed_total[5m])",
                        "legendFormat": "Processing Rate"
                    }
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": 5,
                "title": "Risk Assessment Accuracy",
                "type": "stat",
                "targets": [
                    {
                        "expr": "risk_assessment_accuracy_score",
                        "legendFormat": "Accuracy Score"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percentunit",
                        "min": 0,
                        "max": 1
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
            },
            {
                "id": 6,
                "title": "Fraud Detection Rate",
                "type": "stat",
                "targets": [
                    {
                        "expr": "fraud_detection_rate",
                        "legendFormat": "Detection Rate"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percentunit"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
            },
            {
                "id": 7,
                "title": "Communication Agent Messages",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(messages_sent_total[24h])",
                        "legendFormat": "Messages Today"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16}
            },
            {
                "id": 8,
                "title": "Integration API Calls",
                "type": "stat",
                "targets": [
                    {
                        "expr": "increase(external_api_calls_total[24h])",
                        "legendFormat": "API Calls Today"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16}
            }
        ],
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "refresh": "30s"
    }
}

# Business Metrics Dashboard
BUSINESS_METRICS_DASHBOARD = {
    "dashboard": {
        "id": None,
        "title": "Insurance Business Metrics",
        "tags": ["insurance", "business", "metrics"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "Policy Conversion Rate",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(policies_bound_total[1h]) / rate(quotes_generated_total[1h])",
                        "legendFormat": "Conversion Rate"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 1},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Claims Settlement Time",
                "type": "graph",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.95, rate(claim_settlement_duration_days_bucket[24h]))",
                        "legendFormat": "95th percentile"
                    },
                    {
                        "expr": "histogram_quantile(0.50, rate(claim_settlement_duration_days_bucket[24h]))",
                        "legendFormat": "Median"
                    }
                ],
                "yAxes": [
                    {"label": "Days", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "id": 3,
                "title": "Premium Volume",
                "type": "graph",
                "targets": [
                    {
                        "expr": "increase(premium_collected_total[24h])",
                        "legendFormat": "Daily Premium"
                    }
                ],
                "yAxes": [
                    {"label": "USD", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 4,
                "title": "Claims Ratio",
                "type": "graph",
                "targets": [
                    {
                        "expr": "increase(claims_paid_total[30d]) / increase(premium_collected_total[30d])",
                        "legendFormat": "30-day Claims Ratio"
                    }
                ],
                "yAxes": [
                    {"label": "Ratio", "min": 0, "max": 2},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": 5,
                "title": "Customer Satisfaction",
                "type": "stat",
                "targets": [
                    {
                        "expr": "avg(customer_satisfaction_score)",
                        "legendFormat": "Satisfaction Score"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short",
                        "min": 1,
                        "max": 5
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
            },
            {
                "id": 6,
                "title": "Straight Through Processing",
                "type": "stat",
                "targets": [
                    {
                        "expr": "rate(automated_decisions_total[24h]) / rate(total_decisions_total[24h])",
                        "legendFormat": "STP Rate"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percentunit"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
            },
            {
                "id": 7,
                "title": "Average Quote Time",
                "type": "stat",
                "targets": [
                    {
                        "expr": "histogram_quantile(0.50, rate(quote_generation_duration_seconds_bucket[1h]))",
                        "legendFormat": "Median Quote Time"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "s"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16}
            },
            {
                "id": 8,
                "title": "Policy Retention Rate",
                "type": "stat",
                "targets": [
                    {
                        "expr": "1 - (rate(policies_cancelled_total[30d]) / rate(policies_renewed_total[30d]))",
                        "legendFormat": "Retention Rate"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percentunit"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16}
            }
        ],
        "time": {
            "from": "now-24h",
            "to": "now"
        },
        "refresh": "5m"
    }
}

# Infrastructure Dashboard
INFRASTRUCTURE_DASHBOARD = {
    "dashboard": {
        "id": None,
        "title": "Infrastructure Monitoring",
        "tags": ["infrastructure", "system"],
        "timezone": "browser",
        "panels": [
            {
                "id": 1,
                "title": "CPU Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
                        "legendFormat": "{{instance}}"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 100},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
            },
            {
                "id": 2,
                "title": "Memory Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100",
                        "legendFormat": "{{instance}}"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 100},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
            },
            {
                "id": 3,
                "title": "Disk Usage",
                "type": "graph",
                "targets": [
                    {
                        "expr": "(1 - (node_filesystem_avail_bytes{fstype!=\"tmpfs\"} / node_filesystem_size_bytes{fstype!=\"tmpfs\"})) * 100",
                        "legendFormat": "{{instance}} {{mountpoint}}"
                    }
                ],
                "yAxes": [
                    {"label": "Percentage", "min": 0, "max": 100},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
            },
            {
                "id": 4,
                "title": "Network I/O",
                "type": "graph",
                "targets": [
                    {
                        "expr": "rate(node_network_receive_bytes_total[5m])",
                        "legendFormat": "{{instance}} {{device}} RX"
                    },
                    {
                        "expr": "rate(node_network_transmit_bytes_total[5m])",
                        "legendFormat": "{{instance}} {{device}} TX"
                    }
                ],
                "yAxes": [
                    {"label": "Bytes/sec", "min": 0},
                    {"show": False}
                ],
                "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
            },
            {
                "id": 5,
                "title": "Database Connections",
                "type": "stat",
                "targets": [
                    {
                        "expr": "pg_stat_database_numbackends",
                        "legendFormat": "Active Connections"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 0, "y": 16}
            },
            {
                "id": 6,
                "title": "Redis Memory Usage",
                "type": "stat",
                "targets": [
                    {
                        "expr": "redis_memory_used_bytes / redis_memory_max_bytes * 100",
                        "legendFormat": "Memory Usage %"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "percent"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 6, "y": 16}
            },
            {
                "id": 7,
                "title": "Container Status",
                "type": "stat",
                "targets": [
                    {
                        "expr": "count(container_last_seen{name=~\"insurance.*\"})",
                        "legendFormat": "Running Containers"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 12, "y": 16}
            },
            {
                "id": 8,
                "title": "Load Average",
                "type": "stat",
                "targets": [
                    {
                        "expr": "node_load1",
                        "legendFormat": "1m Load"
                    }
                ],
                "fieldConfig": {
                    "defaults": {
                        "color": {"mode": "palette-classic"},
                        "unit": "short"
                    }
                },
                "gridPos": {"h": 4, "w": 6, "x": 18, "y": 16}
            }
        ],
        "time": {
            "from": "now-1h",
            "to": "now"
        },
        "refresh": "30s"
    }
}

# Grafana Configuration
GRAFANA_CONFIG = {
    "apiVersion": 1,
    "datasources": [
        {
            "name": "Prometheus",
            "type": "prometheus",
            "access": "proxy",
            "url": "http://prometheus:9090",
            "isDefault": True,
            "editable": True
        },
        {
            "name": "Loki",
            "type": "loki",
            "access": "proxy",
            "url": "http://loki:3100",
            "editable": True
        }
    ]
}

# Alert Rules
ALERT_RULES = {
    "groups": [
        {
            "name": "insurance_ai_alerts",
            "rules": [
                {
                    "alert": "HighErrorRate",
                    "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) > 0.1",
                    "for": "5m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "High error rate detected",
                        "description": "Error rate is above 10% for 5 minutes"
                    }
                },
                {
                    "alert": "HighResponseTime",
                    "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "High response time detected",
                        "description": "95th percentile response time is above 2 seconds"
                    }
                },
                {
                    "alert": "ServiceDown",
                    "expr": "up == 0",
                    "for": "1m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "Service is down",
                        "description": "{{$labels.instance}} has been down for more than 1 minute"
                    }
                },
                {
                    "alert": "HighMemoryUsage",
                    "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "High memory usage",
                        "description": "Memory usage is above 90% on {{$labels.instance}}"
                    }
                },
                {
                    "alert": "HighCPUUsage",
                    "expr": "100 - (avg by (instance) (irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100) > 80",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "High CPU usage",
                        "description": "CPU usage is above 80% on {{$labels.instance}}"
                    }
                },
                {
                    "alert": "DatabaseConnectionsHigh",
                    "expr": "pg_stat_database_numbackends > 80",
                    "for": "5m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "High database connections",
                        "description": "Database has more than 80 active connections"
                    }
                },
                {
                    "alert": "ClaimsProcessingBacklog",
                    "expr": "claims_processing_queue_size > 100",
                    "for": "10m",
                    "labels": {
                        "severity": "warning"
                    },
                    "annotations": {
                        "summary": "Claims processing backlog",
                        "description": "Claims processing queue has more than 100 items"
                    }
                },
                {
                    "alert": "AIAgentFailure",
                    "expr": "ai_agent_health_status == 0",
                    "for": "2m",
                    "labels": {
                        "severity": "critical"
                    },
                    "annotations": {
                        "summary": "AI Agent failure",
                        "description": "AI Agent {{$labels.agent_name}} is not responding"
                    }
                }
            ]
        }
    ]
}

def export_dashboards():
    """Export all dashboards as JSON files"""
    dashboards = {
        "system_overview": SYSTEM_OVERVIEW_DASHBOARD,
        "ai_agents": AI_AGENTS_DASHBOARD,
        "business_metrics": BUSINESS_METRICS_DASHBOARD,
        "infrastructure": INFRASTRUCTURE_DASHBOARD
    }
    
    return dashboards

def export_config():
    """Export Grafana configuration"""
    return GRAFANA_CONFIG

def export_alerts():
    """Export alert rules"""
    return ALERT_RULES

if __name__ == "__main__":
    # Export all configurations
    import os
    
    # Create output directory
    os.makedirs("grafana_export", exist_ok=True)
    
    # Export dashboards
    dashboards = export_dashboards()
    for name, dashboard in dashboards.items():
        with open(f"grafana_export/{name}_dashboard.json", "w") as f:
            json.dump(dashboard, f, indent=2)
    
    # Export configuration
    with open("grafana_export/datasources.json", "w") as f:
        json.dump(export_config(), f, indent=2)
    
    # Export alerts
    with open("grafana_export/alert_rules.yml", "w") as f:
        import yaml
        yaml.dump(export_alerts(), f, default_flow_style=False)
    
    print("Grafana configuration exported successfully!")

