import React, { useState, useEffect } from 'react';
import { Bot, Activity, Settings, Play, Pause, RotateCcw, AlertCircle, CheckCircle, Clock, TrendingUp, TrendingDown, Users, Zap, Brain, MessageSquare, FileSearch, Shield } from 'lucide-react';

const AgentManagementInterface = () => {
  const [agents, setAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [agentMetrics, setAgentMetrics] = useState({});
  const [systemHealth, setSystemHealth] = useState({});
  const [filters, setFilters] = useState({
    status: 'all',
    type: 'all',
    performance: 'all'
  });

  // Fetch agents and metrics on component mount
  useEffect(() => {
    fetchAgents();
    fetchSystemHealth();
    const interval = setInterval(() => {
      fetchAgents();
      fetchSystemHealth();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const fetchAgents = async () => {
    try {
      const response = await fetch('/api/v1/agents', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setAgents(data.agents || []);
        
        // Fetch metrics for each agent
        const metricsPromises = data.agents.map(agent => 
          fetch(`/api/v1/agents/${agent.id}/metrics`, {
            headers: {
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            }
          }).then(res => res.json())
        );

        const metricsResults = await Promise.all(metricsPromises);
        const metricsMap = {};
        data.agents.forEach((agent, index) => {
          metricsMap[agent.id] = metricsResults[index];
        });
        setAgentMetrics(metricsMap);
      }
    } catch (error) {
      console.error('Error fetching agents:', error);
    }
  };

  const fetchSystemHealth = async () => {
    try {
      const response = await fetch('/api/v1/system/health', {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setSystemHealth(data);
      }
    } catch (error) {
      console.error('Error fetching system health:', error);
    }
  };

  const startAgent = async (agentId) => {
    try {
      const response = await fetch(`/api/v1/agents/${agentId}/start`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        fetchAgents();
        alert('Agent started successfully');
      }
    } catch (error) {
      console.error('Error starting agent:', error);
    }
  };

  const stopAgent = async (agentId) => {
    try {
      const response = await fetch(`/api/v1/agents/${agentId}/stop`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        fetchAgents();
        alert('Agent stopped successfully');
      }
    } catch (error) {
      console.error('Error stopping agent:', error);
    }
  };

  const restartAgent = async (agentId) => {
    try {
      const response = await fetch(`/api/v1/agents/${agentId}/restart`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        fetchAgents();
        alert('Agent restarted successfully');
      }
    } catch (error) {
      console.error('Error restarting agent:', error);
    }
  };

  const updateAgentConfig = async (agentId, config) => {
    try {
      const response = await fetch(`/api/v1/agents/${agentId}/config`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(config)
      });

      if (response.ok) {
        fetchAgents();
        alert('Agent configuration updated successfully');
      }
    } catch (error) {
      console.error('Error updating agent config:', error);
    }
  };

  const getAgentIcon = (agentType) => {
    switch (agentType) {
      case 'document_analysis':
        return <FileSearch className="w-6 h-6" />;
      case 'communication':
        return <MessageSquare className="w-6 h-6" />;
      case 'risk_assessment':
        return <Brain className="w-6 h-6" />;
      case 'compliance':
        return <Shield className="w-6 h-6" />;
      case 'evidence_processing':
        return <Activity className="w-6 h-6" />;
      default:
        return <Bot className="w-6 h-6" />;
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'running':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'stopped':
        return <AlertCircle className="w-5 h-5 text-red-500" />;
      case 'starting':
      case 'stopping':
        return <Clock className="w-5 h-5 text-yellow-500" />;
      default:
        return <AlertCircle className="w-5 h-5 text-gray-500" />;
    }
  };

  const getPerformanceColor = (performance) => {
    if (performance >= 90) return 'text-green-600 bg-green-100';
    if (performance >= 70) return 'text-yellow-600 bg-yellow-100';
    return 'text-red-600 bg-red-100';
  };

  const filteredAgents = agents.filter(agent => {
    if (filters.status !== 'all' && agent.status !== filters.status) return false;
    if (filters.type !== 'all' && agent.type !== filters.type) return false;
    if (filters.performance !== 'all') {
      const metrics = agentMetrics[agent.id];
      if (!metrics) return false;
      const performance = metrics.performance_score || 0;
      if (filters.performance === 'high' && performance < 90) return false;
      if (filters.performance === 'medium' && (performance < 70 || performance >= 90)) return false;
      if (filters.performance === 'low' && performance >= 70) return false;
    }
    return true;
  });

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Agent Management</h1>
        <p className="text-gray-600">Monitor and manage AI agents, view performance metrics, and configure agent settings</p>
      </div>

      {/* System Health Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Total Agents</p>
              <p className="text-2xl font-bold text-gray-900">{agents.length}</p>
            </div>
            <Users className="w-8 h-8 text-blue-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Running Agents</p>
              <p className="text-2xl font-bold text-green-600">
                {agents.filter(a => a.status === 'running').length}
              </p>
            </div>
            <CheckCircle className="w-8 h-8 text-green-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">System Load</p>
              <p className="text-2xl font-bold text-yellow-600">
                {systemHealth.cpu_usage || 0}%
              </p>
            </div>
            <Activity className="w-8 h-8 text-yellow-500" />
          </div>
        </div>

        <div className="bg-white rounded-lg shadow-md border border-gray-200 p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-600">Avg Performance</p>
              <p className="text-2xl font-bold text-blue-600">
                {Object.values(agentMetrics).length > 0 
                  ? Math.round(Object.values(agentMetrics).reduce((sum, m) => sum + (m.performance_score || 0), 0) / Object.values(agentMetrics).length)
                  : 0}%
              </p>
            </div>
            <TrendingUp className="w-8 h-8 text-blue-500" />
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="mb-6 flex flex-wrap gap-4">
        <select
          value={filters.status}
          onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Status</option>
          <option value="running">Running</option>
          <option value="stopped">Stopped</option>
          <option value="starting">Starting</option>
          <option value="stopping">Stopping</option>
        </select>

        <select
          value={filters.type}
          onChange={(e) => setFilters(prev => ({ ...prev, type: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Types</option>
          <option value="document_analysis">Document Analysis</option>
          <option value="communication">Communication</option>
          <option value="risk_assessment">Risk Assessment</option>
          <option value="compliance">Compliance</option>
          <option value="evidence_processing">Evidence Processing</option>
        </select>

        <select
          value={filters.performance}
          onChange={(e) => setFilters(prev => ({ ...prev, performance: e.target.value }))}
          className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="all">All Performance</option>
          <option value="high">High (90%+)</option>
          <option value="medium">Medium (70-89%)</option>
          <option value="low">Low (&lt;70%)</option>
        </select>
      </div>

      {/* Agents Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        {filteredAgents.map((agent) => {
          const metrics = agentMetrics[agent.id] || {};
          return (
            <div
              key={agent.id}
              className="bg-white rounded-lg shadow-md border border-gray-200 p-6 hover:shadow-lg transition-shadow cursor-pointer"
              onClick={() => setSelectedAgent(agent)}
            >
              <div className="flex items-start justify-between mb-4">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    {getAgentIcon(agent.type)}
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">{agent.name}</h3>
                    <p className="text-sm text-gray-600">{agent.type.replace('_', ' ').toUpperCase()}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {getStatusIcon(agent.status)}
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                    agent.status === 'running' ? 'bg-green-100 text-green-800' :
                    agent.status === 'stopped' ? 'bg-red-100 text-red-800' :
                    'bg-yellow-100 text-yellow-800'
                  }`}>
                    {agent.status}
                  </span>
                </div>
              </div>

              <div className="space-y-3 mb-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Performance</span>
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPerformanceColor(metrics.performance_score || 0)}`}>
                    {metrics.performance_score || 0}%
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Tasks Processed</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics.tasks_processed || 0}
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Success Rate</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics.success_rate || 0}%
                  </span>
                </div>

                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">Avg Response Time</span>
                  <span className="text-sm font-medium text-gray-900">
                    {metrics.avg_response_time || 0}ms
                  </span>
                </div>
              </div>

              <div className="flex justify-between items-center">
                <div className="flex gap-2">
                  {agent.status === 'stopped' ? (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        startAgent(agent.id);
                      }}
                      className="p-2 bg-green-600 text-white rounded hover:bg-green-700"
                      title="Start Agent"
                    >
                      <Play className="w-4 h-4" />
                    </button>
                  ) : (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        stopAgent(agent.id);
                      }}
                      className="p-2 bg-red-600 text-white rounded hover:bg-red-700"
                      title="Stop Agent"
                    >
                      <Pause className="w-4 h-4" />
                    </button>
                  )}
                  
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      restartAgent(agent.id);
                    }}
                    className="p-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                    title="Restart Agent"
                  >
                    <RotateCcw className="w-4 h-4" />
                  </button>
                </div>

                <span className="text-xs text-gray-500">
                  Last seen: {agent.last_heartbeat ? new Date(agent.last_heartbeat).toLocaleTimeString() : 'Never'}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Agent Details Modal */}
      {selectedAgent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    {getAgentIcon(selectedAgent.type)}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-gray-900">{selectedAgent.name}</h2>
                    <p className="text-gray-600">{selectedAgent.type.replace('_', ' ').toUpperCase()} Agent</p>
                  </div>
                </div>
                <button
                  onClick={() => setSelectedAgent(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <AlertCircle className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Agent Status</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Status</span>
                        <div className="flex items-center gap-2">
                          {getStatusIcon(selectedAgent.status)}
                          <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                            selectedAgent.status === 'running' ? 'bg-green-100 text-green-800' :
                            selectedAgent.status === 'stopped' ? 'bg-red-100 text-red-800' :
                            'bg-yellow-100 text-yellow-800'
                          }`}>
                            {selectedAgent.status}
                          </span>
                        </div>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Uptime</span>
                        <span className="text-sm font-medium text-gray-900">
                          {selectedAgent.uptime || '0h 0m'}
                        </span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Version</span>
                        <span className="text-sm font-medium text-gray-900">
                          {selectedAgent.version || '1.0.0'}
                        </span>
                      </div>

                      <div className="flex justify-between items-center">
                        <span className="text-sm text-gray-600">Last Heartbeat</span>
                        <span className="text-sm font-medium text-gray-900">
                          {selectedAgent.last_heartbeat ? new Date(selectedAgent.last_heartbeat).toLocaleString() : 'Never'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Performance Metrics</h3>
                    <div className="space-y-3">
                      {agentMetrics[selectedAgent.id] && Object.entries(agentMetrics[selectedAgent.id]).map(([key, value]) => (
                        <div key={key} className="flex justify-between items-center">
                          <span className="text-sm text-gray-600">{key.replace('_', ' ').toUpperCase()}</span>
                          <span className="text-sm font-medium text-gray-900">
                            {typeof value === 'number' ? 
                              (key.includes('rate') || key.includes('score') ? `${value}%` : 
                               key.includes('time') ? `${value}ms` : value) : 
                              value}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Configuration</h3>
                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Max Concurrent Tasks</label>
                        <input
                          type="number"
                          defaultValue={selectedAgent.config?.max_concurrent_tasks || 5}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Timeout (seconds)</label>
                        <input
                          type="number"
                          defaultValue={selectedAgent.config?.timeout || 300}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Retry Attempts</label>
                        <input
                          type="number"
                          defaultValue={selectedAgent.config?.retry_attempts || 3}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        />
                      </div>

                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Log Level</label>
                        <select
                          defaultValue={selectedAgent.config?.log_level || 'INFO'}
                          className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                        >
                          <option value="DEBUG">DEBUG</option>
                          <option value="INFO">INFO</option>
                          <option value="WARNING">WARNING</option>
                          <option value="ERROR">ERROR</option>
                        </select>
                      </div>

                      <div className="flex items-center gap-2">
                        <input
                          type="checkbox"
                          id="auto-restart"
                          defaultChecked={selectedAgent.config?.auto_restart || false}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <label htmlFor="auto-restart" className="text-sm text-gray-700">Auto Restart on Failure</label>
                      </div>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Recent Activity</h3>
                    <div className="space-y-2 max-h-48 overflow-y-auto">
                      {selectedAgent.recent_logs?.map((log, index) => (
                        <div key={index} className="text-xs p-2 bg-gray-50 rounded">
                          <span className="text-gray-500">{new Date(log.timestamp).toLocaleTimeString()}</span>
                          <span className={`ml-2 px-1 rounded text-xs ${
                            log.level === 'ERROR' ? 'bg-red-100 text-red-800' :
                            log.level === 'WARNING' ? 'bg-yellow-100 text-yellow-800' :
                            'bg-blue-100 text-blue-800'
                          }`}>
                            {log.level}
                          </span>
                          <p className="mt-1 text-gray-700">{log.message}</p>
                        </div>
                      )) || (
                        <p className="text-sm text-gray-500">No recent activity</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="mt-6 flex justify-between items-center">
                <div className="flex gap-4">
                  {selectedAgent.status === 'stopped' ? (
                    <button
                      onClick={() => startAgent(selectedAgent.id)}
                      className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700 flex items-center gap-2"
                    >
                      <Play className="w-4 h-4" />
                      Start Agent
                    </button>
                  ) : (
                    <button
                      onClick={() => stopAgent(selectedAgent.id)}
                      className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 flex items-center gap-2"
                    >
                      <Pause className="w-4 h-4" />
                      Stop Agent
                    </button>
                  )}
                  
                  <button
                    onClick={() => restartAgent(selectedAgent.id)}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Restart Agent
                  </button>
                </div>

                <div className="flex gap-4">
                  <button
                    onClick={() => setSelectedAgent(null)}
                    className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
                  >
                    Close
                  </button>
                  <button
                    onClick={() => {
                      // Collect configuration from form inputs
                      const config = {
                        max_concurrent_tasks: parseInt(document.querySelector('input[type="number"]').value),
                        timeout: parseInt(document.querySelectorAll('input[type="number"]')[1].value),
                        retry_attempts: parseInt(document.querySelectorAll('input[type="number"]')[2].value),
                        log_level: document.querySelector('select').value,
                        auto_restart: document.querySelector('input[type="checkbox"]').checked
                      };
                      updateAgentConfig(selectedAgent.id, config);
                    }}
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 flex items-center gap-2"
                  >
                    <Settings className="w-4 h-4" />
                    Save Configuration
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AgentManagementInterface;

