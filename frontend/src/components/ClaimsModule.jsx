import React, { useState, useEffect, useCallback } from 'react';
import { Upload, FileText, Camera, Video, AlertTriangle, CheckCircle, Clock, X, Download, Eye, MapPin, Phone, Mail, DollarSign, Calendar } from 'lucide-react';

const ClaimsModule = () => {
  const [claims, setClaims] = useState([]);
  const [selectedClaim, setSelectedClaim] = useState(null);
  const [evidenceFiles, setEvidenceFiles] = useState([]);
  const [isUploading, setIsUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [filters, setFilters] = useState({
    status: 'all',
    claimType: 'all',
    priority: 'all',
    dateRange: '30'
  });
  const [newClaim, setNewClaim] = useState({
    policyNumber: '',
    claimType: 'auto',
    incidentDate: '',
    incidentLocation: '',
    description: '',
    estimatedAmount: '',
    contactPhone: '',
    contactEmail: '',
    priority: 'medium',
    evidence: []
  });
  const [showNewClaimForm, setShowNewClaimForm] = useState(false);
  const [processingEvidence, setProcessingEvidence] = useState(false);

  // Fetch claims on component mount
  useEffect(() => {
    fetchClaims();
  }, [filters]);

  const fetchClaims = async () => {
    try {
      const queryParams = new URLSearchParams({
        status: filters.status !== 'all' ? filters.status : '',
        claimType: filters.claimType !== 'all' ? filters.claimType : '',
        priority: filters.priority !== 'all' ? filters.priority : '',
        dateRange: filters.dateRange
      });

      const response = await fetch(`/api/v1/claims?${queryParams}`, {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        const data = await response.json();
        setClaims(data.claims || []);
      }
    } catch (error) {
      console.error('Error fetching claims:', error);
    }
  };

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(async (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    const files = Array.from(e.dataTransfer.files);
    await handleEvidenceUpload(files);
  }, []);

  const handleEvidenceUpload = async (files) => {
    setIsUploading(true);
    setProcessingEvidence(true);
    const allowedTypes = [
      'application/pdf', 'image/jpeg', 'image/png', 'image/gif',
      'video/mp4', 'video/avi', 'video/mov', 'video/wmv',
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      'audio/mp3', 'audio/wav', 'audio/m4a'
    ];
    const maxSize = 50 * 1024 * 1024; // 50MB for evidence files

    for (const file of files) {
      if (!allowedTypes.includes(file.type)) {
        alert(`File type ${file.type} is not allowed. Please upload supported evidence files.`);
        continue;
      }

      if (file.size > maxSize) {
        alert(`File ${file.name} is too large. Maximum size is 50MB.`);
        continue;
      }

      const formData = new FormData();
      formData.append('file', file);
      formData.append('documentType', 'evidence');
      formData.append('claimId', selectedClaim?.id || 'new');
      formData.append('evidenceType', getEvidenceType(file.type));

      try {
        const response = await fetch('/api/v1/claims/evidence/upload', {
          method: 'POST',
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          },
          body: formData
        });

        if (response.ok) {
          const result = await response.json();
          
          // Process evidence with AI analysis
          const analysisResponse = await fetch('/api/v1/claims/evidence/analyze', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${localStorage.getItem('token')}`
            },
            body: JSON.stringify({
              evidenceId: result.id,
              fileType: file.type,
              fileName: file.name
            })
          });

          let analysisData = {};
          if (analysisResponse.ok) {
            analysisData = await analysisResponse.json();
          }

          setEvidenceFiles(prev => [...prev, {
            id: result.id,
            name: file.name,
            size: file.size,
            type: file.type,
            evidenceType: getEvidenceType(file.type),
            uploadedAt: new Date().toISOString(),
            status: 'processed',
            url: result.url,
            thumbnail: result.thumbnail,
            analysis: analysisData,
            metadata: {
              damageAssessment: analysisData.damageAssessment,
              objectsDetected: analysisData.objectsDetected,
              textExtracted: analysisData.textExtracted,
              fraudIndicators: analysisData.fraudIndicators,
              estimatedCost: analysisData.estimatedCost
            }
          }]);
        } else {
          alert(`Failed to upload ${file.name}`);
        }
      } catch (error) {
        console.error('Upload error:', error);
        alert(`Error uploading ${file.name}`);
      }
    }
    setIsUploading(false);
    setProcessingEvidence(false);
  };

  const getEvidenceType = (mimeType) => {
    if (mimeType.startsWith('image/')) return 'photo';
    if (mimeType.startsWith('video/')) return 'video';
    if (mimeType.startsWith('audio/')) return 'audio';
    if (mimeType === 'application/pdf') return 'document';
    return 'other';
  };

  const handleFileInputChange = (e) => {
    const files = Array.from(e.target.files);
    handleEvidenceUpload(files);
  };

  const removeEvidence = (evidenceId) => {
    setEvidenceFiles(prev => prev.filter(file => file.id !== evidenceId));
  };

  const submitClaim = async () => {
    try {
      const claimData = {
        ...newClaim,
        evidence: evidenceFiles.map(file => file.id),
        submittedAt: new Date().toISOString(),
        status: 'submitted',
        claimNumber: `CLM-${Date.now()}`
      };

      const response = await fetch('/api/v1/claims', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify(claimData)
      });

      if (response.ok) {
        const result = await response.json();
        setClaims(prev => [result, ...prev]);
        setShowNewClaimForm(false);
        setNewClaim({
          policyNumber: '',
          claimType: 'auto',
          incidentDate: '',
          incidentLocation: '',
          description: '',
          estimatedAmount: '',
          contactPhone: '',
          contactEmail: '',
          priority: 'medium',
          evidence: []
        });
        setEvidenceFiles([]);
        alert('Claim submitted successfully!');
      } else {
        alert('Failed to submit claim');
      }
    } catch (error) {
      console.error('Submission error:', error);
      alert('Error submitting claim');
    }
  };

  const approveClaim = async (claimId, amount) => {
    try {
      const response = await fetch(`/api/v1/claims/${claimId}/approve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ approvedAmount: amount })
      });

      if (response.ok) {
        fetchClaims();
        alert('Claim approved successfully!');
      }
    } catch (error) {
      console.error('Approval error:', error);
    }
  };

  const rejectClaim = async (claimId, reason) => {
    try {
      const response = await fetch(`/api/v1/claims/${claimId}/reject`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ reason })
      });

      if (response.ok) {
        fetchClaims();
        alert('Claim rejected');
      }
    } catch (error) {
      console.error('Rejection error:', error);
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'approved':
        return <CheckCircle className="w-5 h-5 text-green-500" />;
      case 'rejected':
        return <X className="w-5 h-5 text-red-500" />;
      case 'investigating':
        return <AlertTriangle className="w-5 h-5 text-orange-500" />;
      case 'submitted':
        return <Clock className="w-5 h-5 text-blue-500" />;
      default:
        return <Clock className="w-5 h-5 text-gray-500" />;
    }
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'high':
        return 'text-red-600 bg-red-100';
      case 'medium':
        return 'text-yellow-600 bg-yellow-100';
      case 'low':
        return 'text-green-600 bg-green-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getEvidenceIcon = (evidenceType) => {
    switch (evidenceType) {
      case 'photo':
        return <Camera className="w-5 h-5" />;
      case 'video':
        return <Video className="w-5 h-5" />;
      case 'document':
        return <FileText className="w-5 h-5" />;
      default:
        return <FileText className="w-5 h-5" />;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Claims Management</h1>
        <p className="text-gray-600">Process insurance claims, analyze evidence, and manage settlements</p>
      </div>

      {/* Action Bar */}
      <div className="mb-6 flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div className="flex flex-wrap gap-4">
          <select
            value={filters.status}
            onChange={(e) => setFilters(prev => ({ ...prev, status: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="submitted">Submitted</option>
            <option value="investigating">Investigating</option>
            <option value="approved">Approved</option>
            <option value="rejected">Rejected</option>
          </select>

          <select
            value={filters.claimType}
            onChange={(e) => setFilters(prev => ({ ...prev, claimType: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Types</option>
            <option value="auto">Auto</option>
            <option value="home">Home</option>
            <option value="health">Health</option>
            <option value="life">Life</option>
            <option value="business">Business</option>
          </select>

          <select
            value={filters.priority}
            onChange={(e) => setFilters(prev => ({ ...prev, priority: e.target.value }))}
            className="px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Priorities</option>
            <option value="high">High Priority</option>
            <option value="medium">Medium Priority</option>
            <option value="low">Low Priority</option>
          </select>
        </div>

        <button
          onClick={() => setShowNewClaimForm(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 flex items-center gap-2"
        >
          <FileText className="w-4 h-4" />
          New Claim
        </button>
      </div>

      {/* Claims Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-8">
        {claims.map((claim) => (
          <div
            key={claim.id}
            className="bg-white rounded-lg shadow-md border border-gray-200 p-6 hover:shadow-lg transition-shadow cursor-pointer"
            onClick={() => setSelectedClaim(claim)}
          >
            <div className="flex justify-between items-start mb-4">
              <div>
                <h3 className="text-lg font-semibold text-gray-900">{claim.claimNumber}</h3>
                <p className="text-sm text-gray-600">{claim.claimType} Claim</p>
              </div>
              <div className="flex items-center gap-2">
                {getStatusIcon(claim.status)}
                <span className={`px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(claim.priority)}`}>
                  {claim.priority} Priority
                </span>
              </div>
            </div>

            <div className="space-y-2 mb-4">
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <DollarSign className="w-4 h-4" />
                Estimated: ${claim.estimatedAmount?.toLocaleString()}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Calendar className="w-4 h-4" />
                Incident: {new Date(claim.incidentDate).toLocaleDateString()}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <MapPin className="w-4 h-4" />
                Location: {claim.incidentLocation}
              </div>
              <div className="flex items-center gap-2 text-sm text-gray-600">
                <Camera className="w-4 h-4" />
                Evidence: {claim.evidence?.length || 0} files
              </div>
            </div>

            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-900">
                Policy: {claim.policyNumber}
              </span>
              {claim.status === 'submitted' && (
                <div className="flex gap-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const amount = prompt('Approved amount:');
                      if (amount) approveClaim(claim.id, parseFloat(amount));
                    }}
                    className="px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700"
                  >
                    Approve
                  </button>
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      const reason = prompt('Rejection reason:');
                      if (reason) rejectClaim(claim.id, reason);
                    }}
                    className="px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700"
                  >
                    Reject
                  </button>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* New Claim Modal */}
      {showNewClaimForm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-4xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">New Insurance Claim</h2>
                <button
                  onClick={() => setShowNewClaimForm(false)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Policy Number</label>
                  <input
                    type="text"
                    value={newClaim.policyNumber}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, policyNumber: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter policy number"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Claim Type</label>
                  <select
                    value={newClaim.claimType}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, claimType: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="auto">Auto Insurance</option>
                    <option value="home">Home Insurance</option>
                    <option value="health">Health Insurance</option>
                    <option value="life">Life Insurance</option>
                    <option value="business">Business Insurance</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Incident Date</label>
                  <input
                    type="date"
                    value={newClaim.incidentDate}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, incidentDate: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Incident Location</label>
                  <input
                    type="text"
                    value={newClaim.incidentLocation}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, incidentLocation: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter incident location"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Estimated Amount</label>
                  <input
                    type="number"
                    value={newClaim.estimatedAmount}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, estimatedAmount: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter estimated amount"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Priority</label>
                  <select
                    value={newClaim.priority}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, priority: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="low">Low Priority</option>
                    <option value="medium">Medium Priority</option>
                    <option value="high">High Priority</option>
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Contact Phone</label>
                  <input
                    type="tel"
                    value={newClaim.contactPhone}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, contactPhone: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter contact phone"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Contact Email</label>
                  <input
                    type="email"
                    value={newClaim.contactEmail}
                    onChange={(e) => setNewClaim(prev => ({ ...prev, contactEmail: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter contact email"
                  />
                </div>
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Description</label>
                <textarea
                  value={newClaim.description}
                  onChange={(e) => setNewClaim(prev => ({ ...prev, description: e.target.value }))}
                  rows={4}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Describe the incident in detail..."
                />
              </div>

              {/* Evidence Upload Area */}
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Evidence Files</label>
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                    dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-lg font-medium text-gray-900 mb-2">
                    Drop evidence files here or click to upload
                  </p>
                  <p className="text-sm text-gray-600 mb-4">
                    Supports photos, videos, documents, and audio files up to 50MB each
                  </p>
                  <input
                    type="file"
                    multiple
                    accept=".pdf,.jpg,.jpeg,.png,.gif,.mp4,.avi,.mov,.wmv,.docx,.mp3,.wav,.m4a"
                    onChange={handleFileInputChange}
                    className="hidden"
                    id="evidence-upload"
                  />
                  <label
                    htmlFor="evidence-upload"
                    className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 cursor-pointer inline-block"
                  >
                    Choose Evidence Files
                  </label>
                </div>

                {/* Evidence Files List */}
                {evidenceFiles.length > 0 && (
                  <div className="mt-4">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Evidence Files</h4>
                    <div className="space-y-3">
                      {evidenceFiles.map((file) => (
                        <div key={file.id} className="flex items-start justify-between p-4 bg-gray-50 rounded-md">
                          <div className="flex items-start gap-3">
                            {getEvidenceIcon(file.evidenceType)}
                            <div className="flex-1">
                              <p className="text-sm font-medium text-gray-900">{file.name}</p>
                              <p className="text-xs text-gray-600">
                                {(file.size / 1024 / 1024).toFixed(2)} MB • {file.evidenceType} • 
                                Uploaded {new Date(file.uploadedAt).toLocaleTimeString()}
                              </p>
                              {file.analysis && (
                                <div className="mt-2 text-xs text-gray-600">
                                  {file.analysis.damageAssessment && (
                                    <p>Damage: {file.analysis.damageAssessment}</p>
                                  )}
                                  {file.analysis.estimatedCost && (
                                    <p>Est. Cost: ${file.analysis.estimatedCost.toLocaleString()}</p>
                                  )}
                                  {file.analysis.fraudIndicators && file.analysis.fraudIndicators.length > 0 && (
                                    <p className="text-red-600">⚠️ Fraud indicators detected</p>
                                  )}
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2">
                            <button
                              onClick={() => window.open(file.url, '_blank')}
                              className="p-1 text-blue-600 hover:text-blue-800"
                            >
                              <Eye className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => removeEvidence(file.id)}
                              className="p-1 text-red-600 hover:text-red-800"
                            >
                              <X className="w-4 h-4" />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {processingEvidence && (
                  <div className="mt-4 p-4 bg-blue-50 rounded-md">
                    <p className="text-sm text-blue-800">🔍 Processing evidence with AI analysis...</p>
                  </div>
                )}
              </div>

              {/* Action Buttons */}
              <div className="flex justify-end gap-4">
                <button
                  onClick={() => setShowNewClaimForm(false)}
                  className="px-4 py-2 text-gray-700 border border-gray-300 rounded-md hover:bg-gray-50"
                >
                  Cancel
                </button>
                <button
                  onClick={submitClaim}
                  disabled={!newClaim.policyNumber || !newClaim.incidentDate || isUploading}
                  className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isUploading ? 'Processing...' : 'Submit Claim'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Claim Details Modal */}
      {selectedClaim && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg max-w-6xl w-full max-h-[90vh] overflow-y-auto">
            <div className="p-6">
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-2xl font-bold text-gray-900">Claim Details - {selectedClaim.claimNumber}</h2>
                <button
                  onClick={() => setSelectedClaim(null)}
                  className="text-gray-400 hover:text-gray-600"
                >
                  <X className="w-6 h-6" />
                </button>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <div className="space-y-6">
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Claim Information</h3>
                    <div className="space-y-2">
                      <p><span className="font-medium">Policy Number:</span> {selectedClaim.policyNumber}</p>
                      <p><span className="font-medium">Claim Type:</span> {selectedClaim.claimType}</p>
                      <p><span className="font-medium">Status:</span> 
                        <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${
                          selectedClaim.status === 'approved' ? 'bg-green-100 text-green-800' :
                          selectedClaim.status === 'rejected' ? 'bg-red-100 text-red-800' :
                          selectedClaim.status === 'investigating' ? 'bg-orange-100 text-orange-800' :
                          'bg-blue-100 text-blue-800'
                        }`}>
                          {selectedClaim.status}
                        </span>
                      </p>
                      <p><span className="font-medium">Priority:</span> 
                        <span className={`ml-2 px-2 py-1 rounded-full text-xs font-medium ${getPriorityColor(selectedClaim.priority)}`}>
                          {selectedClaim.priority}
                        </span>
                      </p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Incident Details</h3>
                    <div className="space-y-2">
                      <p><span className="font-medium">Date:</span> {new Date(selectedClaim.incidentDate).toLocaleDateString()}</p>
                      <p><span className="font-medium">Location:</span> {selectedClaim.incidentLocation}</p>
                      <p><span className="font-medium">Estimated Amount:</span> ${selectedClaim.estimatedAmount?.toLocaleString()}</p>
                      <p><span className="font-medium">Description:</span></p>
                      <p className="text-gray-700 bg-gray-50 p-3 rounded-md">{selectedClaim.description}</p>
                    </div>
                  </div>

                  <div>
                    <h3 className="text-lg font-semibold text-gray-900 mb-3">Contact Information</h3>
                    <div className="space-y-2">
                      <div className="flex items-center gap-2">
                        <Phone className="w-4 h-4 text-gray-500" />
                        <span>{selectedClaim.contactPhone}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <Mail className="w-4 h-4 text-gray-500" />
                        <span>{selectedClaim.contactEmail}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3">Evidence Files</h3>
                  <div className="space-y-3 max-h-96 overflow-y-auto">
                    {selectedClaim.evidence?.map((evidence, index) => (
                      <div key={index} className="flex items-start justify-between p-3 bg-gray-50 rounded-md">
                        <div className="flex items-start gap-3">
                          {getEvidenceIcon(evidence.evidenceType)}
                          <div>
                            <p className="text-sm font-medium">{evidence.name}</p>
                            <p className="text-xs text-gray-600">{evidence.evidenceType} • {(evidence.size / 1024 / 1024).toFixed(2)} MB</p>
                            {evidence.analysis && (
                              <div className="mt-1 text-xs text-gray-600">
                                {evidence.analysis.damageAssessment && (
                                  <p>Damage: {evidence.analysis.damageAssessment}</p>
                                )}
                                {evidence.analysis.estimatedCost && (
                                  <p>Est. Cost: ${evidence.analysis.estimatedCost.toLocaleString()}</p>
                                )}
                              </div>
                            )}
                          </div>
                        </div>
                        <div className="flex gap-2">
                          <button
                            onClick={() => window.open(evidence.url, '_blank')}
                            className="p-1 text-blue-600 hover:text-blue-800"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => {
                              const link = document.createElement('a');
                              link.href = evidence.url;
                              link.download = evidence.name;
                              link.click();
                            }}
                            className="p-1 text-green-600 hover:text-green-800"
                          >
                            <Download className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {selectedClaim.status === 'submitted' && (
                <div className="mt-6 flex justify-end gap-4">
                  <button
                    onClick={() => {
                      const reason = prompt('Rejection reason:');
                      if (reason) {
                        rejectClaim(selectedClaim.id, reason);
                        setSelectedClaim(null);
                      }
                    }}
                    className="px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700"
                  >
                    Reject Claim
                  </button>
                  <button
                    onClick={() => {
                      const amount = prompt('Approved amount:', selectedClaim.estimatedAmount);
                      if (amount) {
                        approveClaim(selectedClaim.id, parseFloat(amount));
                        setSelectedClaim(null);
                      }
                    }}
                    className="px-4 py-2 bg-green-600 text-white rounded-md hover:bg-green-700"
                  >
                    Approve Claim
                  </button>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ClaimsModule;

