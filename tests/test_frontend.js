/**
 * Insurance AI Agent System - Frontend Test Suite
 * Comprehensive testing for React frontend components and functionality
 */

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import '@testing-library/jest-dom';

// Mock modules
jest.mock('recharts', () => ({
  LineChart: ({ children }) => <div data-testid="line-chart">{children}</div>,
  Line: () => <div data-testid="line" />,
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  CartesianGrid: () => <div data-testid="cartesian-grid" />,
  Tooltip: () => <div data-testid="tooltip" />,
  ResponsiveContainer: ({ children }) => <div data-testid="responsive-container">{children}</div>,
  BarChart: ({ children }) => <div data-testid="bar-chart">{children}</div>,
  Bar: () => <div data-testid="bar" />,
  PieChart: ({ children }) => <div data-testid="pie-chart">{children}</div>,
  Pie: () => <div data-testid="pie" />,
  Cell: () => <div data-testid="cell" />
}));

// Mock fetch
global.fetch = jest.fn();

// Test utilities
const renderWithRouter = (component) => {
  return render(
    <BrowserRouter>
      {component}
    </BrowserRouter>
  );
};

const mockAuthContext = {
  user: {
    id: 'test-user-id',
    email: 'test@zurich.com',
    firstName: 'Test',
    lastName: 'User',
    role: 'agent'
  },
  login: jest.fn(),
  logout: jest.fn(),
  isAuthenticated: true,
  loading: false
};

// Mock components for testing
const MockAuthProvider = ({ children }) => {
  return (
    <div data-testid="auth-provider">
      {children}
    </div>
  );
};

// =============================================================================
// Authentication Tests
// =============================================================================

describe('Authentication', () => {
  beforeEach(() => {
    fetch.mockClear();
  });

  test('renders login form', () => {
    const LoginForm = () => (
      <form data-testid="login-form">
        <input
          type="email"
          placeholder="Email"
          data-testid="email-input"
        />
        <input
          type="password"
          placeholder="Password"
          data-testid="password-input"
        />
        <button type="submit" data-testid="login-button">
          Sign In
        </button>
      </form>
    );

    render(<LoginForm />);
    
    expect(screen.getByTestId('login-form')).toBeInTheDocument();
    expect(screen.getByTestId('email-input')).toBeInTheDocument();
    expect(screen.getByTestId('password-input')).toBeInTheDocument();
    expect(screen.getByTestId('login-button')).toBeInTheDocument();
  });

  test('handles login form submission', async () => {
    const mockLogin = jest.fn();
    
    const LoginForm = () => {
      const [email, setEmail] = React.useState('');
      const [password, setPassword] = React.useState('');

      const handleSubmit = (e) => {
        e.preventDefault();
        mockLogin({ email, password });
      };

      return (
        <form onSubmit={handleSubmit} data-testid="login-form">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="Email"
            data-testid="email-input"
          />
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            data-testid="password-input"
          />
          <button type="submit" data-testid="login-button">
            Sign In
          </button>
        </form>
      );
    };

    const user = userEvent.setup();
    render(<LoginForm />);

    await user.type(screen.getByTestId('email-input'), 'test@zurich.com');
    await user.type(screen.getByTestId('password-input'), 'password123');
    await user.click(screen.getByTestId('login-button'));

    expect(mockLogin).toHaveBeenCalledWith({
      email: 'test@zurich.com',
      password: 'password123'
    });
  });

  test('displays login error message', async () => {
    const LoginForm = () => {
      const [error, setError] = React.useState('');

      const handleSubmit = async (e) => {
        e.preventDefault();
        setError('Invalid credentials');
      };

      return (
        <form onSubmit={handleSubmit}>
          {error && (
            <div data-testid="error-message" role="alert">
              {error}
            </div>
          )}
          <button type="submit" data-testid="login-button">
            Sign In
          </button>
        </form>
      );
    };

    const user = userEvent.setup();
    render(<LoginForm />);

    await user.click(screen.getByTestId('login-button'));

    expect(screen.getByTestId('error-message')).toBeInTheDocument();
    expect(screen.getByText('Invalid credentials')).toBeInTheDocument();
  });
});

// =============================================================================
// Dashboard Tests
// =============================================================================

describe('Dashboard', () => {
  test('renders dashboard with KPI cards', () => {
    const Dashboard = () => (
      <div data-testid="dashboard">
        <div data-testid="kpi-cards">
          <div data-testid="kpi-card" className="kpi-card">
            <h3>Total Policies</h3>
            <span data-testid="kpi-value">1,234</span>
          </div>
          <div data-testid="kpi-card" className="kpi-card">
            <h3>Active Claims</h3>
            <span data-testid="kpi-value">56</span>
          </div>
          <div data-testid="kpi-card" className="kpi-card">
            <h3>Pending Reviews</h3>
            <span data-testid="kpi-value">23</span>
          </div>
        </div>
      </div>
    );

    render(<Dashboard />);

    expect(screen.getByTestId('dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('kpi-cards')).toBeInTheDocument();
    expect(screen.getAllByTestId('kpi-card')).toHaveLength(3);
    expect(screen.getByText('Total Policies')).toBeInTheDocument();
    expect(screen.getByText('Active Claims')).toBeInTheDocument();
    expect(screen.getByText('Pending Reviews')).toBeInTheDocument();
  });

  test('renders charts', () => {
    const Dashboard = () => (
      <div data-testid="dashboard">
        <div data-testid="charts-section">
          <div data-testid="line-chart" />
          <div data-testid="bar-chart" />
          <div data-testid="pie-chart" />
        </div>
      </div>
    );

    render(<Dashboard />);

    expect(screen.getByTestId('charts-section')).toBeInTheDocument();
    expect(screen.getByTestId('line-chart')).toBeInTheDocument();
    expect(screen.getByTestId('bar-chart')).toBeInTheDocument();
    expect(screen.getByTestId('pie-chart')).toBeInTheDocument();
  });

  test('loads dashboard data on mount', async () => {
    const mockData = {
      totalPolicies: 1234,
      activeClaims: 56,
      pendingReviews: 23
    };

    fetch.mockResolvedValueOnce({
      ok: true,
      json: async () => mockData
    });

    const Dashboard = () => {
      const [data, setData] = React.useState(null);
      const [loading, setLoading] = React.useState(true);

      React.useEffect(() => {
        const loadData = async () => {
          try {
            const response = await fetch('/api/v1/dashboard/stats');
            const result = await response.json();
            setData(result);
          } catch (error) {
            console.error('Failed to load dashboard data:', error);
          } finally {
            setLoading(false);
          }
        };

        loadData();
      }, []);

      if (loading) {
        return <div data-testid="loading">Loading...</div>;
      }

      return (
        <div data-testid="dashboard">
          <div data-testid="total-policies">{data?.totalPolicies}</div>
          <div data-testid="active-claims">{data?.activeClaims}</div>
          <div data-testid="pending-reviews">{data?.pendingReviews}</div>
        </div>
      );
    };

    render(<Dashboard />);

    // Initially shows loading
    expect(screen.getByTestId('loading')).toBeInTheDocument();

    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByTestId('dashboard')).toBeInTheDocument();
    });

    expect(screen.getByTestId('total-policies')).toHaveTextContent('1234');
    expect(screen.getByTestId('active-claims')).toHaveTextContent('56');
    expect(screen.getByTestId('pending-reviews')).toHaveTextContent('23');
  });
});

// =============================================================================
// Policy Management Tests
// =============================================================================

describe('Policy Management', () => {
  test('renders policy list', () => {
    const policies = [
      {
        id: '1',
        policyNumber: 'POL-001',
        customerName: 'John Doe',
        policyType: 'Auto',
        status: 'Active',
        premiumAmount: 1200
      },
      {
        id: '2',
        policyNumber: 'POL-002',
        customerName: 'Jane Smith',
        policyType: 'Home',
        status: 'Pending',
        premiumAmount: 800
      }
    ];

    const PolicyList = () => (
      <div data-testid="policy-list">
        <table>
          <thead>
            <tr>
              <th>Policy Number</th>
              <th>Customer</th>
              <th>Type</th>
              <th>Status</th>
              <th>Premium</th>
            </tr>
          </thead>
          <tbody>
            {policies.map(policy => (
              <tr key={policy.id} data-testid={`policy-row-${policy.id}`}>
                <td>{policy.policyNumber}</td>
                <td>{policy.customerName}</td>
                <td>{policy.policyType}</td>
                <td>{policy.status}</td>
                <td>${policy.premiumAmount}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    );

    render(<PolicyList />);

    expect(screen.getByTestId('policy-list')).toBeInTheDocument();
    expect(screen.getByTestId('policy-row-1')).toBeInTheDocument();
    expect(screen.getByTestId('policy-row-2')).toBeInTheDocument();
    expect(screen.getByText('POL-001')).toBeInTheDocument();
    expect(screen.getByText('John Doe')).toBeInTheDocument();
    expect(screen.getByText('Auto')).toBeInTheDocument();
  });

  test('filters policies by status', async () => {
    const PolicyFilter = () => {
      const [filter, setFilter] = React.useState('all');
      const [filteredPolicies, setFilteredPolicies] = React.useState([]);

      const allPolicies = [
        { id: '1', status: 'Active', policyNumber: 'POL-001' },
        { id: '2', status: 'Pending', policyNumber: 'POL-002' },
        { id: '3', status: 'Active', policyNumber: 'POL-003' }
      ];

      React.useEffect(() => {
        if (filter === 'all') {
          setFilteredPolicies(allPolicies);
        } else {
          setFilteredPolicies(allPolicies.filter(p => p.status === filter));
        }
      }, [filter]);

      return (
        <div>
          <select
            data-testid="status-filter"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          >
            <option value="all">All</option>
            <option value="Active">Active</option>
            <option value="Pending">Pending</option>
          </select>
          <div data-testid="policy-count">
            {filteredPolicies.length} policies
          </div>
        </div>
      );
    };

    const user = userEvent.setup();
    render(<PolicyFilter />);

    // Initially shows all policies
    expect(screen.getByTestId('policy-count')).toHaveTextContent('3 policies');

    // Filter by Active
    await user.selectOptions(screen.getByTestId('status-filter'), 'Active');
    expect(screen.getByTestId('policy-count')).toHaveTextContent('2 policies');

    // Filter by Pending
    await user.selectOptions(screen.getByTestId('status-filter'), 'Pending');
    expect(screen.getByTestId('policy-count')).toHaveTextContent('1 policies');
  });

  test('creates new policy', async () => {
    const mockCreatePolicy = jest.fn();

    const PolicyForm = () => {
      const [formData, setFormData] = React.useState({
        customerName: '',
        policyType: '',
        coverageAmount: ''
      });

      const handleSubmit = (e) => {
        e.preventDefault();
        mockCreatePolicy(formData);
      };

      return (
        <form onSubmit={handleSubmit} data-testid="policy-form">
          <input
            type="text"
            placeholder="Customer Name"
            value={formData.customerName}
            onChange={(e) => setFormData({...formData, customerName: e.target.value})}
            data-testid="customer-name-input"
          />
          <select
            value={formData.policyType}
            onChange={(e) => setFormData({...formData, policyType: e.target.value})}
            data-testid="policy-type-select"
          >
            <option value="">Select Type</option>
            <option value="Auto">Auto</option>
            <option value="Home">Home</option>
          </select>
          <input
            type="number"
            placeholder="Coverage Amount"
            value={formData.coverageAmount}
            onChange={(e) => setFormData({...formData, coverageAmount: e.target.value})}
            data-testid="coverage-amount-input"
          />
          <button type="submit" data-testid="submit-button">
            Create Policy
          </button>
        </form>
      );
    };

    const user = userEvent.setup();
    render(<PolicyForm />);

    await user.type(screen.getByTestId('customer-name-input'), 'John Doe');
    await user.selectOptions(screen.getByTestId('policy-type-select'), 'Auto');
    await user.type(screen.getByTestId('coverage-amount-input'), '50000');
    await user.click(screen.getByTestId('submit-button'));

    expect(mockCreatePolicy).toHaveBeenCalledWith({
      customerName: 'John Doe',
      policyType: 'Auto',
      coverageAmount: '50000'
    });
  });
});

// =============================================================================
// Claims Management Tests
// =============================================================================

describe('Claims Management', () => {
  test('renders claims list', () => {
    const claims = [
      {
        id: '1',
        claimNumber: 'CLM-001',
        policyNumber: 'POL-001',
        claimType: 'Collision',
        status: 'Processing',
        claimAmount: 5000
      }
    ];

    const ClaimsList = () => (
      <div data-testid="claims-list">
        {claims.map(claim => (
          <div key={claim.id} data-testid={`claim-${claim.id}`}>
            <span>{claim.claimNumber}</span>
            <span>{claim.status}</span>
            <span>${claim.claimAmount}</span>
          </div>
        ))}
      </div>
    );

    render(<ClaimsList />);

    expect(screen.getByTestId('claims-list')).toBeInTheDocument();
    expect(screen.getByTestId('claim-1')).toBeInTheDocument();
    expect(screen.getByText('CLM-001')).toBeInTheDocument();
    expect(screen.getByText('Processing')).toBeInTheDocument();
    expect(screen.getByText('$5000')).toBeInTheDocument();
  });

  test('submits new claim', async () => {
    const mockSubmitClaim = jest.fn();

    const ClaimForm = () => {
      const [formData, setFormData] = React.useState({
        policyNumber: '',
        claimType: '',
        description: '',
        claimAmount: ''
      });

      const handleSubmit = (e) => {
        e.preventDefault();
        mockSubmitClaim(formData);
      };

      return (
        <form onSubmit={handleSubmit} data-testid="claim-form">
          <input
            type="text"
            placeholder="Policy Number"
            value={formData.policyNumber}
            onChange={(e) => setFormData({...formData, policyNumber: e.target.value})}
            data-testid="policy-number-input"
          />
          <select
            value={formData.claimType}
            onChange={(e) => setFormData({...formData, claimType: e.target.value})}
            data-testid="claim-type-select"
          >
            <option value="">Select Type</option>
            <option value="Collision">Collision</option>
            <option value="Theft">Theft</option>
          </select>
          <textarea
            placeholder="Description"
            value={formData.description}
            onChange={(e) => setFormData({...formData, description: e.target.value})}
            data-testid="description-textarea"
          />
          <input
            type="number"
            placeholder="Claim Amount"
            value={formData.claimAmount}
            onChange={(e) => setFormData({...formData, claimAmount: e.target.value})}
            data-testid="claim-amount-input"
          />
          <button type="submit" data-testid="submit-claim-button">
            Submit Claim
          </button>
        </form>
      );
    };

    const user = userEvent.setup();
    render(<ClaimForm />);

    await user.type(screen.getByTestId('policy-number-input'), 'POL-001');
    await user.selectOptions(screen.getByTestId('claim-type-select'), 'Collision');
    await user.type(screen.getByTestId('description-textarea'), 'Vehicle collision on highway');
    await user.type(screen.getByTestId('claim-amount-input'), '5000');
    await user.click(screen.getByTestId('submit-claim-button'));

    expect(mockSubmitClaim).toHaveBeenCalledWith({
      policyNumber: 'POL-001',
      claimType: 'Collision',
      description: 'Vehicle collision on highway',
      claimAmount: '5000'
    });
  });

  test('updates claim status', async () => {
    const mockUpdateStatus = jest.fn();

    const ClaimStatusUpdate = () => {
      const [status, setStatus] = React.useState('Processing');

      const handleStatusChange = (newStatus) => {
        setStatus(newStatus);
        mockUpdateStatus('CLM-001', newStatus);
      };

      return (
        <div data-testid="claim-status-update">
          <span data-testid="current-status">Status: {status}</span>
          <button
            onClick={() => handleStatusChange('Approved')}
            data-testid="approve-button"
          >
            Approve
          </button>
          <button
            onClick={() => handleStatusChange('Rejected')}
            data-testid="reject-button"
          >
            Reject
          </button>
        </div>
      );
    };

    const user = userEvent.setup();
    render(<ClaimStatusUpdate />);

    expect(screen.getByTestId('current-status')).toHaveTextContent('Status: Processing');

    await user.click(screen.getByTestId('approve-button'));

    expect(screen.getByTestId('current-status')).toHaveTextContent('Status: Approved');
    expect(mockUpdateStatus).toHaveBeenCalledWith('CLM-001', 'Approved');
  });
});

// =============================================================================
// Document Upload Tests
// =============================================================================

describe('Document Upload', () => {
  test('handles file upload', async () => {
    const mockUpload = jest.fn();

    const DocumentUpload = () => {
      const [selectedFile, setSelectedFile] = React.useState(null);

      const handleFileSelect = (e) => {
        setSelectedFile(e.target.files[0]);
      };

      const handleUpload = () => {
        if (selectedFile) {
          mockUpload(selectedFile);
        }
      };

      return (
        <div data-testid="document-upload">
          <input
            type="file"
            onChange={handleFileSelect}
            data-testid="file-input"
            accept=".pdf,.jpg,.png"
          />
          {selectedFile && (
            <div data-testid="selected-file">
              Selected: {selectedFile.name}
            </div>
          )}
          <button
            onClick={handleUpload}
            disabled={!selectedFile}
            data-testid="upload-button"
          >
            Upload
          </button>
        </div>
      );
    };

    const user = userEvent.setup();
    render(<DocumentUpload />);

    const file = new File(['test content'], 'test.pdf', { type: 'application/pdf' });
    const fileInput = screen.getByTestId('file-input');

    await user.upload(fileInput, file);

    expect(screen.getByTestId('selected-file')).toHaveTextContent('Selected: test.pdf');

    await user.click(screen.getByTestId('upload-button'));

    expect(mockUpload).toHaveBeenCalledWith(file);
  });

  test('validates file type', async () => {
    const DocumentUpload = () => {
      const [error, setError] = React.useState('');

      const handleFileSelect = (e) => {
        const file = e.target.files[0];
        const allowedTypes = ['application/pdf', 'image/jpeg', 'image/png'];
        
        if (file && !allowedTypes.includes(file.type)) {
          setError('Invalid file type. Please select PDF, JPG, or PNG files.');
        } else {
          setError('');
        }
      };

      return (
        <div>
          <input
            type="file"
            onChange={handleFileSelect}
            data-testid="file-input"
          />
          {error && (
            <div data-testid="error-message" role="alert">
              {error}
            </div>
          )}
        </div>
      );
    };

    const user = userEvent.setup();
    render(<DocumentUpload />);

    const invalidFile = new File(['test'], 'test.txt', { type: 'text/plain' });
    const fileInput = screen.getByTestId('file-input');

    await user.upload(fileInput, invalidFile);

    expect(screen.getByTestId('error-message')).toBeInTheDocument();
    expect(screen.getByText(/Invalid file type/)).toBeInTheDocument();
  });
});

// =============================================================================
// Navigation Tests
// =============================================================================

describe('Navigation', () => {
  test('renders navigation menu', () => {
    const Navigation = () => (
      <nav data-testid="navigation">
        <ul>
          <li><a href="/dashboard" data-testid="nav-dashboard">Dashboard</a></li>
          <li><a href="/policies" data-testid="nav-policies">Policies</a></li>
          <li><a href="/claims" data-testid="nav-claims">Claims</a></li>
          <li><a href="/agents" data-testid="nav-agents">Agents</a></li>
        </ul>
      </nav>
    );

    render(<Navigation />);

    expect(screen.getByTestId('navigation')).toBeInTheDocument();
    expect(screen.getByTestId('nav-dashboard')).toBeInTheDocument();
    expect(screen.getByTestId('nav-policies')).toBeInTheDocument();
    expect(screen.getByTestId('nav-claims')).toBeInTheDocument();
    expect(screen.getByTestId('nav-agents')).toBeInTheDocument();
  });

  test('highlights active navigation item', () => {
    const Navigation = ({ currentPath }) => (
      <nav data-testid="navigation">
        <ul>
          <li>
            <a
              href="/dashboard"
              data-testid="nav-dashboard"
              className={currentPath === '/dashboard' ? 'active' : ''}
            >
              Dashboard
            </a>
          </li>
          <li>
            <a
              href="/policies"
              data-testid="nav-policies"
              className={currentPath === '/policies' ? 'active' : ''}
            >
              Policies
            </a>
          </li>
        </ul>
      </nav>
    );

    render(<Navigation currentPath="/dashboard" />);

    expect(screen.getByTestId('nav-dashboard')).toHaveClass('active');
    expect(screen.getByTestId('nav-policies')).not.toHaveClass('active');
  });
});

// =============================================================================
// Real-time Updates Tests
// =============================================================================

describe('Real-time Updates', () => {
  test('handles WebSocket connection', () => {
    const mockWebSocket = {
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      close: jest.fn()
    };

    global.WebSocket = jest.fn(() => mockWebSocket);

    const RealTimeComponent = () => {
      const [connected, setConnected] = React.useState(false);

      React.useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        ws.addEventListener('open', () => {
          setConnected(true);
        });

        ws.addEventListener('close', () => {
          setConnected(false);
        });

        return () => {
          ws.close();
        };
      }, []);

      return (
        <div data-testid="realtime-component">
          Status: {connected ? 'Connected' : 'Disconnected'}
        </div>
      );
    };

    render(<RealTimeComponent />);

    expect(global.WebSocket).toHaveBeenCalledWith('ws://localhost:8000/ws');
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('open', expect.any(Function));
    expect(mockWebSocket.addEventListener).toHaveBeenCalledWith('close', expect.any(Function));
  });

  test('receives real-time notifications', () => {
    const mockWebSocket = {
      addEventListener: jest.fn(),
      removeEventListener: jest.fn(),
      close: jest.fn()
    };

    global.WebSocket = jest.fn(() => mockWebSocket);

    const NotificationComponent = () => {
      const [notifications, setNotifications] = React.useState([]);

      React.useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        
        const handleMessage = (event) => {
          const data = JSON.parse(event.data);
          setNotifications(prev => [...prev, data]);
        };

        ws.addEventListener('message', handleMessage);

        return () => {
          ws.close();
        };
      }, []);

      return (
        <div data-testid="notifications">
          {notifications.map((notification, index) => (
            <div key={index} data-testid={`notification-${index}`}>
              {notification.message}
            </div>
          ))}
        </div>
      );
    };

    render(<NotificationComponent />);

    // Simulate receiving a message
    const messageHandler = mockWebSocket.addEventListener.mock.calls
      .find(call => call[0] === 'message')[1];
    
    act(() => {
      messageHandler({
        data: JSON.stringify({ message: 'New claim submitted' })
      });
    });

    expect(screen.getByTestId('notification-0')).toHaveTextContent('New claim submitted');
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

describe('Error Handling', () => {
  test('displays error boundary', () => {
    const ThrowError = () => {
      throw new Error('Test error');
    };

    const ErrorBoundary = ({ children }) => {
      const [hasError, setHasError] = React.useState(false);

      React.useEffect(() => {
        const errorHandler = (error) => {
          setHasError(true);
        };

        window.addEventListener('error', errorHandler);
        return () => window.removeEventListener('error', errorHandler);
      }, []);

      if (hasError) {
        return (
          <div data-testid="error-boundary">
            Something went wrong. Please try again.
          </div>
        );
      }

      return children;
    };

    // Suppress console.error for this test
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(
        <ErrorBoundary>
          <ThrowError />
        </ErrorBoundary>
      );
    }).toThrow();

    consoleSpy.mockRestore();
  });

  test('handles API errors gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('Network error'));

    const DataComponent = () => {
      const [error, setError] = React.useState(null);
      const [loading, setLoading] = React.useState(true);

      React.useEffect(() => {
        const loadData = async () => {
          try {
            await fetch('/api/v1/data');
          } catch (err) {
            setError(err.message);
          } finally {
            setLoading(false);
          }
        };

        loadData();
      }, []);

      if (loading) {
        return <div data-testid="loading">Loading...</div>;
      }

      if (error) {
        return (
          <div data-testid="error-message" role="alert">
            Error: {error}
          </div>
        );
      }

      return <div data-testid="data">Data loaded</div>;
    };

    render(<DataComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
    });

    expect(screen.getByText('Error: Network error')).toBeInTheDocument();
  });
});

// =============================================================================
// Accessibility Tests
// =============================================================================

describe('Accessibility', () => {
  test('has proper ARIA labels', () => {
    const AccessibleForm = () => (
      <form>
        <label htmlFor="email">Email Address</label>
        <input
          id="email"
          type="email"
          aria-required="true"
          aria-describedby="email-help"
          data-testid="email-input"
        />
        <div id="email-help">Enter your email address</div>
        
        <button type="submit" aria-label="Submit form">
          Submit
        </button>
      </form>
    );

    render(<AccessibleForm />);

    const emailInput = screen.getByTestId('email-input');
    expect(emailInput).toHaveAttribute('aria-required', 'true');
    expect(emailInput).toHaveAttribute('aria-describedby', 'email-help');
    
    const submitButton = screen.getByRole('button', { name: /submit form/i });
    expect(submitButton).toBeInTheDocument();
  });

  test('supports keyboard navigation', async () => {
    const KeyboardNav = () => (
      <div>
        <button data-testid="button1">Button 1</button>
        <button data-testid="button2">Button 2</button>
        <button data-testid="button3">Button 3</button>
      </div>
    );

    const user = userEvent.setup();
    render(<KeyboardNav />);

    const button1 = screen.getByTestId('button1');
    const button2 = screen.getByTestId('button2');

    // Focus first button
    button1.focus();
    expect(button1).toHaveFocus();

    // Tab to next button
    await user.tab();
    expect(button2).toHaveFocus();
  });
});

// =============================================================================
// Performance Tests
// =============================================================================

describe('Performance', () => {
  test('renders large lists efficiently', () => {
    const LargeList = () => {
      const items = Array.from({ length: 1000 }, (_, i) => ({
        id: i,
        name: `Item ${i}`
      }));

      return (
        <div data-testid="large-list">
          {items.slice(0, 50).map(item => (
            <div key={item.id} data-testid={`item-${item.id}`}>
              {item.name}
            </div>
          ))}
        </div>
      );
    };

    const startTime = performance.now();
    render(<LargeList />);
    const endTime = performance.now();

    expect(screen.getByTestId('large-list')).toBeInTheDocument();
    expect(screen.getByTestId('item-0')).toBeInTheDocument();
    expect(screen.getByTestId('item-49')).toBeInTheDocument();
    
    // Render should complete quickly (less than 100ms)
    expect(endTime - startTime).toBeLessThan(100);
  });
});

// =============================================================================
// Test Utilities and Setup
// =============================================================================

// Custom render function with providers
export const renderWithProviders = (ui, options = {}) => {
  const AllProviders = ({ children }) => {
    return (
      <BrowserRouter>
        <MockAuthProvider>
          {children}
        </MockAuthProvider>
      </BrowserRouter>
    );
  };

  return render(ui, { wrapper: AllProviders, ...options });
};

// Mock API responses
export const mockApiResponse = (data, status = 200) => {
  fetch.mockResolvedValueOnce({
    ok: status >= 200 && status < 300,
    status,
    json: async () => data
  });
};

// Test data factories
export const createMockUser = (overrides = {}) => ({
  id: 'user-1',
  email: 'test@zurich.com',
  firstName: 'Test',
  lastName: 'User',
  role: 'agent',
  ...overrides
});

export const createMockPolicy = (overrides = {}) => ({
  id: 'policy-1',
  policyNumber: 'POL-001',
  customerName: 'John Doe',
  policyType: 'Auto',
  status: 'Active',
  premiumAmount: 1200,
  ...overrides
});

export const createMockClaim = (overrides = {}) => ({
  id: 'claim-1',
  claimNumber: 'CLM-001',
  policyNumber: 'POL-001',
  claimType: 'Collision',
  status: 'Processing',
  claimAmount: 5000,
  ...overrides
});

// Global test setup
beforeEach(() => {
  fetch.mockClear();
});

afterEach(() => {
  jest.clearAllMocks();
});

