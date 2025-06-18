/**
 * Insurance AI Agent System - End-to-End Tests
 * Comprehensive E2E testing using Playwright
 */

import { test, expect } from '@playwright/test';

// Test configuration
const BASE_URL = process.env.BASE_URL || 'http://localhost:80';
const API_URL = process.env.API_URL || 'http://localhost:8000';

// Test data
const TEST_USER = {
  email: 'admin@zurich.com',
  password: 'password123',
  firstName: 'Admin',
  lastName: 'User'
};

const TEST_POLICY = {
  customerName: 'John Doe',
  policyType: 'auto',
  coverageAmount: '50000',
  premiumAmount: '1200'
};

const TEST_CLAIM = {
  policyNumber: 'POL-TEST-001',
  claimType: 'collision',
  description: 'Vehicle collision on highway',
  claimAmount: '5000'
};

// =============================================================================
// Authentication Tests
// =============================================================================

test.describe('Authentication', () => {
  test('should login successfully with valid credentials', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Should show login form
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    
    // Fill login form
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    
    // Submit form
    await page.click('[data-testid="login-button"]');
    
    // Should redirect to dashboard
    await expect(page).toHaveURL(/.*dashboard/);
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('should show error for invalid credentials', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Fill with invalid credentials
    await page.fill('[data-testid="email-input"]', 'invalid@example.com');
    await page.fill('[data-testid="password-input"]', 'wrongpassword');
    
    // Submit form
    await page.click('[data-testid="login-button"]');
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('Invalid');
  });

  test('should logout successfully', async ({ page }) => {
    // Login first
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Wait for dashboard
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    // Logout
    await page.click('[data-testid="user-menu"]');
    await page.click('[data-testid="logout-button"]');
    
    // Should redirect to login
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
  });
});

// =============================================================================
// Dashboard Tests
// =============================================================================

test.describe('Dashboard', () => {
  test.beforeEach(async ({ page }) => {
    // Login before each test
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('should display KPI cards', async ({ page }) => {
    // Check KPI cards are visible
    await expect(page.locator('[data-testid="kpi-cards"]')).toBeVisible();
    
    // Check individual KPI cards
    const kpiCards = page.locator('[data-testid="kpi-card"]');
    await expect(kpiCards).toHaveCount(4); // Total Policies, Active Claims, Pending Reviews, Revenue
    
    // Check KPI values are displayed
    await expect(page.locator('[data-testid="total-policies-value"]')).toBeVisible();
    await expect(page.locator('[data-testid="active-claims-value"]')).toBeVisible();
    await expect(page.locator('[data-testid="pending-reviews-value"]')).toBeVisible();
    await expect(page.locator('[data-testid="revenue-value"]')).toBeVisible();
  });

  test('should display charts', async ({ page }) => {
    // Check charts section is visible
    await expect(page.locator('[data-testid="charts-section"]')).toBeVisible();
    
    // Check individual charts
    await expect(page.locator('[data-testid="policies-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="claims-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="revenue-chart"]')).toBeVisible();
  });

  test('should display recent activities', async ({ page }) => {
    // Check recent activities section
    await expect(page.locator('[data-testid="recent-activities"]')).toBeVisible();
    
    // Check activity items
    const activityItems = page.locator('[data-testid="activity-item"]');
    await expect(activityItems.first()).toBeVisible();
  });

  test('should refresh data when refresh button is clicked', async ({ page }) => {
    // Click refresh button
    await page.click('[data-testid="refresh-button"]');
    
    // Check loading indicator appears
    await expect(page.locator('[data-testid="loading-indicator"]')).toBeVisible();
    
    // Wait for loading to complete
    await expect(page.locator('[data-testid="loading-indicator"]')).not.toBeVisible();
    
    // Check data is still displayed
    await expect(page.locator('[data-testid="kpi-cards"]')).toBeVisible();
  });
});

// =============================================================================
// Policy Management Tests
// =============================================================================

test.describe('Policy Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login and navigate to policies
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await page.click('[data-testid="nav-policies"]');
    await expect(page.locator('[data-testid="policies-page"]')).toBeVisible();
  });

  test('should display policies list', async ({ page }) => {
    // Check policies table is visible
    await expect(page.locator('[data-testid="policies-table"]')).toBeVisible();
    
    // Check table headers
    await expect(page.locator('th:has-text("Policy Number")')).toBeVisible();
    await expect(page.locator('th:has-text("Customer")')).toBeVisible();
    await expect(page.locator('th:has-text("Type")')).toBeVisible();
    await expect(page.locator('th:has-text("Status")')).toBeVisible();
    
    // Check at least one policy row exists
    const policyRows = page.locator('[data-testid="policy-row"]');
    await expect(policyRows.first()).toBeVisible();
  });

  test('should filter policies by status', async ({ page }) => {
    // Select status filter
    await page.selectOption('[data-testid="status-filter"]', 'active');
    
    // Check filtered results
    const policyRows = page.locator('[data-testid="policy-row"]');
    const count = await policyRows.count();
    
    // Verify all visible policies have 'Active' status
    for (let i = 0; i < count; i++) {
      const statusCell = policyRows.nth(i).locator('[data-testid="policy-status"]');
      await expect(statusCell).toContainText('Active');
    }
  });

  test('should search policies by policy number', async ({ page }) => {
    // Enter search term
    await page.fill('[data-testid="search-input"]', 'POL-001');
    await page.press('[data-testid="search-input"]', 'Enter');
    
    // Check search results
    const policyRows = page.locator('[data-testid="policy-row"]');
    const firstRow = policyRows.first();
    await expect(firstRow.locator('[data-testid="policy-number"]')).toContainText('POL-001');
  });

  test('should create new policy', async ({ page }) => {
    // Click create policy button
    await page.click('[data-testid="create-policy-button"]');
    
    // Check modal is visible
    await expect(page.locator('[data-testid="policy-modal"]')).toBeVisible();
    
    // Fill policy form
    await page.fill('[data-testid="customer-name-input"]', TEST_POLICY.customerName);
    await page.selectOption('[data-testid="policy-type-select"]', TEST_POLICY.policyType);
    await page.fill('[data-testid="coverage-amount-input"]', TEST_POLICY.coverageAmount);
    await page.fill('[data-testid="premium-amount-input"]', TEST_POLICY.premiumAmount);
    
    // Submit form
    await page.click('[data-testid="submit-policy-button"]');
    
    // Check success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('Policy created successfully');
    
    // Check modal is closed
    await expect(page.locator('[data-testid="policy-modal"]')).not.toBeVisible();
  });

  test('should view policy details', async ({ page }) => {
    // Click on first policy row
    const firstPolicyRow = page.locator('[data-testid="policy-row"]').first();
    await firstPolicyRow.click();
    
    // Check policy details page
    await expect(page.locator('[data-testid="policy-details"]')).toBeVisible();
    
    // Check policy information sections
    await expect(page.locator('[data-testid="policy-info"]')).toBeVisible();
    await expect(page.locator('[data-testid="customer-info"]')).toBeVisible();
    await expect(page.locator('[data-testid="coverage-info"]')).toBeVisible();
  });
});

// =============================================================================
// Claims Management Tests
// =============================================================================

test.describe('Claims Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login and navigate to claims
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await page.click('[data-testid="nav-claims"]');
    await expect(page.locator('[data-testid="claims-page"]')).toBeVisible();
  });

  test('should display claims list', async ({ page }) => {
    // Check claims table is visible
    await expect(page.locator('[data-testid="claims-table"]')).toBeVisible();
    
    // Check table headers
    await expect(page.locator('th:has-text("Claim Number")')).toBeVisible();
    await expect(page.locator('th:has-text("Policy Number")')).toBeVisible();
    await expect(page.locator('th:has-text("Type")')).toBeVisible();
    await expect(page.locator('th:has-text("Status")')).toBeVisible();
    await expect(page.locator('th:has-text("Amount")')).toBeVisible();
  });

  test('should submit new claim', async ({ page }) => {
    // Click submit claim button
    await page.click('[data-testid="submit-claim-button"]');
    
    // Check modal is visible
    await expect(page.locator('[data-testid="claim-modal"]')).toBeVisible();
    
    // Fill claim form
    await page.fill('[data-testid="policy-number-input"]', TEST_CLAIM.policyNumber);
    await page.selectOption('[data-testid="claim-type-select"]', TEST_CLAIM.claimType);
    await page.fill('[data-testid="description-textarea"]', TEST_CLAIM.description);
    await page.fill('[data-testid="claim-amount-input"]', TEST_CLAIM.claimAmount);
    
    // Submit form
    await page.click('[data-testid="submit-claim-form-button"]');
    
    // Check success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="success-message"]')).toContainText('Claim submitted successfully');
  });

  test('should update claim status', async ({ page }) => {
    // Click on first claim row
    const firstClaimRow = page.locator('[data-testid="claim-row"]').first();
    await firstClaimRow.click();
    
    // Check claim details page
    await expect(page.locator('[data-testid="claim-details"]')).toBeVisible();
    
    // Click update status button
    await page.click('[data-testid="update-status-button"]');
    
    // Select new status
    await page.selectOption('[data-testid="status-select"]', 'approved');
    
    // Add comment
    await page.fill('[data-testid="comment-textarea"]', 'Claim approved after review');
    
    // Submit update
    await page.click('[data-testid="update-status-submit"]');
    
    // Check success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });

  test('should upload claim documents', async ({ page }) => {
    // Click on first claim row
    const firstClaimRow = page.locator('[data-testid="claim-row"]').first();
    await firstClaimRow.click();
    
    // Navigate to documents tab
    await page.click('[data-testid="documents-tab"]');
    
    // Check documents section
    await expect(page.locator('[data-testid="documents-section"]')).toBeVisible();
    
    // Click upload button
    await page.click('[data-testid="upload-document-button"]');
    
    // Check upload modal
    await expect(page.locator('[data-testid="upload-modal"]')).toBeVisible();
    
    // Upload file (mock file upload)
    const fileInput = page.locator('[data-testid="file-input"]');
    await fileInput.setInputFiles({
      name: 'test-document.pdf',
      mimeType: 'application/pdf',
      buffer: Buffer.from('test document content')
    });
    
    // Add document description
    await page.fill('[data-testid="document-description"]', 'Accident report');
    
    // Submit upload
    await page.click('[data-testid="upload-submit-button"]');
    
    // Check success message
    await expect(page.locator('[data-testid="success-message"]')).toBeVisible();
  });
});

// =============================================================================
// Agent Management Tests
// =============================================================================

test.describe('Agent Management', () => {
  test.beforeEach(async ({ page }) => {
    // Login and navigate to agents
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await page.click('[data-testid="nav-agents"]');
    await expect(page.locator('[data-testid="agents-page"]')).toBeVisible();
  });

  test('should display agent status dashboard', async ({ page }) => {
    // Check agent status cards
    await expect(page.locator('[data-testid="agent-status-cards"]')).toBeVisible();
    
    // Check individual agent cards
    await expect(page.locator('[data-testid="document-analysis-agent"]')).toBeVisible();
    await expect(page.locator('[data-testid="risk-assessment-agent"]')).toBeVisible();
    await expect(page.locator('[data-testid="communication-agent"]')).toBeVisible();
    await expect(page.locator('[data-testid="evidence-processing-agent"]')).toBeVisible();
    await expect(page.locator('[data-testid="compliance-agent"]')).toBeVisible();
  });

  test('should show agent performance metrics', async ({ page }) => {
    // Check performance metrics section
    await expect(page.locator('[data-testid="performance-metrics"]')).toBeVisible();
    
    // Check metrics charts
    await expect(page.locator('[data-testid="agent-performance-chart"]')).toBeVisible();
    await expect(page.locator('[data-testid="task-completion-chart"]')).toBeVisible();
  });

  test('should display active workflows', async ({ page }) => {
    // Check workflows section
    await expect(page.locator('[data-testid="active-workflows"]')).toBeVisible();
    
    // Check workflow items
    const workflowItems = page.locator('[data-testid="workflow-item"]');
    await expect(workflowItems.first()).toBeVisible();
  });
});

// =============================================================================
// Real-time Updates Tests
// =============================================================================

test.describe('Real-time Updates', () => {
  test.beforeEach(async ({ page }) => {
    // Login
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
  });

  test('should display real-time notifications', async ({ page }) => {
    // Check notifications bell
    await expect(page.locator('[data-testid="notifications-bell"]')).toBeVisible();
    
    // Click notifications bell
    await page.click('[data-testid="notifications-bell"]');
    
    // Check notifications dropdown
    await expect(page.locator('[data-testid="notifications-dropdown"]')).toBeVisible();
    
    // Check notification items
    const notificationItems = page.locator('[data-testid="notification-item"]');
    if (await notificationItems.count() > 0) {
      await expect(notificationItems.first()).toBeVisible();
    }
  });

  test('should show connection status', async ({ page }) => {
    // Check connection status indicator
    await expect(page.locator('[data-testid="connection-status"]')).toBeVisible();
    
    // Should show connected status
    await expect(page.locator('[data-testid="connection-status"]')).toContainText('Connected');
  });
});

// =============================================================================
// Responsive Design Tests
// =============================================================================

test.describe('Responsive Design', () => {
  test('should work on mobile devices', async ({ page }) => {
    // Set mobile viewport
    await page.setViewportSize({ width: 375, height: 667 });
    
    // Login
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Check mobile navigation
    await expect(page.locator('[data-testid="mobile-menu-button"]')).toBeVisible();
    
    // Open mobile menu
    await page.click('[data-testid="mobile-menu-button"]');
    await expect(page.locator('[data-testid="mobile-menu"]')).toBeVisible();
    
    // Check navigation items
    await expect(page.locator('[data-testid="mobile-nav-dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="mobile-nav-policies"]')).toBeVisible();
    await expect(page.locator('[data-testid="mobile-nav-claims"]')).toBeVisible();
  });

  test('should work on tablet devices', async ({ page }) => {
    // Set tablet viewport
    await page.setViewportSize({ width: 768, height: 1024 });
    
    // Login
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Check tablet layout
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    await expect(page.locator('[data-testid="sidebar"]')).toBeVisible();
    
    // Check KPI cards layout
    const kpiCards = page.locator('[data-testid="kpi-card"]');
    await expect(kpiCards).toHaveCount(4);
  });
});

// =============================================================================
// Performance Tests
// =============================================================================

test.describe('Performance', () => {
  test('should load dashboard within acceptable time', async ({ page }) => {
    const startTime = Date.now();
    
    // Login and navigate to dashboard
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    await expect(page.locator('[data-testid="dashboard"]')).toBeVisible();
    
    const loadTime = Date.now() - startTime;
    
    // Dashboard should load within 5 seconds
    expect(loadTime).toBeLessThan(5000);
  });

  test('should handle large data sets efficiently', async ({ page }) => {
    // Login
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Navigate to policies with large dataset
    await page.click('[data-testid="nav-policies"]');
    
    const startTime = Date.now();
    await expect(page.locator('[data-testid="policies-table"]')).toBeVisible();
    const loadTime = Date.now() - startTime;
    
    // Large dataset should load within 3 seconds
    expect(loadTime).toBeLessThan(3000);
  });
});

// =============================================================================
// Accessibility Tests
// =============================================================================

test.describe('Accessibility', () => {
  test('should have proper ARIA labels', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Check form accessibility
    const emailInput = page.locator('[data-testid="email-input"]');
    await expect(emailInput).toHaveAttribute('aria-label');
    
    const passwordInput = page.locator('[data-testid="password-input"]');
    await expect(passwordInput).toHaveAttribute('aria-label');
    
    const loginButton = page.locator('[data-testid="login-button"]');
    await expect(loginButton).toHaveAttribute('aria-label');
  });

  test('should support keyboard navigation', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // Tab through form elements
    await page.press('body', 'Tab');
    await expect(page.locator('[data-testid="email-input"]')).toBeFocused();
    
    await page.press('body', 'Tab');
    await expect(page.locator('[data-testid="password-input"]')).toBeFocused();
    
    await page.press('body', 'Tab');
    await expect(page.locator('[data-testid="login-button"]')).toBeFocused();
  });

  test('should have sufficient color contrast', async ({ page }) => {
    await page.goto(BASE_URL);
    
    // This would typically use axe-core or similar accessibility testing library
    // For now, we'll check that elements are visible and readable
    await expect(page.locator('[data-testid="login-form"]')).toBeVisible();
    await expect(page.locator('h1')).toBeVisible();
  });
});

// =============================================================================
// Error Handling Tests
// =============================================================================

test.describe('Error Handling', () => {
  test('should handle network errors gracefully', async ({ page }) => {
    // Intercept API calls and return errors
    await page.route('**/api/**', route => {
      route.fulfill({
        status: 500,
        contentType: 'application/json',
        body: JSON.stringify({ error: 'Internal server error' })
      });
    });
    
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Should show error message
    await expect(page.locator('[data-testid="error-message"]')).toBeVisible();
    await expect(page.locator('[data-testid="error-message"]')).toContainText('error');
  });

  test('should handle timeout errors', async ({ page }) => {
    // Set short timeout
    page.setDefaultTimeout(1000);
    
    // Intercept API calls and delay response
    await page.route('**/api/**', route => {
      setTimeout(() => {
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ success: true })
        });
      }, 2000); // 2 second delay
    });
    
    await page.goto(BASE_URL);
    await page.fill('[data-testid="email-input"]', TEST_USER.email);
    await page.fill('[data-testid="password-input"]', TEST_USER.password);
    await page.click('[data-testid="login-button"]');
    
    // Should show timeout or loading state
    await expect(page.locator('[data-testid="loading-indicator"]')).toBeVisible();
  });
});

