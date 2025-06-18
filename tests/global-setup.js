/**
 * Insurance AI Agent System - Global Test Setup
 * Setup for Playwright end-to-end tests
 */

import { chromium } from '@playwright/test';

async function globalSetup() {
  console.log('🚀 Starting global test setup...');
  
  // Launch browser for setup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Wait for services to be ready
    console.log('⏳ Waiting for services to be ready...');
    
    // Check frontend is ready
    let frontendReady = false;
    let attempts = 0;
    const maxAttempts = 30;
    
    while (!frontendReady && attempts < maxAttempts) {
      try {
        const response = await page.goto('http://localhost:5173', { 
          waitUntil: 'networkidle',
          timeout: 5000 
        });
        if (response && response.ok()) {
          frontendReady = true;
          console.log('✅ Frontend is ready');
        }
      } catch (error) {
        attempts++;
        console.log(`⏳ Frontend not ready, attempt ${attempts}/${maxAttempts}`);
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    if (!frontendReady) {
      throw new Error('Frontend failed to start within timeout period');
    }
    
    // Check backend is ready
    let backendReady = false;
    attempts = 0;
    
    while (!backendReady && attempts < maxAttempts) {
      try {
        const response = await page.goto('http://localhost:8000/health', { 
          timeout: 5000 
        });
        if (response && response.ok()) {
          const data = await response.json();
          if (data.status === 'healthy') {
            backendReady = true;
            console.log('✅ Backend is ready');
          }
        }
      } catch (error) {
        attempts++;
        console.log(`⏳ Backend not ready, attempt ${attempts}/${maxAttempts}`);
        await new Promise(resolve => setTimeout(resolve, 2000));
      }
    }
    
    if (!backendReady) {
      throw new Error('Backend failed to start within timeout period');
    }
    
    // Setup test data
    console.log('📊 Setting up test data...');
    
    // Create test user if not exists
    try {
      await page.request.post('http://localhost:8000/api/v1/auth/register', {
        data: {
          email: 'admin@zurich.com',
          password: 'password123',
          firstName: 'Admin',
          lastName: 'User',
          role: 'admin',
          department: 'management'
        }
      });
      console.log('✅ Test user created');
    } catch (error) {
      console.log('ℹ️ Test user already exists or creation failed');
    }
    
    // Create test policies
    try {
      // Login to get token
      const loginResponse = await page.request.post('http://localhost:8000/api/v1/auth/login', {
        data: {
          email: 'admin@zurich.com',
          password: 'password123'
        }
      });
      
      if (loginResponse.ok()) {
        const loginData = await loginResponse.json();
        const token = loginData.access_token;
        
        // Create test policies
        const testPolicies = [
          {
            policy_number: 'POL-TEST-001',
            policy_type: 'auto',
            customer_id: 'CUST-001',
            customer_name: 'John Doe',
            premium_amount: 1200.00,
            coverage_amount: 50000.00,
            status: 'active'
          },
          {
            policy_number: 'POL-TEST-002',
            policy_type: 'home',
            customer_id: 'CUST-002',
            customer_name: 'Jane Smith',
            premium_amount: 800.00,
            coverage_amount: 200000.00,
            status: 'active'
          }
        ];
        
        for (const policy of testPolicies) {
          try {
            await page.request.post('http://localhost:8000/api/v1/policies', {
              headers: {
                'Authorization': `Bearer ${token}`
              },
              data: policy
            });
          } catch (error) {
            console.log(`ℹ️ Test policy ${policy.policy_number} already exists or creation failed`);
          }
        }
        
        console.log('✅ Test policies created');
        
        // Create test claims
        const testClaims = [
          {
            claim_number: 'CLM-TEST-001',
            policy_number: 'POL-TEST-001',
            claim_type: 'collision',
            claim_amount: 5000.00,
            description: 'Vehicle collision on highway',
            status: 'processing'
          }
        ];
        
        for (const claim of testClaims) {
          try {
            await page.request.post('http://localhost:8000/api/v1/claims', {
              headers: {
                'Authorization': `Bearer ${token}`
              },
              data: claim
            });
          } catch (error) {
            console.log(`ℹ️ Test claim ${claim.claim_number} already exists or creation failed`);
          }
        }
        
        console.log('✅ Test claims created');
      }
    } catch (error) {
      console.log('⚠️ Failed to create test data:', error.message);
    }
    
    console.log('✅ Global test setup completed successfully');
    
  } catch (error) {
    console.error('❌ Global test setup failed:', error.message);
    throw error;
  } finally {
    await browser.close();
  }
}

export default globalSetup;

