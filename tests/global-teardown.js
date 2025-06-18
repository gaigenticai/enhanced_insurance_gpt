/**
 * Insurance AI Agent System - Global Test Teardown
 * Cleanup for Playwright end-to-end tests
 */

import { chromium } from '@playwright/test';

async function globalTeardown() {
  console.log('🧹 Starting global test teardown...');
  
  // Launch browser for cleanup
  const browser = await chromium.launch();
  const page = await browser.newPage();
  
  try {
    // Clean up test data
    console.log('🗑️ Cleaning up test data...');
    
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
        
        // Delete test claims
        const testClaimNumbers = ['CLM-TEST-001'];
        for (const claimNumber of testClaimNumbers) {
          try {
            await page.request.delete(`http://localhost:8000/api/v1/claims/${claimNumber}`, {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            });
            console.log(`✅ Deleted test claim: ${claimNumber}`);
          } catch (error) {
            console.log(`ℹ️ Test claim ${claimNumber} not found or already deleted`);
          }
        }
        
        // Delete test policies
        const testPolicyNumbers = ['POL-TEST-001', 'POL-TEST-002'];
        for (const policyNumber of testPolicyNumbers) {
          try {
            await page.request.delete(`http://localhost:8000/api/v1/policies/${policyNumber}`, {
              headers: {
                'Authorization': `Bearer ${token}`
              }
            });
            console.log(`✅ Deleted test policy: ${policyNumber}`);
          } catch (error) {
            console.log(`ℹ️ Test policy ${policyNumber} not found or already deleted`);
          }
        }
        
        // Note: We don't delete the test user as it might be needed for other tests
        // In a real scenario, you might want to clean up test users as well
        
        console.log('✅ Test data cleanup completed');
      }
    } catch (error) {
      console.log('⚠️ Failed to clean up test data:', error.message);
    }
    
    // Generate test report summary
    console.log('📊 Generating test report summary...');
    
    try {
      // Check if test results exist
      const fs = await import('fs');
      const path = await import('path');
      
      const resultsPath = path.join(process.cwd(), 'test-results');
      
      if (fs.existsSync(resultsPath)) {
        const files = fs.readdirSync(resultsPath);
        const reportFiles = files.filter(file => 
          file.endsWith('.json') || file.endsWith('.xml') || file.endsWith('.html')
        );
        
        console.log(`📁 Test results generated: ${reportFiles.length} files`);
        reportFiles.forEach(file => {
          console.log(`   - ${file}`);
        });
      }
    } catch (error) {
      console.log('⚠️ Failed to generate test report summary:', error.message);
    }
    
    console.log('✅ Global test teardown completed successfully');
    
  } catch (error) {
    console.error('❌ Global test teardown failed:', error.message);
    // Don't throw error in teardown to avoid masking test failures
  } finally {
    await browser.close();
  }
}

export default globalTeardown;

