// API Testing utilities for debugging

export const testApiConnection = async () => {
  const apiUrl = (import.meta as any).env.VITE_API_URL || 'http://localhost:8080'
  
  console.log('🔍 Testing API connection...')
  console.log('📍 API URL:', apiUrl)
  
  try {
    // Test basic connectivity
    console.log('1️⃣ Testing basic connectivity...')
    const healthResponse = await fetch(`${apiUrl}/health`)
    console.log('Health check status:', healthResponse.status)
    
    if (healthResponse.ok) {
      const healthData = await healthResponse.json()
      console.log('✅ Health check passed:', healthData)
    } else {
      console.log('❌ Health check failed:', healthResponse.statusText)
    }
    
    // Test documents endpoint
    console.log('2️⃣ Testing documents endpoint...')
    const docsResponse = await fetch(`${apiUrl}/api/v1/documents`)
    console.log('Documents endpoint status:', docsResponse.status)
    
    if (docsResponse.ok) {
      const docsData = await docsResponse.json()
      console.log('✅ Documents endpoint working:', docsData)
    } else {
      console.log('❌ Documents endpoint failed:', docsResponse.statusText)
    }
    
    // Test processing status
    console.log('3️⃣ Testing processing status...')
    const statusResponse = await fetch(`${apiUrl}/api/v1/documents/status/summary`)
    console.log('Processing status endpoint status:', statusResponse.status)
    
    if (statusResponse.ok) {
      const statusData = await statusResponse.json()
      console.log('✅ Processing status working:', statusData)
    } else {
      console.log('❌ Processing status failed:', statusResponse.statusText)
    }
    
  } catch (error) {
    console.error('❌ API connection test failed:', error)
    console.error('Error details:', {
      message: (error as Error).message,
      name: (error as Error).name,
      stack: (error as Error).stack
    })
  }
}

// Make it available globally for debugging
if (typeof window !== 'undefined') {
  (window as any).testApiConnection = testApiConnection
  console.log('🛠️ API test function available as window.testApiConnection()')
}
