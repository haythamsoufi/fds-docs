import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { Toaster } from 'react-hot-toast'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import Documents from './pages/Documents'
import Query from './pages/Query'
import Analytics from './pages/Analytics'
import Settings from './pages/Settings'
import Monitoring from './pages/Monitoring'
import Methodology from './pages/Methodology'
import './utils/apiTest' // Import for global API test function

function App() {
  return (
    <Router
      future={{
        v7_startTransition: true,
        v7_relativeSplatPath: true
      }}
    >
      <div className="min-h-screen bg-ifrc-white flex flex-col">
        <Layout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/documents" element={<Documents />} />
            <Route path="/query" element={<Query />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/settings" element={<Settings />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/methodology" element={<Methodology />} />
          </Routes>
        </Layout>
        <Toaster 
          position="bottom-left"
          toastOptions={{
            duration: 4000,
            className: 'lg:ml-64',
            style: {
              background: '#DC143C', // IFRC Red
              color: '#fff',
            },
          }}
        />
      </div>
    </Router>
  )
}

export default App
