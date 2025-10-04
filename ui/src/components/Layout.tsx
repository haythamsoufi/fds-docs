import { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import {
  Home,
  FileText,
  Search,
  BarChart3,
  Settings,
  Menu,
  X,
  Activity
} from 'lucide-react'
import SystemStatus from './SystemStatus'
import IFRCLogo from './IFRCLogo'
import IFRCFooter from './IFRCFooter'

interface LayoutProps {
  children: React.ReactNode
}

const Layout = ({ children }: LayoutProps) => {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const location = useLocation()

  const navigation = [
    { name: 'Dashboard', href: '/', icon: Home },
    { name: 'Documents', href: '/documents', icon: FileText },
    { name: 'Query', href: '/query', icon: Search },
    { name: 'Analytics', href: '/analytics', icon: BarChart3 },
    { name: 'Monitoring', href: '/monitoring', icon: Activity },
    { name: 'Settings', href: '/settings', icon: Settings },
  ]

  const isActive = (href: string) => {
    return location.pathname === href
  }

  return (
    <div className="min-h-screen bg-ifrc-white">
      {/* Mobile sidebar */}
      <div className={`fixed inset-0 z-50 lg:hidden ${sidebarOpen ? 'block' : 'hidden'}`}>
        <div className="fixed inset-0 bg-ifrc-navy-900 bg-opacity-50" onClick={() => setSidebarOpen(false)} />
        <div className="fixed inset-y-0 left-0 w-64 bg-ifrc-white shadow-xl">
          <div className="flex items-center justify-between h-16 px-4 border-b border-ifrc-navy-200">
            <IFRCLogo size="md" />
            <button
              onClick={() => setSidebarOpen(false)}
              className="p-2 rounded-md text-ifrc-navy-400 hover:text-ifrc-navy-600"
            >
              <X className="h-6 w-6" />
            </button>
          </div>
          <nav className="mt-4 px-4">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`nav-item ${
                    isActive(item.href)
                      ? 'nav-item-active'
                      : 'nav-item-inactive'
                  }`}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              )
            })}
          </nav>
        </div>
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:flex lg:w-64 lg:flex-col overflow-visible">
        <div className="flex flex-col flex-grow bg-ifrc-white border-r border-ifrc-navy-200 overflow-visible">
          <div className="flex items-center h-16 px-4 border-b border-ifrc-navy-200">
            <IFRCLogo size="md" />
          </div>
          <nav className="mt-4 flex-1 px-4 space-y-1">
            {navigation.map((item) => {
              const Icon = item.icon
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  className={`nav-item ${
                    isActive(item.href)
                      ? 'nav-item-active'
                      : 'nav-item-inactive'
                  }`}
                >
                  <Icon className="mr-3 h-5 w-5" />
                  {item.name}
                </Link>
              )
            })}
          </nav>
          
          {/* System status */}
          <SystemStatus />
        </div>
      </div>

      {/* Main content */}
      <div className="lg:pl-64">
        {/* Top bar */}
        <div className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-ifrc-navy-200 bg-ifrc-white px-4 shadow-sm sm:gap-x-6 sm:px-6 lg:px-8">
          <button
            type="button"
            className="-m-2.5 p-2.5 text-ifrc-navy-700 lg:hidden"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu className="h-6 w-6" />
          </button>
          <div className="flex flex-1 gap-x-4 self-stretch lg:gap-x-6">
            <div className="flex flex-1"></div>
            <div className="flex items-center gap-x-4 lg:gap-x-6">
              <div className="hidden lg:block lg:h-6 lg:w-px lg:bg-ifrc-navy-200" />
            </div>
          </div>
        </div>

        {/* Page content */}
        <main className="py-6 min-h-screen">
          <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
            {children}
          </div>
        </main>
        
        {/* Footer */}
        <IFRCFooter />
      </div>
    </div>
  )
}

export default Layout
