
const IFRCFooter = () => {
  return (
    <footer className="bg-ifrc-navy-800 text-ifrc-white py-6 mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* IFRC Branding */}
          <div className="space-y-4">
            <div className="flex items-center space-x-2">
              <div className="h-8 w-8 bg-ifrc-red-500 rounded-sm flex items-center justify-center">
                <svg
                  className="h-5 w-5 text-ifrc-white"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                >
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
                </svg>
              </div>
              <div>
                <h3 className="text-lg font-bold">IFRC</h3>
                <p className="text-sm text-ifrc-navy-300">International Federation of Red Cross and Red Crescent Societies</p>
              </div>
            </div>
          </div>

          {/* Quick Links */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-ifrc-navy-200 uppercase tracking-wider">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#" className="text-ifrc-navy-300 hover:text-ifrc-white transition-colors">
                  About IFRC
                </a>
              </li>
              <li>
                <a href="#" className="text-ifrc-navy-300 hover:text-ifrc-white transition-colors">
                  Our Work
                </a>
              </li>
              <li>
                <a href="#" className="text-ifrc-navy-300 hover:text-ifrc-white transition-colors">
                  Get Help
                </a>
              </li>
              <li>
                <a href="#" className="text-ifrc-navy-300 hover:text-ifrc-white transition-colors">
                  Contact
                </a>
              </li>
            </ul>
          </div>

          {/* System Info */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-ifrc-navy-200 uppercase tracking-wider">System</h4>
            <div className="text-sm text-ifrc-navy-300">
              <p>FDS Documentation System</p>
              <p className="mt-1">Powered by IFRC Technology</p>
              <p className="mt-2 text-xs">
                © {new Date().getFullYear()} IFRC. All rights reserved.
              </p>
            </div>
          </div>
        </div>

        {/* Bottom Bar */}
        <div className="mt-8 pt-6 border-t border-ifrc-navy-700">
          <div className="flex flex-col md:flex-row justify-between items-center space-y-2 md:space-y-0">
            <p className="text-xs text-ifrc-navy-400">
              This system is designed to support humanitarian operations and disaster response.
            </p>
            <div className="flex space-x-4 text-xs">
              <a href="#" className="text-ifrc-navy-400 hover:text-ifrc-white transition-colors">
                Privacy Policy
              </a>
              <a href="#" className="text-ifrc-navy-400 hover:text-ifrc-white transition-colors">
                Terms of Service
              </a>
            </div>
          </div>
        </div>
      </div>
    </footer>
  )
}

export default IFRCFooter
