
interface IFRCLogoProps {
  className?: string
  size?: 'sm' | 'md' | 'lg'
  showText?: boolean
}

const IFRCLogo = ({ className = '', size = 'md', showText = true }: IFRCLogoProps) => {
  const sizeClasses = {
    sm: 'h-6 w-6',
    md: 'h-8 w-8',
    lg: 'h-12 w-12'
  }

  const textSizeClasses = {
    sm: 'text-sm',
    md: 'text-lg',
    lg: 'text-2xl'
  }

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      {/* IFRC Red Cross Symbol */}
      <div className={`${sizeClasses[size]} flex items-center justify-center bg-ifrc-red-500 rounded-sm`}>
        <svg
          className="h-3/4 w-3/4 text-ifrc-white"
          viewBox="0 0 24 24"
          fill="currentColor"
        >
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z"/>
        </svg>
      </div>
      
      {/* IFRC Text */}
      {showText && (
        <div className="flex flex-col">
          <span className={`font-bold text-ifrc-navy-900 ${textSizeClasses[size]}`}>
            IFRC
          </span>
          <span className="text-xs text-ifrc-navy-600 -mt-1">
            Docs
          </span>
        </div>
      )}
    </div>
  )
}

export default IFRCLogo
