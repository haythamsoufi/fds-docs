import IFRCLogo from './IFRCLogo'

interface IFRCHeaderProps {
  title: string
  subtitle?: string
  theme?: 'default' | 'climate' | 'values' | 'health' | 'migration' | 'disaster'
  className?: string
}

const IFRCHeader = ({ 
  title, 
  subtitle, 
  theme = 'default',
  className = '' 
}: IFRCHeaderProps) => {
  const themeClasses = {
    default: 'bg-ifrc-white border-ifrc-navy-200',
    climate: 'theme-climate',
    values: 'theme-values',
    health: 'theme-health',
    migration: 'theme-migration',
    disaster: 'theme-disaster'
  }

  return (
    <div className={`${themeClasses[theme]} border-b ${className}`}>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between">
          <div className="flex-1">
            <h1 className="heading-primary text-3xl font-bold text-ifrc-navy-900">
              {title}
            </h1>
            {subtitle && (
              <p className="mt-2 text-lg text-muted">
                {subtitle}
              </p>
            )}
          </div>
          
          {/* IFRC Logo */}
          <div className="hidden md:block">
            <IFRCLogo size="lg" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default IFRCHeader
