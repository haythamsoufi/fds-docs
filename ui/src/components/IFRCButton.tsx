
interface IFRCButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'outline-red'
  size?: 'sm' | 'md' | 'lg'
  theme?: 'default' | 'climate' | 'values' | 'health' | 'migration' | 'disaster'
  children: React.ReactNode
}

const IFRCButton = ({ 
  variant = 'primary', 
  size = 'md', 
  theme = 'default',
  className = '',
  children,
  ...props 
}: IFRCButtonProps) => {
  const baseClasses = 'btn'
  
  const variantClasses = {
    primary: 'btn-primary',
    secondary: 'btn-secondary',
    outline: 'btn-outline',
    'outline-red': 'btn-outline-red'
  }
  
  const sizeClasses = {
    sm: 'px-3 py-1.5 text-xs',
    md: 'px-4 py-2 text-sm',
    lg: 'px-6 py-3 text-base'
  }
  
  const themeClasses = {
    default: '',
    climate: 'theme-climate',
    values: 'theme-values',
    health: 'theme-health',
    migration: 'theme-migration',
    disaster: 'theme-disaster'
  }

  return (
    <button
      className={`${baseClasses} ${variantClasses[variant]} ${sizeClasses[size]} ${themeClasses[theme]} ${className}`}
      {...props}
    >
      {children}
    </button>
  )
}

export default IFRCButton
