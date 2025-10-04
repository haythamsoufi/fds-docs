# FDS Docs - React UI

Modern React-based user interface for the FDS Docs Enterprise RAG System.

## 🚀 Quick Start

1. **Install dependencies**:
   ```bash
   npm install
   ```

2. **Start development server**:
   ```bash
   npm run dev
   ```

3. **Build for production**:
   ```bash
   npm run build
   ```

## 📁 Project Structure

```
ui/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/         # Page components
│   ├── services/      # API service layer
│   ├── App.tsx        # Main application
│   ├── main.tsx       # Entry point
│   └── index.css      # Global styles
├── package.json       # Dependencies
├── vite.config.ts     # Vite configuration
└── tailwind.config.js # Tailwind CSS config
```

## 🎯 Features

- **Dashboard**: System overview and metrics
- **Document Management**: Upload, view, and manage documents
- **Query Interface**: Advanced search with multiple search types
- **Analytics**: Performance monitoring and usage statistics
- **Settings**: System configuration and management

## 🔧 Configuration

Set the API URL in your environment:

```env
VITE_API_URL=http://localhost:8000
```

## 🚀 Deployment

### Development
```bash
npm run dev
```

### Production
```bash
npm run build
npm run preview
```

### Docker
```bash
docker build -f Dockerfile.ui -t fds-docs-ui .
```
