# Methodology Page Guide

The Methodology page provides an integrated documentation viewer within the UI that allows users to read and browse all system documentation directly from the web interface.

## Features

### 📚 **Documentation Browser**
- **Organized Categories**: Documents are grouped by category (Processing, Setup, Configuration, Architecture, Operations)
- **Expandable Sections**: Collapsible navigation for easy browsing
- **Search & Filter**: Find documents by title, description, or filename

### 🎨 **Rich Markdown Rendering**
- **Full Markdown Support**: Headers, lists, code blocks, tables, links, and more
- **GitHub Flavored Markdown**: Support for tables, strikethrough, task lists
- **Custom Styling**: Consistent with IFRC design system
- **Syntax Highlighting**: Code blocks with proper formatting

### 📱 **Responsive Design**
- **Mobile Friendly**: Works on all screen sizes
- **Sidebar Navigation**: Easy document switching
- **Quick Actions**: Download and external links

### 🔧 **API Integration**
- **RESTful API**: Clean endpoints for documentation management
- **File Serving**: Direct access to markdown files
- **Health Monitoring**: Service status and availability checks

## Usage

### Accessing Documentation

1. **Navigate to Methodology**: Click "Methodology" in the sidebar navigation
2. **Browse Categories**: Expand categories to see available documents
3. **Select Document**: Click on any document to view its contents
4. **Quick Actions**: Use sidebar buttons for downloads and external links

### Available Documentation

The following documents are available through the Methodology page:

#### **Processing**
- **Multimodal Processing Guide**: Complete guide to table and chart extraction from PDFs

#### **Setup & Configuration**
- **Migration Guide**: How to migrate from previous versions
- **OCR Replacement Guide**: Upgrading OCR capabilities and configuration

#### **Architecture**
- **RAG Upgrade Plan**: Retrieval-Augmented Generation system improvements

#### **Operations**
- **Rollback Procedures**: How to rollback changes if needed

#### **Advanced Features**
- **Citation and Numeric Answers**: Complete guide to citation system and numeric answer capabilities

## API Endpoints

The Methodology page uses the following API endpoints:

### **Documentation Management**
```http
GET /api/v1/docs/                    # List all documentation files
GET /api/v1/docs/{filename}          # Download a documentation file
GET /api/v1/docs/{filename}/content  # Get file content as text
GET /api/v1/docs/categories/list     # List documentation categories
GET /api/v1/docs/health              # Check service health
```

### **Example Usage**
```bash
# List all documentation
curl http://localhost:8080/api/v1/docs/

# Get a specific file
curl http://localhost:8080/api/v1/docs/MULTIMODAL_PROCESSING_GUIDE.md

# Check service health
curl http://localhost:8080/api/v1/docs/health
```

## Technical Implementation

### **Frontend Components**
- **Methodology.tsx**: Main page component with navigation and content display
- **documentationService.ts**: Service for API communication
- **ReactMarkdown**: Library for markdown rendering with custom components

### **Backend API**
- **docs.py**: FastAPI router for documentation endpoints
- **File Serving**: Secure file access with path validation
- **Metadata Management**: Document categorization and descriptions

### **Security Features**
- **Path Validation**: Prevents directory traversal attacks
- **File Type Restrictions**: Only markdown files are served
- **Input Sanitization**: Safe filename handling

## Customization

### **Adding New Documentation**

1. **Add Markdown File**: Place your `.md` file in the `docs/` directory
2. **Update Metadata**: Add file information to the documentation service
3. **Categorize**: Assign appropriate category for organization

### **Customizing Styles**

The markdown rendering uses custom components that can be modified in `Methodology.tsx`:

```typescript
const markdownComponents = {
  h1: ({ children }) => (
    <h1 className="text-3xl font-bold text-secondary-900 mb-6">
      {children}
    </h1>
  ),
  // ... other components
}
```

### **Adding New Categories**

Update the categories array in the Methodology component:

```typescript
const categories = [
  {
    id: 'new-category',
    name: 'New Category',
    description: 'Description of the new category',
    icon: NewIcon
  }
]
```

## Dependencies

### **Frontend Dependencies**
```json
{
  "react-markdown": "^9.0.1",
  "remark-gfm": "^4.0.0"
}
```

### **Backend Dependencies**
- FastAPI (already included)
- Path validation utilities
- File system access

## Troubleshooting

### **Common Issues**

#### **Documents Not Loading**
- Check if the `docs/` directory exists
- Verify file permissions
- Check API endpoint health: `/api/v1/docs/health`

#### **Markdown Not Rendering**
- Ensure markdown files are valid
- Check for special characters that might break parsing
- Verify ReactMarkdown dependencies are installed

#### **Navigation Issues**
- Clear browser cache
- Check for JavaScript errors in console
- Verify route configuration in App.tsx

### **Debug Mode**

Enable debug logging for the documentation service:

```typescript
// In documentationService.ts
console.log('Loading documentation:', filename)
console.log('API response:', response.data)
```

## Future Enhancements

### **Planned Features**
- **Search Functionality**: Full-text search across all documentation
- **Bookmarking**: Save favorite documents
- **Print Support**: Clean printing of documentation
- **PDF Export**: Convert markdown to PDF
- **Version History**: Track documentation changes
- **Comments**: Add notes to documentation sections

### **Integration Opportunities**
- **Help System**: Context-sensitive help integration
- **Tutorial Mode**: Guided tours using documentation
- **API Documentation**: Auto-generated API docs
- **User Guides**: Step-by-step user tutorials

## Support

For issues with the Methodology page:

1. Check the browser console for errors
2. Verify API endpoints are accessible
3. Test with different markdown files
4. Check file permissions and paths
5. Review the documentation service logs

The Methodology page provides a comprehensive way to access and browse system documentation, making it easier for users to understand how the RAG system works and how to use its features effectively.
