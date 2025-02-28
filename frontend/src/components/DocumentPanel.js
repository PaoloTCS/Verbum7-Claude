// /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/frontend/src/components/DocumentPanel.js
import React, { useState } from 'react';
import { getDocumentUrl, queryDocument } from '../services/apiService';
import '../styles/DocumentPanel.css';

/**
 * DocumentPanel component for displaying and interacting with documents
 */
const DocumentPanel = ({ document, onClose }) => {
  const [query, setQuery] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5001';

  // If no document is selected, don't render the panel
  if (!document) {
    return null;
  }

  // Get document URL for embedding
  const documentUrl = getDocumentUrl(document.path);

  // Handle document query submission
  const handleQuerySubmit = async (e) => {
    e.preventDefault();
    
    if (!query.trim()) return;
    
    setLoading(true);
    setError(null);
    
    try {
      const data = await queryDocument(document.path, query);
      setResponse(data.response);
    } catch (err) {
      console.error('Error querying document:', err);
      setError('Failed to process query. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="document-panel visible">
      <div className="document-toolbar">
        <h2 className="document-title">{document.name}</h2>
        <button className="close-btn" onClick={onClose}>&times;</button>
      </div>
      
      <div className="document-viewer">
        {document.type.includes('pdf') ? (
          <iframe 
            src={`${backendUrl}/api/documents/${encodeURIComponent(document.path)}`}
            title={document.name}
            width="100%"
            height="100%"
            type="application/pdf"
            frameBorder="0"
          />
        ) : (
          <div className="document-content">
            <p>This document cannot be previewed.</p>
            <a 
              href={documentUrl} 
              target="_blank" 
              rel="noopener noreferrer"
              className="download-link"
            >
              Download Document
            </a>
          </div>
        )}
      </div>
      
      <div className="document-query">
        <h3 className="query-title">Ask a question about this document</h3>
        <form onSubmit={handleQuerySubmit}>
          <input
            type="text"
            className="query-input"
            placeholder="e.g. What is the main topic?"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            disabled={loading}
          />
          <button 
            type="submit" 
            className="query-submit"
            disabled={loading}
          >
            {loading ? 'Loading...' : 'Ask'}
          </button>
        </form>
        
        {error && (
          <div className="query-error">
            {error}
          </div>
        )}
        
        {response && (
          <div className="query-response">
            <h4>Response:</h4>
            <p>{response}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DocumentPanel;