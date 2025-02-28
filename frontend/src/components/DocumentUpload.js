// /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/frontend/src/components/DocumentUpload.js
import React, { useState } from 'react';
import { uploadDocument } from '../services/apiService';
import '../styles/DocumentUpload.css';

/**
 * DocumentUpload component for uploading files to a domain
 */
const DocumentUpload = ({ domainId, onUploadComplete }) => {
  const [file, setFile] = useState(null);
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [isFormVisible, setIsFormVisible] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file) {
      setError('Please select a file');
      return;
    }
    
    setLoading(true);
    setError(null);
    
    try {
      await uploadDocument(domainId, file, description);
      setFile(null);
      setDescription('');
      setIsFormVisible(false);
      
      // Notify parent component that upload is complete
      if (onUploadComplete) {
        onUploadComplete();
      }
    } catch (err) {
      console.error('Error uploading document:', err);
      setError('Failed to upload document. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Show/hide upload form
  const toggleForm = () => {
    setIsFormVisible(!isFormVisible);
    setError(null);
  };

  return (
    <div className="document-upload-container">
      {!isFormVisible ? (
        <button 
          className="upload-button"
          onClick={toggleForm}
        >
          + Upload Document
        </button>
      ) : (
        <form onSubmit={handleSubmit} className="upload-form">
          <div className="form-group">
            <label htmlFor="file-input" className="form-label">Select Document</label>
            <input
              id="file-input"
              type="file"
              className="file-input"
              onChange={handleFileChange}
              accept="application/pdf,.pdf,.doc,.docx,.txt"
            />
            {file && (
              <div className="selected-file">
                Selected: {file.name}
              </div>
            )}
          </div>
          
          <div className="form-group">
            <label htmlFor="doc-description" className="form-label">Description (optional)</label>
            <textarea
              id="doc-description"
              className="form-input form-textarea"
              placeholder="Enter a description for this document"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
          
          {error && (
            <div className="upload-error">
              {error}
            </div>
          )}
          
          <div className="form-buttons">
            <button 
              type="submit" 
              className="form-button"
              disabled={loading || !file}
            >
              {loading ? 'Uploading...' : 'Upload'}
            </button>
            <button 
              type="button" 
              className="form-button secondary"
              onClick={toggleForm}
              disabled={loading}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

export default DocumentUpload;