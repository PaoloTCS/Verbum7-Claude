// /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/frontend/src/components/DomainForm.js
import React, { useState } from 'react';
import '../styles/DomainForm.css';

/**
 * DomainForm component for adding new domains
 */
const DomainForm = ({ onAdd }) => {
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [isFormVisible, setIsFormVisible] = useState(false);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    if (!name.trim()) return;
    
    onAdd(name.trim(), description.trim());
    setName('');
    setDescription('');
    setIsFormVisible(false);
  };

  return (
    <div className="domain-form-container">
      {!isFormVisible ? (
        <button 
          className="add-domain-button"
          onClick={() => setIsFormVisible(true)}
        >
          + Add New Domain
        </button>
      ) : (
        <form onSubmit={handleSubmit} className="domain-form">
          <div className="form-group">
            <label htmlFor="domain-name" className="form-label">Domain Name</label>
            <input
              id="domain-name"
              type="text"
              className="form-input"
              placeholder="Enter a domain name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </div>
          
          <div className="form-group">
            <label htmlFor="domain-description" className="form-label">Description (optional)</label>
            <textarea
              id="domain-description"
              className="form-input form-textarea"
              placeholder="Enter a description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={3}
            />
          </div>
          
          <div className="form-buttons">
            <button type="submit" className="form-button">
              Add Domain
            </button>
            <button 
              type="button" 
              className="form-button secondary"
              onClick={() => setIsFormVisible(false)}
            >
              Cancel
            </button>
          </div>
        </form>
      )}
    </div>
  );
};

export default DomainForm;