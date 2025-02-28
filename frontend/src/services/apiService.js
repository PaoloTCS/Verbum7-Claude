// /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/frontend/src/services/apiService.js
import axios from 'axios';

// Base URL - use environment variable in production
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

// Create an axios instance with timeout
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
});

/**
 * Fetch domains at a specific level
 * @param {string|null} parentId - Parent domain ID or null for root level
 * @returns {Promise<Object>} - Domains and semantic distances
 */
export const fetchDomains = async (parentId = null) => {
  try {
    const url = parentId ? 
      `/domains?parentId=${parentId}` : 
      `/domains`;
      
    const response = await api.get(url);
    return response.data;
  } catch (error) {
    console.error('Error fetching domains:', error);
    throw error;
  }
};

/**
 * Fetch a single domain by ID
 * @param {string} domainId - Domain ID
 * @returns {Promise<Object>} - Domain data
 */
export const fetchDomain = async (domainId) => {
  try {
    const response = await api.get(`/domains/${domainId}`);
    return response.data;
  } catch (error) {
    console.error('Error fetching domain:', error);
    throw error;
  }
};

/**
 * Fetch the path to a domain
 * @param {string} domainId - Domain ID
 * @returns {Promise<Array>} - Path to the domain
 */
export const fetchDomainPath = async (domainId) => {
  try {
    const response = await api.get(`/domains/${domainId}/path`);
    return response.data.path;
  } catch (error) {
    console.error('Error fetching domain path:', error);
    throw error;
  }
};

/**
 * Add a new domain
 * @param {string} name - Domain name
 * @param {string|null} parentId - Parent domain ID or null for root level
 * @param {string} description - Domain description
 * @returns {Promise<Object>} - New domain data
 */
export const addDomain = async (name, parentId = null, description = '') => {
  try {
    const response = await api.post(`/domains`, {
      name,
      parentId,
      description
    });
    return response.data;
  } catch (error) {
    console.error('Error adding domain:', error);
    throw error;
  }
};

/**
 * Update a domain
 * @param {string} domainId - Domain ID
 * @param {Object} updates - Updates to apply
 * @returns {Promise<Object>} - Updated domain data
 */
export const updateDomain = async (domainId, updates) => {
  try {
    const response = await api.put(`/domains/${domainId}`, updates);
    return response.data;
  } catch (error) {
    console.error('Error updating domain:', error);
    throw error;
  }
};

/**
 * Delete a domain
 * @param {string} domainId - Domain ID
 * @returns {Promise<Object>} - Success status
 */
export const deleteDomain = async (domainId) => {
  try {
    const response = await api.delete(`/domains/${domainId}`);
    return response.data;
  } catch (error) {
    console.error('Error deleting domain:', error);
    throw error;
  }
};

/**
 * Update domain positions
 * @param {Object} positions - Dictionary of domain_id -> {x, y}
 * @returns {Promise<Object>} - Success status
 */
export const updateDomainPositions = async (positions) => {
  try {
    const response = await api.post(`/domains/positions`, {
      positions
    });
    return response.data;
  } catch (error) {
    console.error('Error updating domain positions:', error);
    throw error;
  }
};

/**
 * Upload a document to a domain
 * @param {string} domainId - Domain ID
 * @param {File} file - Document file
 * @param {string} description - Document description
 * @returns {Promise<Object>} - Document data
 */
export const uploadDocument = async (domainId, file, description = '') => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('domainId', domainId);
    formData.append('description', description);
    
    const response = await api.post(
      `/domains/${domainId}/documents`, 
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    console.error('Error uploading document:', error);
    throw error;
  }
};

/**
 * Remove a document from a domain
 * @param {string} domainId - Domain ID
 * @param {string} documentId - Document ID
 * @returns {Promise<Object>} - Success status
 */
export const removeDocument = async (domainId, documentId) => {
  try {
    const response = await api.delete(
      `/domains/${domainId}/documents/${documentId}`
    );
    return response.data;
  } catch (error) {
    console.error('Error removing document:', error);
    throw error;
  }
};

/**
 * Get a document summary
 * @param {string} documentPath - Document path
 * @returns {Promise<Object>} - Document summary
 */
export const getDocumentSummary = async (documentPath) => {
  try {
    const response = await api.get(
      `/documents/${encodeURIComponent(documentPath)}/summary`
    );
    return response.data;
  } catch (error) {
    console.error('Error getting document summary:', error);
    throw error;
  }
};

/**
 * Query a document
 * @param {string} documentPath - Document path
 * @param {string} query - Query text
 * @returns {Promise<Object>} - Query response
 */
export const queryDocument = async (documentPath, query) => {
  try {
    const response = await api.post(
      `/documents/${encodeURIComponent(documentPath)}/query`,
      { query }
    );
    return response.data;
  } catch (error) {
    console.error('Error querying document:', error);
    throw error;
  }
};

/**
 * Get document file URL
 * @param {string} documentPath - Document path
 * @returns {string} - Document URL
 */
export const getDocumentUrl = (documentPath) => {
  return `/api/documents/${encodeURIComponent(documentPath)}`;
};