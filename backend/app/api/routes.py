# /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/backend/app/api/routes.py
"""
app/api/routes.py
API routes for the Semantic Tiles application.
"""

import os
import uuid
from datetime import datetime
from flask import Blueprint, jsonify, request, current_app, send_file
from werkzeug.utils import secure_filename
from app.core.domain_model import DomainStore
from app.core.semantic_processor import SemanticProcessor

# Create blueprint
api_bp = Blueprint('api', __name__)

# Setup domain store
domain_store = None
semantic_processor = None

def get_domain_store():
    """Get or initialize domain store."""
    global domain_store
    if (domain_store is None):
        storage_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], 'data')
        os.makedirs(storage_dir, exist_ok=True)
        domain_store = DomainStore(storage_dir)
    return domain_store

def get_semantic_processor():
    """Get or initialize semantic processor."""
    global semantic_processor
    if (semantic_processor is None):
        semantic_processor = SemanticProcessor(current_app.config['UPLOAD_FOLDER'])
    return semantic_processor

@api_bp.route('/domains', methods=['GET'])
def get_domains():
    """Get domains at a specific level."""
    try:
        parent_id = request.args.get('parentId')
        
        # Get domains
        domain_store = get_domain_store()
        domains = domain_store.get_domains(parent_id)
        
        # Compute semantic distances if needed
        if domains and len(domains) > 1:
            semantic_processor = get_semantic_processor()
            distances = semantic_processor.compute_distances(domains)
            
            # Format distances for output
            formatted_distances = {
                f"{k[0]}|{k[1]}": v for k, v in distances.items()
            }
            
            return jsonify({
                "domains": domains,
                "semanticDistances": formatted_distances
            })
        
        return jsonify({"domains": domains, "semanticDistances": {}})
        
    except Exception as e:
        current_app.logger.error(f"Error getting domains: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['GET'])
def get_domain(domain_id):
    """Get a single domain by ID."""
    try:
        domain_store = get_domain_store()
        domain = domain_store.get_domain(domain_id)
        
        if domain:
            return jsonify(domain)
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error getting domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains', methods=['POST'])
def add_domain():
    """Add a new domain."""
    try:
        data = request.json
        name = data.get('name')
        parent_id = data.get('parentId')
        description = data.get('description', '')
        
        if not name:
            return jsonify({"error": "Name is required"}), 400
        
        domain_store = get_domain_store()
        domain = domain_store.add_domain(name, parent_id, description)
        
        if domain:
            return jsonify(domain)
        else:
            return jsonify({"error": "Failed to add domain"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error adding domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['PUT'])
def update_domain(domain_id):
    """Update a domain."""
    try:
        data = request.json
        updates = {}
        
        # Only allow specific fields to be updated
        allowed_fields = ['name', 'description', 'x', 'y']
        for field in allowed_fields:
            if field in data:
                updates[field] = data[field]
        
        domain_store = get_domain_store()
        domain = domain_store.update_domain(domain_id, updates)
        
        if domain:
            return jsonify(domain)
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error updating domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>', methods=['DELETE'])
def delete_domain(domain_id):
    """Delete a domain and its children."""
    try:
        domain_store = get_domain_store()
        success = domain_store.delete_domain(domain_id)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Domain not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error deleting domain: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/path', methods=['GET'])
def get_domain_path(domain_id):
    """Get the path from root to a domain."""
    try:
        domain_store = get_domain_store()
        path = domain_store.get_domain_path(domain_id)
        
        return jsonify({"path": path})
            
    except Exception as e:
        current_app.logger.error(f"Error getting domain path: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/documents', methods=['POST'])
def add_document(domain_id):
    """Upload and add a document to a domain."""
    try:
        # domain_id = request.form.get('domainId')
        description = request.form.get('description', '')
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not domain_id:
            return jsonify({"error": "Domain ID is required"}), 400
            
        # Save the file
        filename = secure_filename(file.filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Create a unique filename
        unique_filename = f"{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], 'documents', unique_filename)
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the file
        file.save(file_path)
        
        # Add document to domain
        domain_store = get_domain_store()
        document = {
            "name": filename,
            "path": os.path.join('documents', unique_filename),
            "type": file.content_type,
            "dateAdded": datetime.now().isoformat(),
            "description": description
        }
        
        result = domain_store.add_document(domain_id, document)
        
        if result:
            return jsonify(result)
        else:
            # Clean up file if document not added
            os.remove(file_path)
            return jsonify({"error": "Failed to add document"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error adding document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/<domain_id>/documents/<document_id>', methods=['DELETE'])
def remove_document(domain_id, document_id):
    """Remove a document from a domain."""
    try:
        domain_store = get_domain_store()
        
        # Get the document info first so we can delete the file
        domain = domain_store.get_domain(domain_id)
        document = None
        
        if domain and "documents" in domain:
            for doc in domain["documents"]:
                if doc.get("id") == document_id:
                    document = doc
                    break
        
        success = domain_store.remove_document(domain_id, document_id)
        
        if success and document:
            # Delete the file if it exists
            if "path" in document:
                file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], document["path"])
                if os.path.exists(file_path):
                    os.remove(file_path)
            
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Document not found"}), 404
            
    except Exception as e:
        current_app.logger.error(f"Error removing document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>', methods=['GET'])
def get_document(document_path):
    """Serve a document file."""
    try:
        full_path = os.path.join(current_app.config['UPLOAD_FOLDER'], document_path)
        
        if not os.path.exists(full_path):
            return jsonify({"error": "Document not found"}), 404
        
        # Check if the file is a PDF and set the correct mimetype
        if document_path.lower().endswith('.pdf'):
            return send_file(
                full_path,
                as_attachment=False,
                download_name=os.path.basename(document_path),
                mimetype='application/pdf'  # Explicitly set PDF mimetype
            )
        
        # For other file types, let Flask guess the mimetype
        return send_file(
            full_path,
            as_attachment=False,
            download_name=os.path.basename(document_path)
        )
            
    except Exception as e:
        current_app.logger.error(f"Error serving document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>/summary', methods=['GET'])
def get_document_summary(document_path):
    """Get a summary of a document."""
    try:
        semantic_processor = get_semantic_processor()
        summary = semantic_processor.get_document_summary(document_path)
        
        return jsonify({"summary": summary})
            
    except Exception as e:
        current_app.logger.error(f"Error getting document summary: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/documents/<path:document_path>/query', methods=['POST'])
def query_document(document_path):
    """Process a query about a document."""
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
            
        semantic_processor = get_semantic_processor()
        response = semantic_processor.process_document_query(document_path, query)
        
        return jsonify({"response": response})
            
    except Exception as e:
        current_app.logger.error(f"Error querying document: {str(e)}")
        return jsonify({"error": str(e)}), 500

@api_bp.route('/domains/positions', methods=['POST'])
def update_domain_positions():
    """Update domain positions."""
    try:
        data = request.json
        positions = data.get('positions', {})
        
        domain_store = get_domain_store()
        success = domain_store.update_domain_positions(positions)
        
        if success:
            return jsonify({"success": True})
        else:
            return jsonify({"error": "Failed to update positions"}), 500
            
    except Exception as e:
        current_app.logger.error(f"Error updating domain positions: {str(e)}")
        return jsonify({"error": str(e)}), 500