// /Users/paolopignatelli/VerbumTechnologies/Verbum7-Claude/frontend/src/components/BreadcrumbNav.js
import React from 'react';
import '../styles/BreadcrumbNav.css';

/**
 * BreadcrumbNav component for navigation through the domain hierarchy
 */
const BreadcrumbNav = ({ path, onNavigate }) => {
  return (
    <div className="breadcrumb">
      <span 
        className={`breadcrumb-item ${path.length === 0 ? 'active' : ''}`}
        onClick={() => onNavigate(null)}
      >
        Home
      </span>
      
      {path.map((domain, index) => (
        <React.Fragment key={domain.id}>
          <span className="breadcrumb-separator">/</span>
          <span 
            className={`breadcrumb-item ${index === path.length - 1 ? 'active' : ''}`}
            onClick={() => index !== path.length - 1 ? onNavigate(domain.id) : null}
          >
            {domain.name}
          </span>
        </React.Fragment>
      ))}
    </div>
  );
};

export default BreadcrumbNav;