# Verbum7-Claude Development Guide

## Build & Run Commands
- **Frontend:** `cd frontend && npm start` (runs on port 3000)
- **Backend:** `cd backend && python run.py` (runs on port 5001)
- **Install Backend Dependencies:** `pip install -r requirements.txt`
- **Install Frontend Dependencies:** `cd frontend && npm install`
- **Run Test:** `cd frontend && npm test` or `cd frontend && npm test -- -t "test-name"` for single test
- **Lint Frontend:** `cd frontend && npm run lint`

## Code Style Guidelines
### Python
- Type hints required for function parameters and return values
- Use docstrings with Args/Returns format for all functions
- Exception handling with specific error messages and proper logging
- PEP 8 naming: snake_case for functions/variables, PascalCase for classes
- Proper error handling with try/except and logging

### JavaScript/React
- ES6+ syntax with functional components
- Use proper state management and hooks
- Descriptive variable/function names in camelCase 
- Component files named with PascalCase
- Keep components small and focused on a single responsibility

### General
- Detailed exception/error handling with logging
- Use meaningful comments for complex logic
- Organize imports alphabetically
- 2-space indentation for JavaScript, 4-space for Python