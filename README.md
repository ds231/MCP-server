# GitHub MCP Server

A FastAPI server implementing the Model Context Protocol for GitHub interactions.

## Setup

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate 
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your GitHub Personal Access Token:
   ```
   GITHUB_TOKEN=your_token_here
   ```

### Usage

Start the server:
```bash
uvicorn server:app --reload --port 8000
```

## API Endpoints

### Get Repository Details
```http
GET /repository/{owner}/{repo}
```

### List Issues
```http
GET /repository/{owner}/{repo}/issues
```

### Create Issue
```http
POST /repository/{owner}/{repo}/issues
```

### Create Comment
```http
POST /repository/{owner}/{repo}/issues/{issue_number}/comments
```

## Error Handling

The server implements comprehensive error handling for all GitHub API interactions. Errors are logged and appropriate HTTP status codes are returned.

## Type Hints

The codebase uses type hints throughout for better maintainability and IDE support.
