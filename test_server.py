import httpx
import pytest
from fastapi.testclient import TestClient
from .server import app

client = TestClient(app)

def test_get_repository_details():
    """Test repository details endpoint"""
    response = client.get("/repository/microsoft/vscode")
    assert response.status_code == 200
    data = response.json()
    assert "stars" in data
    assert "forks" in data
    assert "contributors" in data
    assert "cached_at" in data

def test_list_issues():
    """Test listing issues endpoint"""
    response = client.get("/repository/microsoft/vscode/issues")
    assert response.status_code == 200
    issues = response.json()
    assert isinstance(issues, list)

def test_create_issue():
    """Test creating an issue"""
    issue_data = {
        "title": "Test Issue",
        "body": "This is a test issue created by automated testing",
        "labels": ["test"]
    }
    response = client.post("/repository/your-username/your-repo/issues", json=issue_data)
    assert response.status_code in [201, 403]  # 403 if no write permissions

def test_invalid_repository():
    """Test invalid repository name"""
    response = client.get("/repository/invalid/invalid")
    assert response.status_code == 404

def test_rate_limiting():
    """Test rate limiting"""
    for _ in range(101):  # Exceed rate limit
        client.get("/repository/microsoft/vscode")
    response = client.get("/repository/microsoft/vscode")
    assert response.status_code == 429
