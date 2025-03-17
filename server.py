from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import httpx
import os
from dotenv import load_dotenv
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="GitHub MCP Server")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_API_BASE = "https://api.github.com"

class Issue(BaseModel):
    title: str
    body: str
    labels: Optional[List[str]] = None

class Comment(BaseModel):
    body: str

async def get_github_headers():
    if not GITHUB_TOKEN:
        raise HTTPException(
            status_code=500,
            detail="GitHub token not configured. Please check your .env file."
        )
    return {
        "Authorization": f"token {GITHUB_TOKEN}",  # Changed from "Bearer" to "token"
        "Accept": "application/vnd.github.v3+json"
    }

# Create a base HTTP client with timeout settings
async def get_client():
    return httpx.AsyncClient(timeout=30.0)

@app.get("/repository/{owner}/{repo}")
async def get_repository_details(owner: str, repo: str, headers: dict = Depends(get_github_headers)):
    """Fetch repository details including stars, forks, and contributors."""
    async with await get_client() as client:
        try:
            repo_response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}",
                headers=headers
            )
            
            if repo_response.status_code == 401:
                logger.error("Authentication failed. Please check your GitHub token.")
                raise HTTPException(
                    status_code=401,
                    detail="GitHub authentication failed. Please check your token."
                )
            elif repo_response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {owner}/{repo} not found"
                )
                
            repo_response.raise_for_status()
            
            contributors_response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contributors",
                headers=headers
            )
            contributors_response.raise_for_status()

            repo_data = repo_response.json()
            return {
                "stars": repo_data["stargazers_count"],
                "forks": repo_data["forks_count"],
                "contributors": len(contributors_response.json())
            }
        except httpx.HTTPError as e:
            logger.error(f"GitHub API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch repository details")

@app.get("/repository/{owner}/{repo}/issues")
async def list_issues(owner: str, repo: str, headers: dict = Depends(get_github_headers)):
    """List open issues in a repository."""
    async with await get_client() as client:
        try:
            response = await client.get(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
                headers=headers
            )
            
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your GitHub token.")
                raise HTTPException(
                    status_code=401,
                    detail="GitHub authentication failed. Please check your token."
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {owner}/{repo} not found"
                )
                
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"GitHub API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to fetch issues")

@app.post("/repository/{owner}/{repo}/issues")
async def create_issue(
    owner: str, 
    repo: str, 
    issue: Issue,
    headers: dict = Depends(get_github_headers)
):
    """Create a new issue in a repository."""
    async with await get_client() as client:
        try:
            response = await client.post(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues",
                headers=headers,
                json=issue.dict()
            )
            
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your GitHub token.")
                raise HTTPException(
                    status_code=401,
                    detail="GitHub authentication failed. Please check your token."
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {owner}/{repo} not found"
                )
                
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"GitHub API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create issue")

@app.post("/repository/{owner}/{repo}/issues/{issue_number}/comments")
async def create_comment(
    owner: str,
    repo: str,
    issue_number: int,
    comment: Comment,
    headers: dict = Depends(get_github_headers)
):
    """Post a comment on an issue or pull request."""
    async with await get_client() as client:
        try:
            response = await client.post(
                f"{GITHUB_API_BASE}/repos/{owner}/{repo}/issues/{issue_number}/comments",
                headers=headers,
                json=comment.dict()
            )
            
            if response.status_code == 401:
                logger.error("Authentication failed. Please check your GitHub token.")
                raise HTTPException(
                    status_code=401,
                    detail="GitHub authentication failed. Please check your token."
                )
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=404,
                    detail=f"Repository {owner}/{repo} not found"
                )
                
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            logger.error(f"GitHub API error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to create comment")
