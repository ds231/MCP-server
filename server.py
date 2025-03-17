from fastapi import FastAPI, HTTPException, Depends, Request, Response, Path, Query
from pydantic import BaseModel, Field, ConfigDict
from typing import Annotated
import httpx
import os
from typing import List, Optional, Dict, Any, Tuple
import logging
import re
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import uuid
from tenacity import retry, stop_after_attempt, wait_exponential
from contextlib import asynccontextmanager
from fastapi.middleware.gzip import GZipMiddleware
from cachetools import TTLCache
import asyncio
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response as StarletteResponse
import time
import json
from config import settings

# hey this is a test change to check if git push works correctly!
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

# Initialize encryption key (store this securely!)
ENCRYPTION_KEY = Fernet.generate_key()
fernet = Fernet(ENCRYPTION_KEY)

# Rate limiting settings
RATE_LIMIT = settings.rate_limit_requests  # requests per minute
rate_limit_storage = {}

# Initialize cache
repo_cache = TTLCache(
    maxsize=settings.cache_max_size, 
    ttl=settings.repo_cache_ttl
)
issues_cache = TTLCache(
    maxsize=settings.cache_max_size, 
    ttl=settings.issues_cache_ttl
)

# Initialize metrics
request_counter = Counter(
    'github_api_requests_total',
    'Total GitHub API requests',
    ['method', 'endpoint', 'status']
)
request_duration = Histogram(
    'github_api_request_duration_seconds',
    'GitHub API request duration',
    ['method', 'endpoint']
)
cache_hits = Counter(
    'github_api_cache_hits_total',
    'Total cache hits',
    ['endpoint']
)
github_api_errors = Counter(
    'github_api_errors_total',
    'Total GitHub API errors',
    ['type']
)

class JSONLogFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
        }
        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        return json.dumps(log_entry)

# Update logging configuration
logger = logging.getLogger(__name__)
json_handler = logging.StreamHandler()
json_handler.setFormatter(JSONLogFormatter())
logger.handlers = [json_handler]

app = FastAPI(title="GitHub MCP Server")
app.add_middleware(GZipMiddleware, minimum_size=1000)

def validate_github_token(token: str) -> bool:
    """Validate GitHub token format."""
    # GitHub tokens are 40 chars for classic or start with ghp_ for fine-grained
    pattern = r'^(ghp_[A-Za-z0-9_]{36}|[A-Za-z0-9_]{40})$'
    return bool(re.match(pattern, token))

def encrypt_token(token: str) -> bytes:
    """Encrypt token for storage."""
    return fernet.encrypt(token.encode())

def decrypt_token(encrypted_token: bytes) -> str:
    """Decrypt stored token."""
    return fernet.decrypt(encrypted_token).decode()

async def check_rate_limit(request: Request):
    """Implement rate limiting."""
    client_ip = request.client.host
    current_time = datetime.now()
    
    if client_ip in rate_limit_storage:
        requests = rate_limit_storage[client_ip]["requests"]
        last_reset = rate_limit_storage[client_ip]["last_reset"]
        
        if current_time - last_reset > timedelta(minutes=1):
            rate_limit_storage[client_ip] = {
                "requests": 1,
                "last_reset": current_time
            }
        elif requests >= RATE_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        else:
            rate_limit_storage[client_ip]["requests"] += 1
    else:
        rate_limit_storage[client_ip] = {
            "requests": 1,
            "last_reset": current_time
        }

async def get_github_headers():
    """Get validated and secure GitHub headers."""
    if not settings.github_token:
        raise HTTPException(
            status_code=500,
            detail="GitHub token not configured"
        )
    
    return {
        "Authorization": f"token {settings.github_token}",
        "Accept": "application/vnd.github.v3+json"
    }

class GitHubClient:
    def __init__(self):
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout),
            limits=httpx.Limits(
                max_connections=settings.connection_pool_size,
                max_keepalive_connections=settings.keepalive_connections
            ),
            transport=httpx.AsyncHTTPTransport(retries=settings.max_retries)
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry_error_callback=lambda retry_state: retry_state.outcome.result()
    )
    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        response = await self.client.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    async def close(self):
        await self.client.aclose()

@asynccontextmanager
async def get_client():
    """Get a configured GitHub client with retry mechanism."""
    client = GitHubClient()
    try:
        yield client
    finally:
        await client.close()

# Add rate limiting middleware to all routes
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    await check_rate_limit(request)
    response = await call_next(request)
    return response

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    request_counter.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response

@app.get("/metrics")
async def metrics():
    return StarletteResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

async def log_request_details(request_id: str, owner: str, repo: str, extra: Dict[str, Any] = None) -> None:
    """Log structured request details"""
    log_data = {
        "request_id": request_id,
        "owner": owner,
        "repo": repo,
        "timestamp": datetime.now().isoformat(),
    }
    if extra:
        log_data.update(extra)
    logger.info(f"Request details: {log_data}")

class RepositoryParams(BaseModel):
    owner: str = Field(
        min_length=1,
        max_length=39,
        pattern=r"^[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9]$"
    )
    repo: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-zA-Z0-9_.-]+$"
    )

    model_config = ConfigDict(str_strip_whitespace=True)

    @classmethod
    def validate_name(cls, value: str) -> str:
        if value.lower() in ['none', 'undefined', 'null']:
            raise ValueError('Invalid name')
        return value

    @classmethod
    def validate_owner(cls, value: str) -> str:
        value = cls.validate_name(value)
        if '--' in value:
            raise ValueError('Name cannot contain consecutive hyphens')
        return value

class Issue(BaseModel):
    title: str = Field(min_length=1, max_length=256)
    body: str = Field(min_length=1, max_length=65536)
    labels: list[str] | None = Field(default=None, max_items=10)

    model_config = ConfigDict(str_strip_whitespace=True)

class Comment(BaseModel):
    body: str = Field(min_length=1, max_length=65536)
    
    model_config = ConfigDict(str_strip_whitespace=True)

async def batch_github_requests(
    client: GitHubClient,
    requests: List[Tuple[str, str, dict]],
) -> List[httpx.Response]:
    """Batch multiple GitHub API requests"""
    tasks = [
        client.request(method, url, **kwargs)
        for method, url, kwargs in requests
    ]
    return await asyncio.gather(*tasks, return_exceptions=True)

@app.get("/repository/{owner}/{repo}")
async def get_repository_details(
    request: Request,
    params: RepositoryParams = Depends(),
    headers: dict = Depends(get_github_headers),
    response: Response = None,
):
    """Fetch repository details with validation."""
    request_id = str(uuid.uuid4())
    logger.info("Starting repository details request", extra={
        'request_id': request_id,
        'owner': params.owner,
        'repo': params.repo
    })
    
    cache_key = f"{params.owner}/{params.repo}"
    if cache_key in repo_cache:
        cache_hits.labels(endpoint="/repository/{owner}/{repo}").inc()
        response.headers["X-Cache"] = "HIT"
        response.headers["X-Request-ID"] = request_id
        return repo_cache[cache_key]

    async with get_client() as client:
        try:
            # Batch requests
            requests = [
                ("GET", f"{settings.github_api_base}/repos/{params.owner}/{params.repo}", {"headers": headers}),
                ("GET", f"{settings.github_api_base}/repos/{params.owner}/{params.repo}/contributors", {"headers": headers})
            ]
            
            responses = await batch_github_requests(client, requests)
            
            # Check for errors in responses
            for resp in responses:
                if isinstance(resp, Exception):
                    raise resp
            
            repo_response, contributors_response = responses
            repo_data = repo_response.json()
            
            result = {
                "stars": repo_data["stargazers_count"],
                "forks": repo_data["forks_count"],
                "contributors": len(contributors_response.json()),
                "cached_at": datetime.now().isoformat()
            }
            
            repo_cache[cache_key] = result
            response.headers["X-Cache"] = "MISS"
            response.headers["X-Request-ID"] = request_id
            return result
            
        except httpx.TimeoutException:
            github_api_errors.labels(type="timeout").inc()
            logger.error("GitHub API timeout", extra={'request_id': request_id})
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "GitHub API timeout",
                    "request_id": request_id
                }
            )
        except httpx.RequestError as e:
            github_api_errors.labels(type="connection").inc()
            logger.error(f"Connection error", extra={
                'request_id': request_id,
                'error': str(e)
            })
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Unable to reach GitHub API",
                    "request_id": request_id,
                    "error": str(e)
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request_id}: GitHub API error: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": str(e),
                    "request_id": request_id,
                    "github_status": e.response.status_code
                }
            )
        except Exception as e:
            logger.error(f"Request {request_id}: Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Internal server error",
                    "request_id": request_id
                }
            )

@app.get("/repository/{owner}/{repo}/issues")
async def list_issues(
    request: Request,
    params: RepositoryParams = Depends(),
    headers: dict = Depends(get_github_headers)
):
    """List open issues with validation."""
    request_id = str(uuid.uuid4())
    await log_request_details(request_id, params.owner, params.repo)
    
    async with get_client() as client:
        try:
            response = await client.request(
                "GET",
                f"{settings.github_api_base}/repos/{params.owner}/{params.repo}/issues",
                headers=headers
            )
            
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            logger.error(f"Request {request_id}: Timeout while accessing GitHub API")
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "GitHub API timeout",
                    "request_id": request_id
                }
            )
        except httpx.RequestError as e:
            github_api_errors.labels(type="connection").inc()
            logger.error(f"Connection error", extra={
                'request_id': request_id,
                'error': str(e)
            })
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Unable to reach GitHub API",
                    "request_id": request_id,
                    "error": str(e)
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request_id}: GitHub API error: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": str(e),
                    "request_id": request_id,
                    "github_status": e.response.status_code
                }
            )
        except Exception as e:
            logger.error(f"Request {request_id}: Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Internal server error",
                    "request_id": request_id
                }
            )

@app.post("/repository/{owner}/{repo}/issues")
async def create_issue(
    request: Request,
    issue: Issue,
    params: RepositoryParams = Depends(),
    headers: dict = Depends(get_github_headers)
):
    """Create a new issue with validation."""
    request_id = str(uuid.uuid4())
    await log_request_details(request_id, params.owner, params.repo)
    
    async with get_client() as client:
        try:
            response = await client.request(
                "POST",
                f"{settings.github_api_base}/repos/{params.owner}/{params.repo}/issues",
                headers=headers,
                json=issue.dict()
            )
            
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            github_api_errors.labels(type="timeout").inc()
            logger.error("GitHub API timeout", extra={'request_id': request_id})
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "GitHub API timeout",
                    "request_id": request_id
                }
            )
        except httpx.RequestError as e:
            github_api_errors.labels(type="connection").inc()
            logger.error(f"Connection error", extra={
                'request_id': request_id,
                'error': str(e)
            })
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Unable to reach GitHub API",
                    "request_id": request_id,
                    "error": str(e)
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request_id}: GitHub API error: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": str(e),
                    "request_id": request_id,
                    "github_status": e.response.status_code
                }
            )
        except Exception as e:
            logger.error(f"Request {request_id}: Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Internal server error",
                    "request_id": request_id
                }
            )

@app.post("/repository/{owner}/{repo}/issues/{issue_number}/comments")
async def create_comment(
    request: Request,
    owner: Annotated[str, Path(...)],
    repo: Annotated[str, Path(...)],
    issue_number: Annotated[int, Path(ge=1)],
    comment: Comment,
    headers: dict = Depends(get_github_headers)
):
    """Post a comment with validation."""
    params = RepositoryParams(owner=owner, repo=repo)
    request_id = str(uuid.uuid4())
    await log_request_details(request_id, params.owner, params.repo)
    
    async with get_client() as client:
        try:
            response = await client.request(
                "POST",
                f"{settings.github_api_base}/repos/{params.owner}/{params.repo}/issues/{issue_number}/comments",
                headers=headers,
                json=comment.dict()
            )
            
            response.raise_for_status()
            return response.json()
        except httpx.TimeoutException:
            github_api_errors.labels(type="timeout").inc()
            logger.error("GitHub API timeout", extra={'request_id': request_id})
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "GitHub API timeout",
                    "request_id": request_id
                }
            )
        except httpx.RequestError as e:
            github_api_errors.labels(type="connection").inc()
            logger.error(f"Connection error", extra={
                'request_id': request_id,
                'error': str(e)
            })
            raise HTTPException(
                status_code=502,
                detail={
                    "message": "Unable to reach GitHub API",
                    "request_id": request_id,
                    "error": str(e)
                }
            )
        except httpx.HTTPStatusError as e:
            logger.error(f"Request {request_id}: GitHub API error: {str(e)}")
            raise HTTPException(
                status_code=e.response.status_code,
                detail={
                    "message": str(e),
                    "request_id": request_id,
                    "github_status": e.response.status_code
                }
            )
        except Exception as e:
            logger.error(f"Request {request_id}: Unexpected error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Internal server error",
                    "request_id": request_id
                }
            )
