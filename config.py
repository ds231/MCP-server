from pydantic import BaseSettings, validator
from typing import Optional

class Settings(BaseSettings):
    # GitHub settings
    github_token: str
    github_api_base: str = "https://api.github.com"
    
    # Client settings
    request_timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 100
    keepalive_connections: int = 20
    
    # Cache settings
    repo_cache_ttl: int = 300
    issues_cache_ttl: int = 60
    cache_max_size: int = 100
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Logging
    log_level: str = "INFO"

    @validator('github_token')
    def validate_token(cls, v):
        if not v:
            raise ValueError('GitHub token is required')
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
