from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional

class Settings(BaseSettings):
    github_token: str = Field(..., description="GitHub API token")
    github_api_base: str = Field(
        default="https://api.github.com",
        description="GitHub API base URL"
    )
    
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

    @field_validator('github_token')
    def validate_token(cls, v):
        if not v:
            raise ValueError('GitHub token is required')
        return v

    model_config = SettingsConfigDict(
        env_file='.env',
        case_sensitive=False,
        env_file_encoding='utf-8'
    )

settings = Settings()
