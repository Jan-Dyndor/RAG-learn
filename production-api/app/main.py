import time
from contextlib import asynccontextmanager
from logging import Logger

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langsmith import traceable
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from app.agent import ProductionAgent
from app.cache import ResponseCache
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
)
from app.security import SecurityPipeline

logger = Logger(name="prod_test")

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize all components on startup, clean up on shutdown.
    This is the modern FastAPI pattern
    """

    global security, cache, agent
    settings = get_settings()
    security = SecurityPipeline()
    cache = ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    agent = ProductionAgent()
    logger.info("App up")
    yield
    logger.info("App down")


# ! Rate Limiting
limiter = Limiter(key_func=get_remote_address)

#! FastAPI app
app = FastAPI(lifespan=lifespan, title="Production LangGraph API")
app.state.limiter = limiter


#! Exception Handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(content={"error": "too many requests"}, status_code=429)


#! ENDPOINTS
@app.post("/chat", response_model=ChatResponse)
@limiter.limit(get_settings().rate_limit)
@traceable(name="chat_endpoint")
async def chat(request: Request, body: ChatRequest):
    """Main chat endpoint


    Flow:
    1. Security check ( injection + PII masking)
    2. Cache lookup
    3. LangGraph agent invoke ( if cache miss)
    4. Output validation
    5. Cache store
    6. Return response
    """
    security_notes = []

    is_allowed, cleaned_message, notes = security.check_input(body.message)

    if not is_allowed:
        logger.warning("Requests blocked by security ")
        raise HTTPException(
            status_code=400, detail="Your message was blocked by security"
        )

    # Cache
    cached_response = cache.get(cleaned_message)
    if cached_response is not None:
        logger.info("Chace hit")

        return ChatResponse(
            response=cached_response,
            thread_id=body.thread_id,
            model_used="cache",
            cached=True,
            processing_time_ms=0,
        )

    # Step 3 Invoke LangGraph Agent
    try:
        result = agent.invoke(cleaned_message)
    except Exception as ex:
        logger.error("Error at agent invoke {e}")
        raise HTTPException(status_code=500, detail="An error occured")

    response_text = result["response"]
    model_used = result["model_used"]
    # Step 4 Output validaion
    validated_response, output_warnings = security.check_output(response_text)
    security_notes.extend(output_warnings)

    # Step 5 Cache Store
    cache.set(cleaned_message, validated_response)

    return ChatResponse(
        response=validated_response,
        thread_id=body.thread_id,
        model_used=model_used,
        cached=False,
        processing_time_ms=1,  # I do not count it
    )


# @app.get("/metrics", response_model=MetricsResponse)
# async def get_metrics():
#     """Metrics for monitoring dashboards."""
#     summary = metrics.summary
#     return MetricsResponse(**summary)


@app.get("/cache/stats")
async def cache_stats():
    """Cache performance statistics."""
    return cache.stats


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check for Docker/Kubernetes."""
    settings = get_settings()

    checks = {
        "agent": agent is not None,
        "security": security is not None,
        "cache": cache is not None,
    }

    all_healthy = all(checks.values())

    return HealthResponse(
        status="healthy" if all_healthy else "degraded",
        environment=settings.app_env,
        checks=checks,
    )
