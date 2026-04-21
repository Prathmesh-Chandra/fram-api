import os

from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data
from app.scheduler import start_scheduler, stop_scheduler
from app.routers.analytics import router as analytics_router
from app.routers import pricing, hedging, risk


def _build_allowed_origins() -> list[str]:
    origins = {
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://fram-frontend.vercel.app",
    }

    frontend_url = os.getenv("FRONTEND_URL")
    if frontend_url:
        origins.add(frontend_url.rstrip("/"))

    extra_origins = os.getenv("CORS_ORIGINS", "")
    for origin in extra_origins.split(","):
        cleaned = origin.strip().rstrip("/")
        if cleaned:
            origins.add(cleaned)

    return sorted(origins)

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(title="FRAM API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_build_allowed_origins(),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router, prefix="/data")
app.include_router(analytics_router)
app.include_router(pricing.router)
app.include_router(hedging.router)
app.include_router(risk.router)

@app.get("/health")
def health():
    return {"status": "ok"}