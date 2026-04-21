from dotenv import load_dotenv
load_dotenv()

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data
from app.scheduler import start_scheduler, stop_scheduler
from app.routers.analytics import router as analytics_router
from app.routers import pricing, hedging, risk

@asynccontextmanager
async def lifespan(app: FastAPI):
    start_scheduler()
    yield
    stop_scheduler()

app = FastAPI(title="FRAM API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://fram-frontend.vercel.app",
    ],
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