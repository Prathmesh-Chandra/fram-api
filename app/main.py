from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import data

app = FastAPI(title="FRAM API")

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

@app.get("/health")
def health():
    return {"status": "ok"}