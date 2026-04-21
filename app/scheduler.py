from apscheduler.schedulers.background import BackgroundScheduler
from app.utils.upstox_client import _token_state
import time
import logging

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler()

def check_token():
    expires_at = _token_state.get("expires_at", 0)
    time_left = expires_at - time.time()
    if time_left < 3600:
        logger.warning("Upstox token expiring soon. Visit /data/upstox/login to re-authenticate.")
        _token_state["access_token"] = None

def start_scheduler():
    scheduler.add_job(check_token, trigger="interval", hours=23, id="token_check", replace_existing=True)
    scheduler.start()

def stop_scheduler():
    scheduler.shutdown()