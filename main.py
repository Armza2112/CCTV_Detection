from app.web import app, socketio, web_state 
from ai.model_loader import load_model
from services.snapshot_service import snapshot_job
from config import (PORT)

if __name__ == "__main__":
    model = load_model()
    socketio.start_background_task(
        snapshot_job, model, socketio, web_state
    )

    print(f"Web Server running on http://0.0.0.0:{PORT}")
    socketio.run(app, host="0.0.0.0", port=PORT, use_reloader=False)