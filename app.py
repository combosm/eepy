from __future__ import annotations

import atexit
import os

from flask import Flask, render_template, Response, jsonify
from vision.camera import (
    data_store,
    generate_frames,
    socketio,
    start_monitoring,
    stop_monitoring,
)

app = Flask(__name__)
socketio.init_app(app, cors_allowed_origins="*")
atexit.register(stop_monitoring)


@app.before_request
def ensure_monitoring_started() -> None:
    """Start once in the process serving requests, independent of video clients."""
    start_monitoring()


# route for video streaming
@app.route('/video_feed')
def video_feed() -> Response:
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_output')
def ai_output() -> Response:
    """Run the optional assistant without making it a startup dependency."""
    try:
        # These imports intentionally happen only for an assistant request.
        # Missing AI/audio dependencies must not prevent local monitoring.
        from ai.ai_agent import answer_once
        from ai.stt_tts import record_audio

        query = record_audio()
        if not query:
            return jsonify({"message": "Sorry, I didn't catch that."})

        answer = answer_once(query)
    except Exception:
        app.logger.exception("Optional AI assistant request failed")
        return jsonify(
            {
                "message": (
                    "The optional AI assistant is currently unavailable. "
                    "Local fatigue monitoring is still running."
                )
            }
        ), 503

    data_store["ai_response"] = answer
    socketio.emit("update_data", dict(data_store))
    return jsonify({"message": answer})


@app.route('/data')
def get_data() -> Response:
    return jsonify(data_store)

@app.route('/')
def index() -> str:
    return render_template('index.html')

@app.route('/home')
def home() -> str:
    return render_template('home.html')

if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    # Werkzeug's debug parent must not compete with its serving child for camera 0.
    if not debug_enabled or os.getenv("WERKZEUG_RUN_MAIN") == "true":
        start_monitoring()
    socketio.run(app, host="0.0.0.0", port=5001, debug=debug_enabled)
