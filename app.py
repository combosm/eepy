from flask import Flask, render_template, Response, jsonify
from vision.camera import generate_frames, data_store, socketio
import os
from ai.ai_agent import answer_once
from ai.stt_tts import record_audio

app = Flask(__name__)
socketio.init_app(app, cors_allowed_origins="*")


# route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_output')
def ai_output():
    # one listen-and-answer turn
    query = record_audio()
    if not query:
        return jsonify({"message": "Sorry, I didn't catch that."})

    answer = answer_once(query)
    data_store["ai_response"] = answer
    socketio.emit("update_data", dict(data_store))
    return jsonify({"message": answer})


@app.route('/data')
def get_data():
    return jsonify(data_store)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    debug_enabled = os.getenv("FLASK_DEBUG", "0").lower() in ("1", "true", "yes", "on")
    socketio.run(app, host="0.0.0.0", port=5001, debug=debug_enabled)