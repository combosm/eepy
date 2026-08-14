from flask import Flask, render_template, Response, jsonify
from scripts.camera import generate_frames, data_store, socketio
import time
import threading
from scripts.ai.ai_agent import main

app = Flask(__name__)
socketio.init_app(app, cors_allowed_origins="*")

data_store = {"EAR": 0, "MAR": 0, "is_drowsy": False, "ai_response": "TEST"}

# Route for video streaming
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ai_output')
def ai_output():
    ai_response = main()
    data_store["ai_response"] = ai_response
    socketio.emit("update_data", data_store)
    return jsonify({"message": ai_response})

# Route to get data
@app.route('/data')
def get_data():
    return jsonify(data_store)

def generate_data():
    while True:
        data = {
            "EAR": f"{data_store['EAR']:.3f}",
            "MAR": f"{data_store['MAR']:.3f}",
            "is_drowsy": data_store["is_drowsy"],
            "ai_response": data_store["ai_response"]
        }
        socketio.emit("update_data", data)
        time.sleep(1)

thread = threading.Thread(target=generate_data)
thread.daemon = True
thread.start()

# Route for index page
@app.route('/')
def index():
    return render_template('index.html')

# Route for home page
@app.route('/home')
def home():
    return render_template('home.html')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5001, debug=True)