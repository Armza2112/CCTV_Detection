from flask import Flask, send_from_directory, render_template_string
from flask_socketio import SocketIO
from config import CAPTURE_DIR, INTERVAL_MINUTES

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

web_state = {
    "latest_img": "",
    "last_time": "Waiting..."
}
@app.route("/")
def index():
    return render_template_string(""" 
<html>
        <head>
            <title>AI CCTV WebSocket</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
            <style>
                body { background: #121212; color: #e0e0e0; text-align: center; font-family: sans-serif; padding: 20px; }
                .card { background: #1e1e1e; max-width: 800px; margin: auto; padding: 20px; border-radius: 15px; border: 1px solid #333; }
                img { width: 100%; border-radius: 10px; border: 3px solid #4CAF50; margin-top: 15px; }
                .live-tag { color: #ff4444; font-weight: bold; animation: blink 1s infinite; }
                @keyframes blink { 50% { opacity: 0; } }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>AI Smart Snapshot <span class="live-tag">● LIVE</span></h1>
                <p>Update: <span id="last_time">{{ last_time }}</span> | Interval: {{ interval }} min</p>
                
                <img id="cctv_img" src="{{ '/download/' + img_name if img_name else '' }}" 
                     style="{{ 'display:block' if img_name else 'display:none' }}">
                
                <div id="wait_msg" style="{{ 'display:none' if img_name else 'display:block; padding:50px;' }}">
                    Waiting for AI processing...
                </div>
            </div>

            <script>
                var socket = io();
                socket.on('new_detection', function(data) {
                    var img = document.getElementById('cctv_img');
                    var msg = document.getElementById('wait_msg');
                    img.src = "/download/" + data.img_name + "?t=" + new Date().getTime();
                    img.style.display = "block";
                    msg.style.display = "none";
                    document.getElementById('last_time').innerHTML = data.time;
                });
            </script>
        </body>
    </html>
    """, img_name=web_state["latest_img"],
    last_time=web_state["last_time"],
    interval=INTERVAL_MINUTES)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(CAPTURE_DIR, filename)