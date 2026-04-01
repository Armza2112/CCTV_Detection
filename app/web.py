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
        <title>AI CCTV Smart Dashboard</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script>
        <style>
            body { background: #0f0f0f; color: #e0e0e0; text-align: center; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; }
            .container { max-width: 1100px; margin: auto; }
            .header { margin-bottom: 30px; border-bottom: 2px solid #333; padding-bottom: 10px; }
            .grid { display: flex; gap: 20px; justify-content: center; flex-wrap: wrap; }
            .card { background: #1e1e1e; flex: 1; min-width: 450px; padding: 15px; border-radius: 12px; border: 1px solid #333; box-shadow: 0 4px 15px rgba(0,0,0,0.5); }
            h2 { font-size: 1.2rem; color: #bbb; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px; }
            img { width: 100%; border-radius: 8px; border: 2px solid #444; background: #000; min-height: 250px; }
            .status-bar { background: #252525; padding: 10px; border-radius: 8px; margin-bottom: 20px; display: inline-block; padding: 10px 30px; }
            .highlight { color: #4CAF50; font-weight: bold; }
            @keyframes blink { 50% { opacity: 0; } }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Vision AI Processing</h1>
                <div class="status-bar">
                    Last Sync: <span id="last_time" class="highlight">{{ last_time }}</span> | 
                    Scan every: <span class="highlight">{{ interval }} min</span>
                </div>
            </div>

            <div class="grid">
                <div class="card">
                    <h2>Step 1: Motion Processing (Binary)</h2>
                    <img id="proc_img" src="/download/motion_debug.jpg">
                </div>

                <div class="card">
                    <h2>Step 2: AI Object Detection</h2>
                    <img id="cctv_img" src="{{ '/download/' + img_name if img_name else '' }}">
                </div>
            </div>

            <div id="wait_msg" style="{{ 'display:none' if img_name else 'display:block; padding:50px; color:#666;' }}">
                <p>Initializing camera stream and waiting for motion...</p>
            </div>
        </div>

        <script>
            var socket = io();

            socket.on('update_motion', function(data) {
                var imgProc = document.getElementById('proc_img');
                var timestamp = new Date().getTime();
                imgProc.src = "/download/motion_debug.jpg?t=" + timestamp;
                document.getElementById('last_time').innerHTML = data.time;
            });

            socket.on('new_detection', function(data) {
                var imgAi = document.getElementById('cctv_img');
                var imgProc = document.getElementById('proc_img');
                var msg = document.getElementById('wait_msg');
                var timestamp = new Date().getTime();

                imgAi.src = "/download/" + data.img_name + "?t=" + timestamp;
                imgProc.src = "/download/motion_debug.jpg?t=" + timestamp; // อัปเดตฝั่งซ้ายด้วย
                
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