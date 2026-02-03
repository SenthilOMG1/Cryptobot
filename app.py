"""
Simple entry point for Railway deployment.
Starts web server first, then trading bot in background.
"""
import os
import threading
import time
from flask import Flask, jsonify, Response, request
from functools import wraps

app = Flask(__name__)

# Bot status (updated by trading thread)
status = {
    "state": "starting",
    "message": "Initializing...",
    "portfolio": 0.0
}

def check_auth(username, password):
    u = os.environ.get("DASHBOARD_USER", "admin")
    p = os.environ.get("DASHBOARD_PASS", "password")
    return username == u and password == p

def auth_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response('Login required', 401, {'WWW-Authenticate': 'Basic realm="Bot"'})
        return f(*args, **kwargs)
    return decorated

@app.route("/")
@auth_required
def home():
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Bot</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ background: #0a0a0a; color: #00ff88; font-family: monospace; padding: 30px; }}
            h1 {{ color: #fff; }}
            .status {{ font-size: 28px; margin: 20px 0; }}
            .info {{ color: #888; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>AI Trading Bot</h1>
        <div class="status">Status: {status['state'].upper()}</div>
        <p class="info">{status['message']}</p>
        <p class="info">Portfolio: ${status['portfolio']:.2f} USDT</p>
        <p class="info">Page auto-refreshes every 10 seconds</p>
    </body>
    </html>
    """

@app.route("/health")
def health():
    return "OK", 200

@app.route("/api/status")
@auth_required
def api_status():
    return jsonify(status)

def run_trading_bot():
    """Run the trading bot in background."""
    global status
    time.sleep(5)  # Let web server stabilize

    try:
        status["state"] = "connecting"
        status["message"] = "Connecting to OKX..."

        # Import heavy modules here (after web server is up)
        from src.main import run_bot
        run_bot(status)

    except Exception as e:
        status["state"] = "error"
        status["message"] = f"Error: {str(e)}"

if __name__ == "__main__":
    # Start trading bot in background
    bot_thread = threading.Thread(target=run_trading_bot, daemon=True)
    bot_thread.start()

    # Run web server (this blocks)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
