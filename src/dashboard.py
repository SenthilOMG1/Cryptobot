"""
Web Dashboard - Remote Monitoring
=================================
Simple web interface to monitor your bot from anywhere.
Protected with password authentication.
"""

import os
import threading
from functools import wraps
from flask import Flask, jsonify, request, Response
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global references to bot components (set by main.py)
_components = {}


def set_components(components: dict):
    """Set references to bot components for dashboard access."""
    global _components
    _components = components


def check_auth(username, password):
    """Check if username/password is valid."""
    correct_user = os.environ.get("DASHBOARD_USER", "admin")
    correct_pass = os.environ.get("DASHBOARD_PASS", "changeme123")
    return username == correct_user and password == correct_pass


def requires_auth(f):
    """Decorator for password protection."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return Response(
                'Login required. Use your dashboard username/password.',
                401,
                {'WWW-Authenticate': 'Basic realm="Trading Bot Dashboard"'}
            )
        return f(*args, **kwargs)
    return decorated


@app.route("/")
@requires_auth
def home():
    """Dashboard home page."""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Trading Bot</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: #0a0a0a;
                color: #fff;
                padding: 20px;
                min-height: 100vh;
            }
            .container { max-width: 800px; margin: 0 auto; }
            h1 {
                color: #00ff88;
                margin-bottom: 20px;
                font-size: 24px;
            }
            .card {
                background: #1a1a1a;
                border-radius: 12px;
                padding: 20px;
                margin-bottom: 20px;
                border: 1px solid #333;
            }
            .card h2 {
                color: #888;
                font-size: 14px;
                text-transform: uppercase;
                margin-bottom: 10px;
            }
            .value {
                font-size: 32px;
                font-weight: bold;
                color: #00ff88;
            }
            .value.negative { color: #ff4444; }
            .positions { margin-top: 15px; }
            .position {
                display: flex;
                justify-content: space-between;
                padding: 10px 0;
                border-bottom: 1px solid #333;
            }
            .position:last-child { border-bottom: none; }
            .pair { font-weight: bold; }
            .pnl { font-weight: bold; }
            .pnl.positive { color: #00ff88; }
            .pnl.negative { color: #ff4444; }
            .status {
                display: inline-block;
                padding: 5px 10px;
                border-radius: 20px;
                font-size: 12px;
                font-weight: bold;
            }
            .status.running { background: #00ff8833; color: #00ff88; }
            .status.paused { background: #ff444433; color: #ff4444; }
            .refresh-btn {
                background: #00ff88;
                color: #000;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-weight: bold;
                margin-top: 10px;
            }
            .trades { font-size: 14px; }
            .trade {
                padding: 8px 0;
                border-bottom: 1px solid #222;
            }
            #data { min-height: 200px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🤖 AI Trading Bot</h1>
            <div id="data">Loading...</div>
            <button class="refresh-btn" onclick="loadData()">Refresh</button>
        </div>
        <script>
            async function loadData() {
                try {
                    const res = await fetch('/api/status');
                    const data = await res.json();
                    document.getElementById('data').innerHTML = renderDashboard(data);
                } catch (e) {
                    document.getElementById('data').innerHTML = '<p>Error loading data</p>';
                }
            }

            function renderDashboard(data) {
                const pnlClass = data.daily_pnl >= 0 ? '' : 'negative';
                const statusClass = data.paused ? 'paused' : 'running';
                const statusText = data.paused ? 'PAUSED' : 'RUNNING';

                let positionsHtml = '';
                if (data.positions && data.positions.length > 0) {
                    positionsHtml = data.positions.map(p => {
                        const pnlClass = p.pnl_pct >= 0 ? 'positive' : 'negative';
                        return `
                            <div class="position">
                                <span class="pair">${p.pair}</span>
                                <span class="pnl ${pnlClass}">${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(2)}%</span>
                            </div>
                        `;
                    }).join('');
                } else {
                    positionsHtml = '<p style="color:#666">No open positions</p>';
                }

                return `
                    <div class="card">
                        <h2>Portfolio Value</h2>
                        <div class="value">$${data.total_value.toFixed(2)}</div>
                        <p style="color:#888;margin-top:5px">USDT Balance: $${data.usdt_balance.toFixed(2)}</p>
                    </div>

                    <div class="card">
                        <h2>Today's P&L</h2>
                        <div class="value ${pnlClass}">${data.daily_pnl >= 0 ? '+' : ''}$${data.daily_pnl.toFixed(2)} (${data.daily_pnl_pct >= 0 ? '+' : ''}${data.daily_pnl_pct.toFixed(2)}%)</div>
                    </div>

                    <div class="card">
                        <h2>Status</h2>
                        <span class="status ${statusClass}">${statusText}</span>
                        <p style="color:#888;margin-top:10px">Uptime: ${data.uptime}</p>
                    </div>

                    <div class="card">
                        <h2>Open Positions (${data.positions ? data.positions.length : 0})</h2>
                        <div class="positions">${positionsHtml}</div>
                    </div>

                    <div class="card">
                        <h2>Recent Trades</h2>
                        <div class="trades">
                            ${data.recent_trades && data.recent_trades.length > 0
                                ? data.recent_trades.map(t => `<div class="trade">${t}</div>`).join('')
                                : '<p style="color:#666">No trades yet</p>'
                            }
                        </div>
                    </div>
                `;
            }

            loadData();
            setInterval(loadData, 30000); // Auto-refresh every 30 seconds
        </script>
    </body>
    </html>
    """


@app.route("/api/status")
@requires_auth
def api_status():
    """API endpoint for bot status."""
    try:
        # Get portfolio summary
        positions_tracker = _components.get("positions")
        risk_manager = _components.get("risk")
        health_monitor = _components.get("health")
        executor = _components.get("executor")

        # Default values if components not ready
        portfolio = {"total_value": 0, "usdt_balance": 0, "positions": []}
        risk_status = {"paused": False, "daily_pnl": 0, "daily_pnl_pct": 0}
        uptime = "Starting..."
        recent_trades = []

        if positions_tracker:
            portfolio = positions_tracker.get_portfolio_summary()

        if risk_manager:
            risk_status = risk_manager.get_risk_status()

        if health_monitor:
            uptime = health_monitor.get_uptime()

        if executor:
            trades = executor.get_trade_history(limit=5)
            recent_trades = [
                f"{t[1][:10]} | {t[3].upper()} {t[2]} | ${t[6]:.2f}"
                for t in trades
            ] if trades else []

        return jsonify({
            "total_value": portfolio.get("total_value", 0),
            "usdt_balance": portfolio.get("usdt_balance", 0),
            "positions": portfolio.get("positions", []),
            "daily_pnl": risk_status.get("daily_pnl", 0),
            "daily_pnl_pct": risk_status.get("daily_pnl_pct", 0),
            "paused": risk_status.get("paused", False),
            "uptime": uptime,
            "recent_trades": recent_trades
        })

    except Exception as e:
        logger.error(f"Dashboard API error: {e}")
        return jsonify({"error": str(e)}), 500


def run_dashboard(port: int = 5000):
    """Run the dashboard in a background thread."""
    def start():
        app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)

    thread = threading.Thread(target=start, daemon=True)
    thread.start()
    logger.info(f"Dashboard running on http://0.0.0.0:{port}")
    return thread
