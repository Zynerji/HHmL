"""
Live Training Dashboard - Modular Real-Time Monitoring
=======================================================
Provides a live HTML dashboard for monitoring training in real-time.

Usage in any training script:
    from hhml.utils.live_dashboard import TrainingDashboard

    # Initialize dashboard
    dashboard = TrainingDashboard(port=8000)
    dashboard.start()

    # During training loop:
    dashboard.update({
        'cycle': cycle,
        'density': vortex_density,
        'quality': vortex_quality,
        'reward': reward,
        'annihilations': num_removed
    })

    # After training:
    dashboard.stop()

Features:
- Real-time chart updates using Chart.js
- Server-Sent Events (SSE) for streaming
- Thread-safe, non-blocking
- Zero external dependencies (uses stdlib only)
- Auto-opens browser window

Author: HHmL Framework
Date: 2025-12-17
"""

import threading
import json
import queue
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime
import webbrowser
from pathlib import Path


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for dashboard requests"""

    # Class variable to share data between handler instances
    data_queue = queue.Queue()
    latest_metrics = {}

    def log_message(self, format, *args):
        """Suppress request logging"""
        pass  # Silent

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            # Serve main dashboard HTML
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self._get_dashboard_html().encode())

        elif self.path == '/stream':
            # Server-Sent Events stream
            self.send_response(200)
            self.send_header('Content-type', 'text/event-stream')
            self.send_header('Cache-Control', 'no-cache')
            self.send_header('Connection', 'keep-alive')
            self.end_headers()

            try:
                while True:
                    # Block until new data available
                    data = self.data_queue.get(timeout=30)

                    # Send as SSE
                    message = f"data: {json.dumps(data)}\\n\\n"
                    self.wfile.write(message.encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, queue.Empty):
                # Client disconnected or timeout
                pass

        elif self.path == '/api/latest':
            # REST endpoint for latest metrics
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(self.latest_metrics).encode())

        else:
            self.send_error(404)

    def _get_dashboard_html(self):
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HHmL Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0e27;
            color: #e0e0e0;
            padding: 20px;
        }
        .header {
            text-align: center;
            padding: 20px 0;
            border-bottom: 2px solid #1e2749;
        }
        h1 {
            color: #4fc3f7;
            font-size: 2em;
            margin-bottom: 10px;
        }
        .subtitle {
            color: #888;
            font-size: 0.9em;
        }
        .status {
            display: inline-block;
            margin: 10px 0;
            padding: 5px 15px;
            background: #1e8449;
            border-radius: 15px;
            font-size: 0.85em;
        }
        .status.waiting { background: #f39c12; }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        .stat-card {
            background: #151d3b;
            border: 1px solid #1e2749;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
        }
        .stat-label {
            color: #888;
            font-size: 0.85em;
            margin-bottom: 10px;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #4fc3f7;
        }
        .stat-value.density { color: #4fc3f7; }
        .stat-value.quality { color: #81c784; }
        .stat-value.reward { color: #ffb74d; }
        .stat-value.annihilations { color: #e57373; }
        .charts-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .chart-container {
            background: #151d3b;
            border: 1px solid #1e2749;
            border-radius: 10px;
            padding: 20px;
        }
        .chart-title {
            font-size: 1.1em;
            margin-bottom: 15px;
            color: #4fc3f7;
        }
        canvas {
            max-height: 300px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>HHmL Training Dashboard</h1>
        <p class="subtitle">Real-Time Monitoring</p>
        <div class="status" id="status">Waiting for data...</div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Cycle</div>
            <div class="stat-value" id="stat-cycle">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Vortex Density</div>
            <div class="stat-value density" id="stat-density">--%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Vortex Quality</div>
            <div class="stat-value quality" id="stat-quality">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Reward</div>
            <div class="stat-value reward" id="stat-reward">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Annihilations</div>
            <div class="stat-value annihilations" id="stat-annihilations">--</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Cycles at Target</div>
            <div class="stat-value" id="stat-stable">--</div>
        </div>
    </div>

    <div class="charts-grid">
        <div class="chart-container">
            <div class="chart-title">Vortex Density & Quality</div>
            <canvas id="densityChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Reward Over Time</div>
            <canvas id="rewardChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Annihilation Events</div>
            <canvas id="annihilationChart"></canvas>
        </div>
        <div class="chart-container">
            <div class="chart-title">Stability (Cycles at Target)</div>
            <canvas id="stabilityChart"></canvas>
        </div>
    </div>

    <script>
        // Chart configurations
        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            plugins: {
                legend: { labels: { color: '#e0e0e0' } }
            },
            scales: {
                x: { ticks: { color: '#888' }, grid: { color: '#1e2749' } },
                y: { ticks: { color: '#888' }, grid: { color: '#1e2749' } }
            }
        };

        // Initialize charts
        const densityChart = new Chart(document.getElementById('densityChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Density',
                        data: [],
                        borderColor: '#4fc3f7',
                        backgroundColor: 'rgba(79, 195, 247, 0.1)',
                        tension: 0.4
                    },
                    {
                        label: 'Quality',
                        data: [],
                        borderColor: '#81c784',
                        backgroundColor: 'rgba(129, 199, 132, 0.1)',
                        tension: 0.4
                    }
                ]
            },
            options: chartOptions
        });

        const rewardChart = new Chart(document.getElementById('rewardChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Reward',
                    data: [],
                    borderColor: '#ffb74d',
                    backgroundColor: 'rgba(255, 183, 77, 0.1)',
                    tension: 0.4
                }]
            },
            options: chartOptions
        });

        const annihilationChart = new Chart(document.getElementById('annihilationChart'), {
            type: 'bar',
            data: {
                labels: [],
                datasets: [{
                    label: 'Removed Vortices',
                    data: [],
                    backgroundColor: '#e57373'
                }]
            },
            options: chartOptions
        });

        const stabilityChart = new Chart(document.getElementById('stabilityChart'), {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Cycles at Target',
                    data: [],
                    borderColor: '#9575cd',
                    backgroundColor: 'rgba(149, 117, 205, 0.1)',
                    fill: true
                }]
            },
            options: chartOptions
        });

        // Data buffers (keep last 100 points)
        const maxPoints = 100;

        // Connect to SSE stream
        const eventSource = new EventSource('/stream');
        const statusEl = document.getElementById('status');

        eventSource.onmessage = function(event) {
            const data = JSON.parse(event.data);

            // Update status
            statusEl.textContent = 'Live';
            statusEl.className = 'status';

            // Update stat cards
            document.getElementById('stat-cycle').textContent = data.cycle || '--';
            document.getElementById('stat-density').textContent =
                data.density !== undefined ? (data.density * 100).toFixed(1) + '%' : '--%';
            document.getElementById('stat-quality').textContent =
                data.quality !== undefined ? data.quality.toFixed(2) : '--';
            document.getElementById('stat-reward').textContent =
                data.reward !== undefined ? data.reward.toFixed(1) : '--';
            document.getElementById('stat-annihilations').textContent = data.annihilations || '--';
            document.getElementById('stat-stable').textContent = data.cycles_at_target || '--';

            // Update charts
            const cycle = data.cycle;

            // Add to density chart
            if (densityChart.data.labels.length >= maxPoints) {
                densityChart.data.labels.shift();
                densityChart.data.datasets[0].data.shift();
                densityChart.data.datasets[1].data.shift();
            }
            densityChart.data.labels.push(cycle);
            densityChart.data.datasets[0].data.push(data.density);
            densityChart.data.datasets[1].data.push(data.quality);
            densityChart.update();

            // Add to reward chart
            if (rewardChart.data.labels.length >= maxPoints) {
                rewardChart.data.labels.shift();
                rewardChart.data.datasets[0].data.shift();
            }
            rewardChart.data.labels.push(cycle);
            rewardChart.data.datasets[0].data.push(data.reward);
            rewardChart.update();

            // Add to annihilation chart
            if (annihilationChart.data.labels.length >= maxPoints) {
                annihilationChart.data.labels.shift();
                annihilationChart.data.datasets[0].data.shift();
            }
            annihilationChart.data.labels.push(cycle);
            annihilationChart.data.datasets[0].data.push(data.annihilations || 0);
            annihilationChart.update();

            // Add to stability chart
            if (stabilityChart.data.labels.length >= maxPoints) {
                stabilityChart.data.labels.shift();
                stabilityChart.data.datasets[0].data.shift();
            }
            stabilityChart.data.labels.push(cycle);
            stabilityChart.data.datasets[0].data.push(data.cycles_at_target || 0);
            stabilityChart.update();
        };

        eventSource.onerror = function() {
            statusEl.textContent = 'Disconnected';
            statusEl.className = 'status waiting';
        };
    </script>
</body>
</html>
        """


class TrainingDashboard:
    """
    Modular live dashboard for training monitoring

    Usage:
        dashboard = TrainingDashboard(port=8000)
        dashboard.start()

        # In training loop:
        dashboard.update({
            'cycle': cycle,
            'density': density,
            'quality': quality,
            'reward': reward,
            'annihilations': num_removed,
            'cycles_at_target': cycles_stable
        })

        dashboard.stop()
    """

    def __init__(self, port=8000, auto_open=True):
        self.port = port
        self.auto_open = auto_open
        self.server = None
        self.server_thread = None
        self.running = False

    def start(self):
        """Start the dashboard server"""
        if self.running:
            print("[Dashboard] Already running")
            return

        # Create server
        self.server = HTTPServer(('localhost', self.port), DashboardHandler)
        self.running = True

        # Start server in background thread
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()

        url = f"http://localhost:{self.port}"
        print(f"[Dashboard] Live dashboard at: {url}")

        # Auto-open browser
        if self.auto_open:
            threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    def _run_server(self):
        """Background server loop"""
        try:
            self.server.serve_forever()
        except Exception as e:
            print(f"[Dashboard] Server error: {e}")

    def update(self, metrics):
        """
        Update dashboard with new metrics

        Args:
            metrics: Dictionary with keys:
                - cycle: Current cycle number
                - density: Vortex density (0-1)
                - quality: Vortex quality (0-1)
                - reward: Current reward
                - annihilations: Number of vortices removed
                - cycles_at_target: Consecutive cycles at target density
        """
        if not self.running:
            return

        # Add timestamp
        metrics['timestamp'] = datetime.now().isoformat()

        # Update latest metrics (for REST API)
        DashboardHandler.latest_metrics = metrics

        # Put in queue for SSE stream
        DashboardHandler.data_queue.put(metrics)

    def stop(self):
        """Stop the dashboard server"""
        if not self.running:
            return

        print("[Dashboard] Shutting down...")
        self.running = False

        if self.server:
            self.server.shutdown()
            self.server.server_close()

        if self.server_thread:
            self.server_thread.join(timeout=2)

        print("[Dashboard] Stopped")


# Example usage
if __name__ == "__main__":
    import time
    import random

    dashboard = TrainingDashboard(port=8000)
    dashboard.start()

    print("Simulating training for 60 seconds...")
    print("Open http://localhost:8000 in your browser")

    try:
        for cycle in range(100):
            # Simulate training metrics
            dashboard.update({
                'cycle': cycle,
                'density': 0.5 + 0.3 * random.random(),
                'quality': 0.3 + 0.2 * random.random(),
                'reward': 20 + 30 * random.random(),
                'annihilations': random.randint(0, 50),
                'cycles_at_target': random.randint(0, 10)
            })
            time.sleep(0.6)
    finally:
        dashboard.stop()
