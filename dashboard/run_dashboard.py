#!/usr/bin/env python3
"""
Evaluator Dashboard Server
==========================
Usage:
    python3 run_dashboard.py
"""

import os
import sys
import http.server
import socketserver
import webbrowser

PORT = 8080
DIRECTORY = os.path.dirname(os.path.abspath(__file__))

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

if __name__ == "__main__":
    print("=" * 60)
    print(f"  🚀 Launching Evaluator UI Dashboard on http://localhost:{PORT}")
    print("=" * 60)
    url = f"http://localhost:{PORT}"
    try:
        webbrowser.open(url)
    except:
        pass
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard stopped.")
