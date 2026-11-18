#!/bin/bash
# ==============================================================================
# PathSnap - Laptop 1 (Model & FastAPI Inference Server)
# ==============================================================================
# Usage:
#   ./start_laptop1_api.sh
# ==============================================================================

cd "$(dirname "$0")"

echo "================================================================================"
echo "          🚀 PATHSNAP MAP-MATCHING INFERENCE SERVER (LAPTOP 1)"
echo "================================================================================"
echo " Detecting your Local Network IP for Laptop 2 connection..."

# Get Local IP on Mac or Linux
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

echo " 📍 Laptop 1 Local IP: http://${LOCAL_IP}:8000"
echo " 🌐 Accessible endpoints:"
echo "    - Root / Info: http://${LOCAL_IP}:8000/"
echo "    - Health:      http://${LOCAL_IP}:8000/health"
echo "    - Trip List:   http://${LOCAL_IP}:8000/trips"
echo "    - Predict:     http://${LOCAL_IP}:8000/predict/{trip_id}"
echo ""
echo " 💡 ON LAPTOP 2: Set the API Endpoint to: http://${LOCAL_IP}:8000"
echo "================================================================================"
echo " Starting FastAPI Server (Live Terminal Logs will stream below)..."
echo "================================================================================"

python3 -m uvicorn api.server:app --host 0.0.0.0 --port 8000
