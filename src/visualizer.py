"""
Interactive Map Matching Visualizer
===================================
Generates standalone interactive HTML/Leaflet map visualization comparing:
- Raw Noisy GPS Track Points (Red)
- Ground Truth Route (Green)
- Predicted HMM Route (Blue)
Includes statistics dashboard, metrics, and layer toggles.
"""

import json
import os
import numpy as np


def generate_visualization(nodes, arcs, track, gt_route, pred_route, metrics, output_html_path, title="Map Matching Visualization"):
    # Subsample nodes/arcs for background rendering to keep HTML lightweight
    # Extract route polylines (lat, lon)
    def route_to_latlngs(route_arcs):
        pts = []
        for arc_id in route_arcs:
            u, v = arcs[arc_id]
            # Leaflet expects [lat, lon]
            p1 = [float(nodes[u][1]), float(nodes[u][0])]
            p2 = [float(nodes[v][1]), float(nodes[v][0])]
            if not pts:
                pts.extend([p1, p2])
            else:
                pts.append(p2)
        return pts

    gt_latlngs = route_to_latlngs(gt_route) if gt_route is not None else []
    pred_latlngs = route_to_latlngs(pred_route)

    # Subsample track points (up to 300 points for crisp visual responsiveness)
    step = max(1, len(track) // 300)
    track_samples = [[float(p[1]), float(p[0]), int(p[2])] for p in track[::step]]

    # Map center
    center_lat = float(track[:, 1].mean())
    center_lon = float(track[:, 0].mean())

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .card {{
            position: absolute;
            top: 20px;
            right: 20px;
            z-index: 1000;
            background: rgba(255, 255, 255, 0.95);
            padding: 18px 22px;
            border-radius: 12px;
            box-shadow: 0 8px 30px rgba(0,0,0,0.2);
            max-width: 320px;
            backdrop-filter: blur(8px);
        }}
        .card h2 {{ margin: 0 0 10px 0; font-size: 18px; color: #1e293b; }}
        .stat-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }}
        .stat-box {{ background: #f8fafc; padding: 8px 10px; border-radius: 8px; border: 1px solid #e2e8f0; }}
        .stat-label {{ font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; }}
        .stat-val {{ font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 2px; }}
        .legend {{ margin-top: 14px; font-size: 13px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 6px; }}
        .legend-color {{ width: 16px; height: 16px; border-radius: 4px; margin-right: 8px; flex-shrink: 0; }}
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; background: #ecfdf5; color: #059669; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="card">
        <h2>{title} <span class="badge">HMM Match</span></h2>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-label">Recall</div>
                <div class="stat-val" style="color: #059669;">{metrics.get('recall', 0)*100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">Precision</div>
                <div class="stat-val" style="color: #2563eb;">{metrics.get('precision', 0)*100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">F1-Score</div>
                <div class="stat-val" style="color: #7c3aed;">{metrics.get('f1', 0)*100:.1f}%</div>
            </div>
            <div class="stat-box">
                <div class="stat-label">LCS Match</div>
                <div class="stat-val" style="color: #ea580c;">{metrics.get('lcs_accuracy', 0)*100:.1f}%</div>
            </div>
        </div>
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: #2563eb;"></div>
                <span><strong>Predicted Route</strong> ({len(pred_route)} arcs)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #10b981;"></div>
                <span><strong>Ground Truth Route</strong> ({len(gt_route) if gt_route else 0} arcs)</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #ef4444; border-radius: 50%;"></div>
                <span><strong>Raw Noisy GPS Points</strong> ({len(track)} pts)</span>
            </div>
        </div>
    </div>

    <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], 14);
        L.tileLayer('https://{{s}}.basemaps.cartocdn.com/rastertiles/voyager/{{z}}/{{x}}/{{y}}{{r}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap, © CARTO'
        }}).addTo(map);

        const gtCoords = {json.dumps(gt_latlngs)};
        const predCoords = {json.dumps(pred_latlngs)};
        const gpsPoints = {json.dumps(track_samples)};

        // 1. Ground Truth Polyline (Green, slightly thicker base)
        if (gtCoords.length > 0) {{
            const gtLine = L.polyline(gtCoords, {{
                color: '#10b981',
                weight: 6,
                opacity: 0.8
            }}).addTo(map).bindPopup("<b>Ground Truth Route</b>");
            map.fitBounds(gtLine.getBounds(), {{ padding: [50, 50] }});
        }}

        // 2. Predicted Route Polyline (Blue, sleek overlay)
        if (predCoords.length > 0) {{
            const predLine = L.polyline(predCoords, {{
                color: '#2563eb',
                weight: 3.5,
                opacity: 0.95,
                dashArray: '6, 6'
            }}).addTo(map).bindPopup("<b>Predicted Route (HMM Viterbi)</b>");
        }}

        // 3. GPS observations (Red circular dots)
        gpsPoints.forEach(p => {{
            L.circleMarker([p[0], p[1]], {{
                radius: 3,
                fillColor: '#ef4444',
                color: '#991b1b',
                weight: 1,
                opacity: 0.8,
                fillOpacity: 0.8
            }}).addTo(map).bindPopup(`GPS Pt: t=${{p[2]}}<br>Lat: ${{p[0].toFixed(5)}}<br>Lon: ${{p[1].toFixed(5)}}`);
        }});
    </script>
</body>
</html>
"""
    with open(output_html_path, "w") as f:
        f.write(html_content)
    print(f"Interactive visualization generated: {output_html_path}")
