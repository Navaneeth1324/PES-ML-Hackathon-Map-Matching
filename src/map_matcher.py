"""
HMM Map Matching Engine
=======================
Implements Hidden Markov Model (HMM) Map Matching based on Newson & Krumm (2009)
with spatial indexing (STRtree), Viterbi trellis decoding, and topological path stitching.

Formulation:
- Hidden states: Directed road network edges (arcs).
- Observations: Noisy GPS trajectory points.
- Emission probability: Gaussian distribution over perpendicular distance to candidate arcs.
- Transition probability: Exponential distribution over difference between network graph distance and Euclidean GPS displacement.
- Optimization: Viterbi Dynamic Programming in log-probability space.
- Post-processing: Blip smoothing and shortest-path graph stitching to guarantee valid topological continuity.
"""

import time
import math
import numpy as np
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


class MapMatcher:
    def __init__(self, sigma_z=7.0, beta=10.0, search_radius=50.0, top_k=5):
        """
        Parameters:
        - sigma_z: Standard deviation of GPS measurement noise (in meters).
        - beta: Scale parameter for transition probability exponential distribution.
        - search_radius: Maximum candidate search radius around each GPS point (meters).
        - top_k: Maximum number of candidate road segments to evaluate per GPS point.
        """
        self.sigma_z = sigma_z
        self.beta = beta
        self.search_radius = search_radius
        self.top_k = top_k

    def load_case(self, case_dir, case_id=None):
        """Loads nodes, arcs, track, and optional ground-truth route."""
        if case_id is None:
            case_id = case_dir.rstrip('/').split('/')[-1]

        with open(f"{case_dir}/{case_id}.nodes") as f:
            nodes = np.array([list(map(float, line.strip().split())) for line in f if line.strip()])

        with open(f"{case_dir}/{case_id}.arcs") as f:
            arcs = np.array([list(map(int, line.strip().split())) for line in f if line.strip()])

        with open(f"{case_dir}/{case_id}.track") as f:
            track = np.array([list(map(float, line.strip().split())) for line in f if line.strip()])

        gt_route = None
        route_file = f"{case_dir}/{case_id}.route"
        import os
        if os.path.exists(route_file):
            with open(route_file) as f:
                gt_route = [int(line.strip()) for line in f if line.strip()]

        return nodes, arcs, track, gt_route

    def match(self, nodes, arcs, track):
        """
        Performs end-to-end map matching on the provided nodes, arcs, and track.
        Returns:
            predicted_route: list of arc indices (ints) forming a continuous route.
            stats: dictionary of timing and internal matching statistics.
        """
        t0 = time.time()
        num_nodes = len(nodes)
        num_arcs = len(arcs)

        # 1. Metric local projection (Equirectangular)
        lon0, lat0 = nodes[:, 0].mean(), nodes[:, 1].mean()
        R = 6371000.0
        kx = np.cos(np.radians(lat0)) * (np.pi / 180.0) * R
        ky = (np.pi / 180.0) * R

        nodes_m = np.empty_like(nodes)
        nodes_m[:, 0] = (nodes[:, 0] - lon0) * kx
        nodes_m[:, 1] = (nodes[:, 1] - lat0) * ky

        track_m = np.empty_like(track[:, :2])
        track_m[:, 0] = (track[:, 0] - lon0) * kx
        track_m[:, 1] = (track[:, 1] - lat0) * ky

        # 2. Filter stationary / redundant GPS points (< 4.0 meters)
        filtered_indices = [0]
        for idx in range(1, len(track_m)):
            dist = np.linalg.norm(track_m[idx] - track_m[filtered_indices[-1]])
            if dist >= 4.0 or idx == len(track_m) - 1:
                filtered_indices.append(idx)
        track_pts = track_m[filtered_indices]

        # 3. Build directed network graph
        G = nx.DiGraph()
        arc_lens = np.empty(num_arcs)
        for arc_id, (u, v) in enumerate(arcs):
            l = np.linalg.norm(nodes_m[u] - nodes_m[v])
            arc_lens[arc_id] = l
            if not G.has_edge(u, v) or G[u][v]['weight'] > l:
                G.add_edge(u, v, weight=l, arc_id=arc_id)

        # 4. Build spatial index with STRtree
        lines = [LineString([nodes_m[u], nodes_m[v]]) for u, v in arcs]
        tree = STRtree(lines)

        # 5. Candidate generation & Emission probabilities
        query_pts = [Point(p[0], p[1]) for p in track_pts]
        cand_pairs = tree.query(query_pts, predicate='dwithin', distance=self.search_radius)

        point_to_cands = [[] for _ in range(len(track_pts))]
        for pt_idx, arc_idx in zip(cand_pairs[0], cand_pairs[1]):
            point_to_cands[pt_idx].append(arc_idx)

        candidates = []
        for i, pt in enumerate(track_pts):
            arc_list = point_to_cands[i]
            if not arc_list:
                nearest_idx = tree.nearest(Point(pt[0], pt[1]))
                arc_list = [nearest_idx]

            cands_for_pt = []
            P = pt
            for a_idx in arc_list:
                u, v = arcs[a_idx]
                A, B = nodes_m[u], nodes_m[v]
                AB = B - A
                L2 = np.dot(AB, AB)
                t = np.clip(np.dot(P - A, AB) / (L2 + 1e-9), 0.0, 1.0)
                proj = A + t * AB
                d = np.linalg.norm(P - proj)

                log_emit = - (d ** 2) / (2 * (self.sigma_z ** 2))
                cands_for_pt.append((a_idx, d, t, log_emit))

            cands_for_pt.sort(key=lambda x: x[3], reverse=True)
            candidates.append(cands_for_pt[:self.top_k])

        # 6. Viterbi Trellis Optimization
        dist_cache = {}

        def get_sp_dist(src, dst, max_cutoff):
            if src == dst:
                return 0.0
            key = (src, dst)
            if key in dist_cache:
                return dist_cache[key]
            try:
                length = nx.shortest_path_length(G, src, dst, weight='weight')
                if length <= max_cutoff:
                    dist_cache[key] = length
                    return length
                else:
                    dist_cache[key] = float('inf')
                    return float('inf')
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                dist_cache[key] = float('inf')
                return float('inf')

        T = len(track_pts)
        V = [{} for _ in range(T)]
        backpointer = [{} for _ in range(T)]

        for cand in candidates[0]:
            V[0][cand[0]] = cand[3]

        for t in range(1, T):
            d_gps = np.linalg.norm(track_pts[t] - track_pts[t - 1])
            curr_cands = candidates[t]
            prev_cands = candidates[t - 1]

            for curr_a, curr_d, curr_t, curr_emit in curr_cands:
                u2, v2 = arcs[curr_a]
                init2 = curr_t * arc_lens[curr_a]
                best_prob = -float('inf')
                best_prev = None

                for prev_a, prev_d, prev_t, prev_emit in prev_cands:
                    prev_prob = V[t - 1].get(prev_a, -float('inf'))
                    if prev_prob == -float('inf'):
                        continue

                    u1, v1 = arcs[prev_a]
                    if prev_a == curr_a:
                        if curr_t >= prev_t:
                            d_route = (curr_t - prev_t) * arc_lens[curr_a]
                        else:
                            d_route = (prev_t - curr_t) * arc_lens[curr_a] + 15.0
                    else:
                        rem1 = (1.0 - prev_t) * arc_lens[prev_a]
                        cutoff = max(200.0, d_gps * 2.2 + 50.0)
                        sp_dist = get_sp_dist(v1, u2, cutoff)
                        if sp_dist == float('inf'):
                            d_route = float('inf')
                        else:
                            d_route = rem1 + sp_dist + init2

                    if d_route == float('inf'):
                        log_trans = -45.0
                    else:
                        diff = abs(d_route - d_gps)
                        log_trans = - diff / self.beta

                    total_prob = prev_prob + log_trans + curr_emit
                    if total_prob > best_prob:
                        best_prob = total_prob
                        best_prev = prev_a

                if best_prev is not None:
                    V[t][curr_a] = best_prob
                    backpointer[t][curr_a] = best_prev
                else:
                    best_cand_prev = max(prev_cands, key=lambda c: V[t - 1].get(c[0], -float('inf')))
                    V[t][curr_a] = V[t - 1].get(best_cand_prev[0], 0.0) - 80.0 + curr_emit
                    backpointer[t][curr_a] = best_cand_prev[0]

        # 7. Backtrack Viterbi path
        best_last = max(V[T - 1].keys(), key=lambda k: V[T - 1][k])
        matched_arcs = [best_last]
        for t in range(T - 1, 0, -1):
            prev_a = backpointer[t].get(matched_arcs[-1], candidates[t - 1][0][0])
            matched_arcs.append(prev_a)
        matched_arcs.reverse()

        # 8. Blip Smoothing (Removes transient 1-2 step projection oscillations)
        for _ in range(2):
            for i in range(1, len(matched_arcs) - 1):
                if matched_arcs[i - 1] == matched_arcs[i + 1]:
                    matched_arcs[i] = matched_arcs[i - 1]
            for i in range(1, len(matched_arcs) - 2):
                if matched_arcs[i - 1] == matched_arcs[i + 2]:
                    matched_arcs[i] = matched_arcs[i - 1]
                    matched_arcs[i + 1] = matched_arcs[i - 1]

        # 9. Path Stitching (Fill gaps between consecutive matched arcs)
        full_route = []
        for i in range(len(matched_arcs)):
            a_curr = matched_arcs[i]
            if not full_route:
                full_route.append(a_curr)
                continue
            a_prev = full_route[-1]
            if a_curr == a_prev:
                continue

            u_prev, v_prev = arcs[a_prev]
            u_curr, v_curr = arcs[a_curr]

            if v_prev == u_curr:
                full_route.append(a_curr)
            else:
                try:
                    path = nx.shortest_path(G, source=v_prev, target=u_curr, weight='weight')
                    for p_idx in range(len(path) - 1):
                        n1, n2 = path[p_idx], path[p_idx + 1]
                        edge_arc = G[n1][n2]['arc_id']
                        if not full_route or full_route[-1] != edge_arc:
                            full_route.append(edge_arc)
                    if not full_route or full_route[-1] != a_curr:
                        full_route.append(a_curr)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    full_route.append(a_curr)

        # Ensure no consecutive duplicate arcs
        clean_route = [full_route[0]]
        for a in full_route[1:]:
            if a != clean_route[-1]:
                clean_route.append(a)

        elapsed = time.time() - t0
        stats = {
            "elapsed_seconds": elapsed,
            "raw_track_points": len(track),
            "filtered_points": len(track_pts),
            "route_arcs_count": len(clean_route)
        }
        return clean_route, stats
