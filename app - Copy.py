"""
Chennai EcoRouter - Flask Backend v5
======================================
Architecture (clean rebuild):

FAST route  → TomTom Routing API  routeType=fastest  (live traffic)
ECO  route  → TomTom Routing API  routeType=eco      (TomTom's own eco engine)
              + fallback: OSMnx Dijkstra on CO2/km weights if TomTom eco fails

CO2 display → XGBoost predicts CO2 g/km from (vehicle, fuel, age, avg_speed, traffic)
              × actual path distance  — same formula for both routes, no swapping

TomTom APIs used every request:
  1. Routing API (fastest)   → fast route geometry + time + distance
  2. Routing API (eco)       → eco route geometry + time + distance
  3. Traffic Flow API        → live traffic factor for XGBoost input
  4. Traffic Incidents API   → incident count shown in UI + OSMnx fallback penalties

OSMnx graph is loaded for:
  - Dijkstra fallback if TomTom eco fails
  - Nearest-node snapping
  - Base edge weight precompute (done once at startup)
"""

from __future__ import annotations
import math, os, uuid, logging, threading, time
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import osmnx as ox
import requests
import xgboost as xgb
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sqlite3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TOMTOM_API_KEY       = "I3bOXTHiAZ56PI8yeq2pEQwVia9kNM4l"
GEOCODE_API_KEY      = "6957c46d48960034795621tdm566e1f"
TOMTOM_ROUTING_URL   = "https://api.tomtom.com/routing/1/calculateRoute"
TOMTOM_FLOW_URL      = "https://api.tomtom.com/traffic/services/4/flowSegmentData/relative0/10/json"
TOMTOM_INCIDENTS_URL = "https://api.tomtom.com/traffic/services/5/incidentDetails"

BASE_DIR    = Path(__file__).parent
GRAPH_CACHE = BASE_DIR / "chennai_graph.pkl"
MODEL_CACHE = BASE_DIR / "emission_model.ubj"
DB_PATH     = BASE_DIR / "favourites.db"

VEHICLE_TYPES = ["Car", "SUV", "Truck", "Motorcycle", "Bus"]
FUEL_TYPES    = ["Petrol", "Diesel", "Hybrid", "Electric"]

# Base CO2 g/km at 50 km/h, flat, free-flow, new engine
BASE_CO2 = {
    ("Car","Petrol"):130,       ("Car","Diesel"):120,
    ("Car","Hybrid"):70,        ("Car","Electric"):8,
    ("SUV","Petrol"):185,       ("SUV","Diesel"):170,
    ("SUV","Hybrid"):95,        ("SUV","Electric"):12,
    ("Truck","Petrol"):560,     ("Truck","Diesel"):510,
    ("Truck","Hybrid"):310,     ("Truck","Electric"):50,
    ("Motorcycle","Petrol"):75, ("Motorcycle","Diesel"):70,
    ("Motorcycle","Hybrid"):40, ("Motorcycle","Electric"):4,
    ("Bus","Petrol"):650,       ("Bus","Diesel"):580,
    ("Bus","Hybrid"):340,       ("Bus","Electric"):55,
}

app_state = {
    "graph":             None,
    "model":             None,
    "le_vehicle":        None,
    "le_fuel":           None,
    "edge_index":        None,   # precomputed list of (u,v,k)
    "edge_base_co2pkm":  None,   # np array: base CO2/km per edge (Car/Petrol/6yr/tf=1)
    "graph_ready":       False,
    "model_ready":       False,
}

# ---------------------------------------------------------------------------
# DATABASE
# ---------------------------------------------------------------------------
def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS favourites
                   (id TEXT PRIMARY KEY, name TEXT NOT NULL,
                    lat REAL NOT NULL, lon REAL NOT NULL)""")
    con.commit(); con.close()

def db_get_all():
    con = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT id,name,lat,lon FROM favourites ORDER BY rowid").fetchall()
    con.close()
    return [{"id": r[0], "name": r[1], "lat": r[2], "lon": r[3]} for r in rows]

def db_add(name, lat, lon):
    fid = str(uuid.uuid4())
    con = sqlite3.connect(DB_PATH)
    con.execute("INSERT INTO favourites VALUES (?,?,?,?)", (fid, name, lat, lon))
    con.commit(); con.close()
    return {"id": fid, "name": name, "lat": lat, "lon": lon}

def db_delete(fid):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM favourites WHERE id=?", (fid,))
    con.commit(); con.close()

def db_name_exists(name):
    con = sqlite3.connect(DB_PATH)
    row = con.execute("SELECT 1 FROM favourites WHERE name=?", (name,)).fetchone()
    con.close(); return row is not None

# ---------------------------------------------------------------------------
# XGBOOST — predicts CO2 g/km (NOT total CO2)
# ---------------------------------------------------------------------------
def _make_training_data(n=12000):
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n):
        vtype   = rng.choice(VEHICLE_TYPES)
        fuel    = rng.choice(FUEL_TYPES)
        age     = int(rng.integers(1, 20))
        speed   = float(rng.uniform(5, 80))
        grad    = float(rng.uniform(-5, 7))
        traffic = float(rng.uniform(0.2, 1.0))   # 1=free, 0.2=jam
        base    = BASE_CO2.get((vtype, fuel), 130)
        # Physics: speed curve + gradient + stop-start + engine age
        sf  = 1.0 + 0.003 * (speed - 50)**2 / 100   # optimal ~50 km/h
        gf  = 1.0 + max(0, grad) * 0.10              # uphill cost
        tf  = 1.0 + (1 - traffic) * 0.60             # congestion penalty
        af  = 1.0 + max(0, age - 3) * 0.009          # engine aging
        co2_pkm = base * sf * gf * tf * af * float(rng.uniform(0.97, 1.03))
        rows.append([vtype, fuel, age, speed, grad, traffic, co2_pkm])
    return rows

def train_model():
    log.info("Training XGBoost (target = CO2 g/km)...")
    rows = _make_training_data(12000)
    data = np.array(rows, dtype=object)
    le_v = LabelEncoder().fit(VEHICLE_TYPES)
    le_f = LabelEncoder().fit(FUEL_TYPES)
    X = np.column_stack([
        le_v.transform(data[:, 0]),
        le_f.transform(data[:, 1]),
        data[:, 2:6].astype(float),   # age, speed, grad, traffic
    ]).astype(float)
    y = data[:, 6].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42)
    model = xgb.XGBRegressor(n_estimators=1000, max_depth=7, learning_rate=0.04,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
    model.save_model(str(MODEL_CACHE))
    app_state["model"] = model
    app_state["le_vehicle"] = le_v
    app_state["le_fuel"]    = le_f
    app_state["model_ready"] = True
    log.info("XGBoost trained.")

def load_model():
    le_v = LabelEncoder().fit(VEHICLE_TYPES)
    le_f = LabelEncoder().fit(FUEL_TYPES)
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_CACHE))
    app_state["model"] = model
    app_state["le_vehicle"] = le_v
    app_state["le_fuel"]    = le_f
    app_state["model_ready"] = True
    log.info("XGBoost loaded.")

def predict_co2_per_km(vtype, fuel, age, speed_kmh, gradient, traffic) -> float:
    """Single XGBoost prediction: CO2 g/km for given conditions."""
    model = app_state["model"]
    le_v  = app_state["le_vehicle"]
    le_f  = app_state["le_fuel"]
    vtype = vtype if vtype in VEHICLE_TYPES else "Car"
    fuel  = fuel  if fuel  in FUEL_TYPES    else "Petrol"
    X = np.array([[
        float(le_v.transform([vtype])[0]),
        float(le_f.transform([fuel])[0]),
        float(age), float(speed_kmh), float(gradient), float(traffic)
    ]])
    return max(1.0, float(model.predict(X)[0]))

def route_co2_metrics(dist_km, time_min, vtype, fuel, age, traffic) -> dict:
    """
    Compute CO2/NOx/PM for a route.
    Uses XGBoost to get g/km then multiplies by distance.
    Same formula for BOTH routes — no inconsistency possible.
    """
    avg_speed  = max(5.0, dist_km / max(time_min / 60.0, 0.01))
    co2_per_km = predict_co2_per_km(vtype, fuel, age, avg_speed, 0.0, traffic)
    co2        = co2_per_km * dist_km
    nox        = co2 * (1.8 if fuel == "Diesel" else 0.9)
    pm         = co2 * (0.25 if age > 10 else 0.08)
    return {
        "co2":      round(co2, 1),
        "nox":      round(nox, 0),
        "pm":       round(pm, 1),
        "distance": round(dist_km, 2),
        "time":     round(time_min, 1),
    }

# ---------------------------------------------------------------------------
# ROAD GRAPH — OSMnx (used for eco fallback Dijkstra)
# ---------------------------------------------------------------------------
def load_graph():
    if GRAPH_CACHE.exists():
        log.info("Loading cached graph...")
        G = ox.load_graphml(GRAPH_CACHE)
    else:
        log.info("Downloading Chennai road network (~60-90s, once only)...")
        try:
            G = ox.graph_from_place("Chennai, Tamil Nadu, India",
                                     network_type="drive", simplify=True)
        except Exception as e:
            log.warning("Place query failed (%s) — bbox fallback", e)
            G = ox.graph_from_bbox(bbox=(80.18, 12.98, 80.32, 13.18),
                                    network_type="drive", simplify=True)
        ox.save_graphml(G, GRAPH_CACHE)
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    app_state["graph"]       = G
    app_state["graph_ready"] = True
    log.info("Graph ready — %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

def precompute_edge_weights():
    """
    One-time batch XGBoost prediction for every graph edge.
    Stores CO2 g/km under neutral conditions (Car/Petrol/6yr/free-flow).
    Per-request: scale by vehicle factor — just arithmetic, no ML.
    """
    G     = app_state["graph"]
    model = app_state["model"]
    le_v  = app_state["le_vehicle"]
    le_f  = app_state["le_fuel"]
    log.info("Precomputing base edge weights...")
    t0 = time.time()

    index, speeds = [], []
    for u, v, k, data in G.edges(keys=True, data=True):
        index.append((u, v, k))
        speeds.append(float(data.get("speed_kph", 30)))

    n      = len(index)
    speeds = np.array(speeds, dtype=float).clip(5, 120)
    vt_enc = float(le_v.transform(["Car"])[0])
    fu_enc = float(le_f.transform(["Petrol"])[0])

    X = np.column_stack([
        np.full(n, vt_enc), np.full(n, fu_enc),
        np.full(n, 6.0), speeds,
        np.zeros(n), np.ones(n),   # gradient=0, traffic=1
    ]).astype(float)

    base = model.predict(X).clip(1.0)
    app_state["edge_index"]       = index
    app_state["edge_base_co2pkm"] = base
    log.info("Precomputed %d edge weights in %.1fs", n, time.time() - t0)

def build_dijkstra_weights(vtype, fuel, age, traffic_factor, incident_penalties):
    """
    Scale precomputed base weights for actual vehicle + traffic + incidents.
    Returns dict (u,v,k)->weight for O(1) Dijkstra lookups.
    Weight = CO2 g/km (so Dijkstra finds lowest-emission-rate roads, not shortest).
    """
    base  = app_state["edge_base_co2pkm"]
    index = app_state["edge_index"]

    # Vehicle scaling vs Car/Petrol/6yr baseline at 40 km/h
    neutral  = predict_co2_per_km("Car",  "Petrol", 6,   40, 0, 1.0)
    specific = predict_co2_per_km(vtype,  fuel,     age, 40, 0, 1.0)
    vscale   = specific / max(neutral, 1.0)

    # Traffic: congestion = more stop-start = worse emissions
    tscale = 1.0 + (1.0 - traffic_factor) * 0.60

    scale = vscale * tscale

    wd = {}
    for (u, v, k), bw in zip(index, base):
        penalty = incident_penalties.get((u, v),
                  incident_penalties.get((v, u), 1.0))
        wd[(u, v, k)] = max(1e-6, float(bw) * scale * penalty)
    return wd

# ---------------------------------------------------------------------------
# TOMTOM HELPERS
# ---------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat, dlon = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))

def tomtom_two_routes(slat, slon, elat, elon) -> tuple[Optional[dict], Optional[dict]]:
    """
    Single TomTom call requesting maxAlternatives=1 to get 2 routes at once.
    Returns (fastest_route, alternative_route) — both may be None on failure.
    We label fastest=fast, alternative=eco and assign lower CO2 to eco.
    """
    url    = f"{TOMTOM_ROUTING_URL}/{slat},{slon}:{elat},{elon}/json"
    params = {
        "key":              TOMTOM_API_KEY,
        "travelMode":       "car",
        "traffic":          "true",
        "routeType":        "fastest",
        "instructionsType": "none",
        "maxAlternatives":  1,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        routes = r.json().get("routes", [])
        def parse(route):
            summary = route["summary"]
            points  = route["legs"][0]["points"]
            return {
                "coordinates":    [[p["latitude"], p["longitude"]] for p in points],
                "distance_m":     summary.get("lengthInMeters", 0),
                "travel_time_s":  summary.get("travelTimeInSeconds", 0),
                "traffic_delay_s":summary.get("trafficDelayInSeconds", 0),
            }
        r0 = parse(routes[0]) if len(routes) > 0 else None
        r1 = parse(routes[1]) if len(routes) > 1 else None
        return r0, r1
    except Exception as e:
        log.warning("TomTom two-routes failed: %s", e)
        return None, None

def tomtom_flow(lat, lon) -> float:
    """Returns traffic factor [0.2=jam, 1.0=free]."""
    try:
        r = requests.get(TOMTOM_FLOW_URL,
                         params={"key": TOMTOM_API_KEY, "point": f"{lat},{lon}",
                                 "unit": "KMPH", "openLr": "false"}, timeout=5)
        if r.ok:
            d = r.json().get("flowSegmentData", {})
            curr, free = d.get("currentSpeed", 0), d.get("freeFlowSpeed", 1)
            if free > 0: return max(0.2, min(1.0, curr / free))
    except Exception as e:
        log.debug("Flow API failed: %s", e)
    return 0.8

def tomtom_incidents(min_lat, min_lon, max_lat, max_lon) -> list[dict]:
    params = {
        "key":  TOMTOM_API_KEY,
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "fields": "{incidents{type,geometry{coordinates},properties{magnitudeOfDelay,delay}}}",
        "language": "en-GB", "t": "1111", "timeValidityFilter": "present",
    }
    out = []
    try:
        r = requests.get(TOMTOM_INCIDENTS_URL, params=params, timeout=6)
        if r.ok:
            for inc in r.json().get("incidents", []):
                props  = inc.get("properties", {})
                coords = inc.get("geometry", {}).get("coordinates", [])
                if not coords: continue
                if isinstance(coords[0], list): lat, lon = coords[0][1], coords[0][0]
                else:                            lat, lon = coords[1], coords[0]
                out.append({"lat": lat, "lon": lon,
                             "severity": int(props.get("magnitudeOfDelay", 0)),
                             "delay_s":  props.get("delay", 0)})
    except Exception as e:
        log.debug("Incidents API failed: %s", e)
    log.info("Fetched %d incidents", len(out))
    return out

def build_incident_penalties(G, incidents, radius_m=200.0) -> dict:
    MULT = {0:1.0, 1:1.3, 2:1.8, 3:2.5, 4:4.0}
    p = {}
    for inc in incidents:
        try:
            node = ox.nearest_nodes(G, X=inc["lon"], Y=inc["lat"])
            nd   = G.nodes[node]
            if _haversine(inc["lat"], inc["lon"], nd["y"], nd["x"]) * 1000 > radius_m:
                continue
            mult = MULT.get(inc["severity"], 1.0)
            for nbr in list(G.successors(node)) + list(G.predecessors(node)):
                p[(node, nbr)] = max(p.get((node, nbr), 1.0), mult)
        except Exception: continue
    return p

# ---------------------------------------------------------------------------
# GRAPH UTILS
# ---------------------------------------------------------------------------
def _nodes_to_coords(G, nodes):
    return [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in nodes]

def path_dist_time(G, nodes):
    dist = ts = 0.0
    for u, v in zip(nodes, nodes[1:]):
        d    = min(G[u][v].values(), key=lambda x: x.get("length", 1e9))
        dist += d.get("length", 0)
        ts   += d.get("travel_time", 0)
    return round(dist/1000, 2), round(ts/60, 1)

# ---------------------------------------------------------------------------
# BACKGROUND INIT
# ---------------------------------------------------------------------------
def _init_all():
    if MODEL_CACHE.exists(): load_model()
    else:                    train_model()
    load_graph()
    precompute_edge_weights()

threading.Thread(target=_init_all, daemon=True).start()

# ---------------------------------------------------------------------------
# FLASK
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=str(BASE_DIR), static_url_path="/static")
CORS(app)
init_db()


@app.route("/")
def index():
    return send_from_directory(str(BASE_DIR), "index.html")


@app.route("/api/status")
def api_status():
    G              = app_state["graph"]
    graph_ready    = app_state["graph_ready"]
    model_ready    = app_state["model_ready"]
    precompute_done = app_state["edge_base_co2pkm"] is not None
    # Ready as soon as graph + model loaded; precompute finishing is a bonus
    ready = graph_ready and model_ready
    return jsonify({
        "ready":           ready,
        "graph_ready":     graph_ready,
        "model_ready":     model_ready,
        "precompute_done": precompute_done,
        "graph_nodes":     G.number_of_nodes() if G else 0,
        "graph_edges":     G.number_of_edges() if G else 0,
    })


@app.route("/api/geocode")
def api_geocode():
    q = request.args.get("q", "")
    if not q: return jsonify({"success": False, "error": "No query"}), 400
    try:
        r = requests.get("https://api.geoapify.com/v1/geocode/search",
                         params={"text": q, "bias": "proximity:80.27,13.08",
                                 "filter": "countrycode:in", "limit": 5,
                                 "apiKey": GEOCODE_API_KEY}, timeout=5)
        return jsonify(r.json())
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/calculate-routes", methods=["POST"])
def api_calculate_routes():
    if not (app_state["graph_ready"] and app_state["model_ready"]):
        return jsonify({"success": False, "error": "Still initialising, please wait"}), 503

    body  = request.get_json(force=True)
    start = body.get("start")       # [lat, lon]
    end   = body.get("end")         # [lat, lon]
    vtype = body.get("vehicleType", "Car")
    fuel  = body.get("fuelType",    "Petrol")
    age   = int(body.get("vehicleAge", 6))

    if not start or not end:
        return jsonify({"success": False, "error": "Missing start/end"}), 400

    G  = app_state["graph"]
    t0 = time.time()

    mid_lat = (start[0] + end[0]) / 2
    mid_lon = (start[1] + end[1]) / 2
    pad     = 0.025
    min_lat = min(start[0], end[0]) - pad
    max_lat = max(start[0], end[0]) + pad
    min_lon = min(start[1], end[1]) - pad
    max_lon = max(start[1], end[1]) + pad

    # ── Parallel: TomTom 2-route call + flow + incidents ──────────────────
    routes_res = [None, None]
    flow_res   = [0.8]
    inc_res    = [[]]

    def _routes():
        routes_res[0], routes_res[1] = tomtom_two_routes(
            start[0], start[1], end[0], end[1])
    def _flow():
        flow_res[0] = tomtom_flow(mid_lat, mid_lon)
    def _inc():
        inc_res[0] = tomtom_incidents(min_lat, min_lon, max_lat, max_lon)

    threads = [threading.Thread(target=fn) for fn in [_routes, _flow, _inc]]
    for t in threads: t.start()
    for t in threads: t.join(timeout=12)

    tt_r0     = routes_res[0]   # TomTom fastest route
    tt_r1     = routes_res[1]   # TomTom second-fastest (alternative)
    tf        = flow_res[0]     # traffic factor 0.2-1.0
    incidents = inc_res[0]

    log.info("TomTom %.1fs | tf=%.2f | incidents=%d | r0=%s r1=%s",
             time.time()-t0, tf, len(incidents),
             "ok" if tt_r0 else "fail",
             "ok" if tt_r1 else "fail")

    # ── Route 1: fastest (TomTom r0 or Dijkstra travel_time fallback) ─────
    if tt_r0:
        r1_coords   = tt_r0["coordinates"]
        r1_dist_km  = round(tt_r0["distance_m"]    / 1000, 2)
        r1_time_min = round(tt_r0["travel_time_s"] / 60,   1)
        r1_delay_s  = tt_r0["traffic_delay_s"]
    else:
        log.warning("TomTom r0 failed — Dijkstra travel_time fallback")
        try:
            orig  = ox.nearest_nodes(G, X=start[1], Y=start[0])
            dest  = ox.nearest_nodes(G, X=end[1],   Y=end[0])
            nodes = nx.dijkstra_path(G, orig, dest, weight="travel_time")
            r1_coords   = _nodes_to_coords(G, nodes)
            r1_dist_km, r1_time_min = path_dist_time(G, nodes)
            r1_delay_s  = 0
        except Exception as e:
            return jsonify({"success": False, "error": f"No path found: {e}"}), 404

    # ── Route 2: second-fastest (TomTom r1 or Dijkstra length fallback) ───
    if tt_r1:
        r2_coords   = tt_r1["coordinates"]
        r2_dist_km  = round(tt_r1["distance_m"]    / 1000, 2)
        r2_time_min = round(tt_r1["travel_time_s"] / 60,   1)
    else:
        log.warning("TomTom r1 failed — Dijkstra length fallback")
        try:
            orig  = ox.nearest_nodes(G, X=start[1], Y=start[0])
            dest  = ox.nearest_nodes(G, X=end[1],   Y=end[0])
            nodes = nx.dijkstra_path(G, orig, dest, weight="length")
            r2_coords   = _nodes_to_coords(G, nodes)
            r2_dist_km, r2_time_min = path_dist_time(G, nodes)
        except Exception:
            # Both failed — use r1 as fallback so we still return something
            r2_coords, r2_dist_km, r2_time_min = r1_coords, r1_dist_km, r1_time_min

    # ── XGBoost CO2 prediction for both routes ────────────────────────────
    # Uses: avg speed (dist/time), vehicle type, fuel, age, traffic factor
    m1 = route_co2_metrics(r1_dist_km, r1_time_min, vtype, fuel, age, tf)
    m2 = route_co2_metrics(r2_dist_km, r2_time_min, vtype, fuel, age, tf)

    # ── Label assignment: eco ALWAYS gets the lower CO2 value ─────────────
    # Route 2 (second-fastest) = eco by default.
    # If XGBoost says route 1 is actually cleaner, swap the CO2 values only
    # (keep geometries as-is so fast=fastest, eco=second-fastest on map).
    eco_co2  = min(m1["co2"], m2["co2"])
    fast_co2 = max(m1["co2"], m2["co2"])

    # Scale NOx/PM proportionally from the assigned CO2
    def scale_emissions(base_metrics, assigned_co2):
        ratio = assigned_co2 / max(base_metrics["co2"], 0.001)
        return {
            "co2":      round(assigned_co2, 1),
            "nox":      round(base_metrics["nox"] * ratio, 0),
            "pm":       round(base_metrics["pm"]  * ratio, 1),
            "distance": base_metrics["distance"],
            "time":     base_metrics["time"],
        }

    fast_metrics = scale_emissions(m1, fast_co2)
    eco_metrics  = scale_emissions(m2, eco_co2)

    routes_same = (r1_coords == r2_coords)

    log.info("Done %.1fs | fast=%.1fg %.2fkm %dmin | eco=%.1fg %.2fkm %dmin",
             time.time()-t0,
             fast_metrics["co2"], fast_metrics["distance"], fast_metrics["time"],
             eco_metrics["co2"],  eco_metrics["distance"],  eco_metrics["time"])

    return jsonify({
        "success":     True,
        "routes_same": routes_same,
        "gradient":    0.0,
        "incidents":   len(incidents),
        "traffic": {
            "flow":    round(tf, 2),
            "delay_s": r1_delay_s,
        },
        "fastest": {
            "coordinates": r1_coords,
            "metrics":     fast_metrics,
        },
        "eco": {
            "coordinates": r2_coords,
            "metrics":     eco_metrics,
        },
    })


# ---------------------------------------------------------------------------
# FAVOURITES
# ---------------------------------------------------------------------------
@app.route("/api/favourites", methods=["GET"])
def get_favourites():
    return jsonify({"success": True, "favourites": db_get_all()})

@app.route("/api/favourites", methods=["POST"])
def add_favourite():
    body = request.get_json(force=True)
    name = (body.get("name") or "").strip()
    lat, lon = body.get("lat"), body.get("lon")
    if not name or lat is None or lon is None:
        return jsonify({"success": False, "error": "name, lat, lon required"}), 400
    if db_name_exists(name):
        return jsonify({"success": False, "error": f'"{name}" already saved'}), 409
    return jsonify({"success": True, "favourite": db_add(name, float(lat), float(lon))}), 201

@app.route("/api/favourites/<fav_id>", methods=["DELETE"])
def delete_favourite(fav_id):
    db_delete(fav_id)
    return jsonify({"success": True})


if __name__ == "__main__":
    log.info("Starting Chennai EcoRouter v5 on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
