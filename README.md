# Chennai EcoRouter

A traffic-aware routing application for Chennai that compares the fastest route against a lower-emission alternative, with real-time CO2 predictions powered by a trained XGBoost model.

---

## Overview

Chennai EcoRouter calculates two routes between any two points in Chennai — the fastest route based on live traffic, and a second alternative labelled as the eco route. CO2, NOx, and PM2.5 emissions are predicted for both routes using a machine learning model trained on vehicle physics data. The eco route is always assigned the lower emissions value of the two.

The application uses the TomTom Routing and Traffic APIs for route geometry and live traffic data, Geoapify for geocoding, and OpenStreetMap via OSMnx as a fallback routing graph.

---

## Features

- Dual-route comparison: fastest vs. eco alternative
- Real-time CO2, NOx, and PM2.5 emission predictions via XGBoost
- Live traffic integration using TomTom Flow and Incidents APIs
- Vehicle configuration: type, fuel, and age all affect emission estimates
- Location search scoped to the Chennai metro region
- Favourite locations saved locally via SQLite
- Map pin support for selecting start and destination directly

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Routing | TomTom Routing API, OSMnx, NetworkX |
| Traffic | TomTom Flow API, TomTom Incidents API |
| Geocoding | Geoapify |
| ML Model | XGBoost |
| Frontend | HTML, CSS, JavaScript, Leaflet.js |
| Database | SQLite |
| Deployment | Render (Gunicorn) |

---

## Project Structure

```
ecorouter/
├── app.py               # Flask backend
├── index.html           # Frontend
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not committed)
├── .gitignore
└── README.md
```

Files generated at runtime (excluded from version control):

```
chennai_graph.pkl        # OSMnx road graph cache (~170MB)
emission_model.ubj       # Trained XGBoost model
favourites.db            # SQLite database
```

---

## Setup

### Prerequisites

- Python 3.10 or higher
- pip

### Installation

```bash
git clone https://github.com/athithyakarthikeyan/ecorouter.git
cd ecorouter
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root:

```
TOMTOM_API_KEY=your_tomtom_key
GEOCODE_API_KEY=your_geoapify_key
```

API keys can be obtained from:
- TomTom: https://developer.tomtom.com
- Geoapify: https://www.geoapify.com

### Running Locally

```bash
python app.py
```

The server starts at `http://localhost:5000`.

On first run, the application will download the Chennai road network from OpenStreetMap (~60-90 seconds) and train the XGBoost model. Both are cached to disk and loaded instantly on subsequent runs.

---

## Deployment on Render

1. Push the repository to GitHub (ensure `chennai_graph.pkl`, `emission_model.ubj`, `favourites.db`, and `.env` are in `.gitignore`)
2. Create a new Web Service on Render pointing to the repository
3. Set the start command to: `gunicorn app:app`
4. Add environment variables in the Render dashboard under the Environment tab:
   - `TOMTOM_API_KEY`
   - `GEOCODE_API_KEY`

Note: On first deploy, Render will download the road network and train the model during startup. This takes approximately 2-3 minutes. Subsequent deploys use the cached files if a persistent disk is attached.

---

## How It Works

### Routing

Each route request fires three parallel API calls:

1. **TomTom Routing API** with `maxAlternatives=1` — returns the fastest route and one alternative in a single call
2. **TomTom Flow API** — returns the current traffic factor at the route midpoint
3. **TomTom Incidents API** — returns active incidents in the bounding box

The fastest route (TomTom result 0) is labelled Latency Optimized. The alternative (TomTom result 1) is labelled Emissions Optimized. If TomTom does not return an alternative, OSMnx Dijkstra is used as a fallback.

### Emission Prediction

An XGBoost regression model is trained at startup on 12,000 synthetic samples generated from vehicle physics equations. The model predicts CO2 in grams per kilometre given:

- Vehicle type and fuel type
- Vehicle age
- Average speed on the route
- Traffic congestion factor

Total CO2 is calculated as `CO2 g/km * route distance`. The eco route is always assigned the lower of the two predicted values. NOx and PM2.5 are derived proportionally from CO2.

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Returns backend readiness state |
| `/api/geocode?q=` | GET | Geocodes a place name within Chennai |
| `/api/calculate-routes` | POST | Returns two routes with emission metrics |
| `/api/favourites` | GET | Lists saved favourite locations |
| `/api/favourites` | POST | Saves a new favourite location |
| `/api/favourites/<id>` | DELETE | Deletes a favourite location |


---

## License

MIT
