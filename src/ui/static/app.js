/**
 * PathSnap Evaluator Interface - Client Application (Laptop 2)
 * Coordinates Leaflet rendering, live API proxying, and diagnostic reporting.
 */

// Application State
const state = {
  selectedTripId: "00000000",
  selectedModel: "Boosting",
  mode: "auto", // 'auto', 'live', or 'mock'
  backendUrl: "http://localhost:8000",
  lastResponse: null,
  isProcessing: false,
};

// API Base calculation (supports both HTTP server and direct file:/// browser opening)
const isFileProtocol = window.location.protocol === "file:";
const apiBase = isFileProtocol ? "http://localhost:8501" : "";
const RESERVED_FALLBACK_TRIPS = [
  "00000000", "00000007", "00000009", "00000040", "00000052",
  "00000056", "00000065", "00000076", "00000077", "00000091"
];

// Map & Layer References
let map = null;
let layerGroups = {
  gps: null,
  predicted: null,
  groundTruth: null,
  roadArcs: null,
};

// DOM Elements
const elements = {
  backendUrlInput: document.getElementById("backendUrlInput"),
  pingBtn: document.getElementById("pingBtn"),
  connectionStatus: document.getElementById("connectionStatus"),
  connectionText: document.getElementById("connectionText"),
  modeAutoBtn: document.getElementById("modeAutoBtn"),
  modeLiveBtn: document.getElementById("modeLiveBtn"),
  modeMockBtn: document.getElementById("modeMockBtn"),
  tripGrid: document.getElementById("tripGrid"),
  customTripInput: document.getElementById("customTripInput"),
  selectCustomBtn: document.getElementById("selectCustomBtn"),
  modelSelect: document.getElementById("modelSelect"),
  runBtn: document.getElementById("runBtn"),
  toggleGps: document.getElementById("toggleGps"),
  togglePredicted: document.getElementById("togglePredicted"),
  toggleGroundTruth: document.getElementById("toggleGroundTruth"),
  toggleArcs: document.getElementById("toggleArcs"),
  loadingOverlay: document.getElementById("loadingOverlay"),
  tabLogsBtn: document.getElementById("tabLogsBtn"),
  tabJsonBtn: document.getElementById("tabJsonBtn"),
  tabSummaryBtn: document.getElementById("tabSummaryBtn"),
  tabLogs: document.getElementById("tabLogs"),
  tabJson: document.getElementById("tabJson"),
  tabSummary: document.getElementById("tabSummary"),
  terminalLogs: document.getElementById("terminalLogs"),
  jsonContent: document.getElementById("jsonContent"),
  copyJsonBtn: document.getElementById("copyJsonBtn"),
  exportGeoJsonBtn: document.getElementById("exportGeoJsonBtn"),
  // Metric card elements
  metricDecodedTop1: document.getElementById("metricDecodedTop1"),
  metricTop1Sub: document.getElementById("metricTop1Sub"),
  metricContinuity: document.getElementById("metricContinuity"),
  metricContinuitySub: document.getElementById("metricContinuitySub"),
  metricF1: document.getElementById("metricF1"),
  metricLoss: document.getElementById("metricLoss"),
  metricLatency: document.getElementById("metricLatency"),
  metricLatencySub: document.getElementById("metricLatencySub"),
  metricPoints: document.getElementById("metricPoints"),
  metricPointsSub: document.getElementById("metricPointsSub"),
};

// Initialize Application
document.addEventListener("DOMContentLoaded", () => {
  initMap();
  setupEventListeners();
  loadTripsList();
  pingBackend();

  // Run initial evaluation for trip 00000000 on startup
  setTimeout(() => {
    executeEvaluation();
  }, 400);
});

/** Initialize Leaflet Map with multi-layer fallback */
function initMap() {
  map = L.map("map", {
    center: [56.25, 54.96],
    zoom: 14,
    zoomControl: true,
  });

  // Base tile layers (Esri primary, OSM fallback)
  const esriTiles = L.tileLayer(
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    {
      maxZoom: 19,
      attribution: "Tiles &copy; Esri &mdash; PathSnap Evaluator",
    }
  );

  const osmTiles = L.tileLayer(
    "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    {
      maxZoom: 19,
      attribution: "&copy; OpenStreetMap contributors",
    }
  );

  const cartoDarkTiles = L.tileLayer(
    "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    {
      maxZoom: 19,
      attribution: "&copy; CARTO",
    }
  );

  esriTiles.addTo(map);

  // Initialize Layer Groups
  layerGroups.roadArcs = L.featureGroup().addTo(map);
  layerGroups.groundTruth = L.featureGroup().addTo(map);
  layerGroups.predicted = L.featureGroup().addTo(map);
  layerGroups.gps = L.featureGroup().addTo(map);

  L.control.scale({ imperial: false, metric: true }).addTo(map);
}

/** Attach DOM event handlers */
function setupEventListeners() {
  // Backend URL input & Ping
  elements.pingBtn.addEventListener("click", () => pingBackend());
  elements.backendUrlInput.addEventListener("change", (e) => {
    state.backendUrl = e.target.value.trim();
  });

  // Mode toggles
  elements.modeAutoBtn.addEventListener("click", () => setMode("auto"));
  elements.modeLiveBtn.addEventListener("click", () => setMode("live"));
  elements.modeMockBtn.addEventListener("click", () => setMode("mock"));

  // Model selector
  elements.modelSelect.addEventListener("change", (e) => {
    state.selectedModel = e.target.value;
    logMessage("INFO", `Selected model architecture: ${state.selectedModel}`);
  });

  // Custom Trip Input
  elements.selectCustomBtn.addEventListener("click", () => {
    const customId = elements.customTripInput.value.trim();
    if (customId) {
      selectTrip(customId);
    }
  });

  // Run Button
  elements.runBtn.addEventListener("click", () => {
    executeEvaluation();
  });

  // Layer Toggles
  elements.toggleGps.addEventListener("change", (e) => {
    toggleLayer(layerGroups.gps, e.target.checked);
  });
  elements.togglePredicted.addEventListener("change", (e) => {
    toggleLayer(layerGroups.predicted, e.target.checked);
  });
  elements.toggleGroundTruth.addEventListener("change", (e) => {
    toggleLayer(layerGroups.groundTruth, e.target.checked);
  });
  elements.toggleArcs.addEventListener("change", (e) => {
    toggleLayer(layerGroups.roadArcs, e.target.checked);
    // If toggled on and we don't have arcs yet, re-run with include_road_arcs=true
    if (e.target.checked && layerGroups.roadArcs.getLayers().length === 0) {
      executeEvaluation();
    }
  });

  // Tabs
  elements.tabLogsBtn.addEventListener("click", () => switchTab("logs"));
  elements.tabJsonBtn.addEventListener("click", () => switchTab("json"));
  elements.tabSummaryBtn.addEventListener("click", () => switchTab("summary"));

  // Copy JSON
  elements.copyJsonBtn.addEventListener("click", () => {
    if (state.lastResponse) {
      navigator.clipboard.writeText(JSON.stringify(state.lastResponse, null, 2));
      const originalText = elements.copyJsonBtn.textContent;
      elements.copyJsonBtn.textContent = "Copied!";
      setTimeout(() => (elements.copyJsonBtn.textContent = originalText), 1500);
    }
  });

  // Export GeoJSON
  elements.exportGeoJsonBtn.addEventListener("click", () => exportGeoJSON());
}

function setMode(newMode) {
  state.mode = newMode;
  elements.modeAutoBtn.classList.toggle("active", newMode === "auto");
  elements.modeLiveBtn.classList.toggle("active", newMode === "live");
  elements.modeMockBtn.classList.toggle("active", newMode === "mock");
  logMessage("INFO", `Switched deployment mode to: ${newMode.toUpperCase()}`);
}

function toggleLayer(layerGroup, visible) {
  if (visible) {
    if (!map.hasLayer(layerGroup)) map.addLayer(layerGroup);
  } else {
    if (map.hasLayer(layerGroup)) map.removeLayer(layerGroup);
  }
}

/** Check connection to Laptop 1 Model API */
async function pingBackend() {
  const targetUrl = elements.backendUrlInput.value.trim();
  state.backendUrl = targetUrl;
  logMessage("INFO", `Pinging target API at ${targetUrl}...`);

  try {
    const res = await fetch(`${apiBase}/api/ping_backend`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ backend_url: targetUrl }),
    });
    const data = await res.json();

    if (data.connected) {
      elements.connectionStatus.className = "status-badge online";
      elements.connectionText.textContent = `Online (${data.latency_ms} ms)`;
      logMessage("SUCCESS", `Connected to Laptop 1 API: ${data.endpoint_reached} (${data.latency_ms} ms)`);
    } else {
      if (state.mode === "mock") {
        elements.connectionStatus.className = "status-badge mock";
        elements.connectionText.textContent = "Offline Mock Mode";
      } else {
        elements.connectionStatus.className = "status-badge offline";
        elements.connectionText.textContent = "Laptop 1 Offline";
      }
      logMessage("WARN", `Could not connect to Laptop 1 at ${targetUrl}: ${data.message}`);
    }
  } catch (err) {
    elements.connectionStatus.className = "status-badge mock";
    elements.connectionText.textContent = "Offline Mock Mode";
    logMessage("WARN", `Backend ping error: ${err.message}. Local engine ready.`);
  }
}

/** Load 10 reserved evaluation trips */
async function loadTripsList() {
  try {
    const res = await fetch(`${apiBase}/api/trips`);
    const data = await res.json();
    elements.tripGrid.innerHTML = "";

    data.trips.forEach((t) => {
      const btn = document.createElement("button");
      btn.className = `trip-btn ${t.trip_id === state.selectedTripId ? "active" : ""}`;
      btn.textContent = t.trip_id;
      btn.title = t.description;
      btn.addEventListener("click", () => selectTrip(t.trip_id));
      elements.tripGrid.appendChild(btn);
    });
  } catch (err) {
    logMessage("WARN", `Failed to load trip inventory from server, using local reserved trip list.`);
    elements.tripGrid.innerHTML = "";
    RESERVED_FALLBACK_TRIPS.forEach((tid) => {
      const btn = document.createElement("button");
      btn.className = `trip-btn ${tid === state.selectedTripId ? "active" : ""}`;
      btn.textContent = tid;
      btn.title = `Reserved Student-Test Trip ${tid}`;
      btn.addEventListener("click", () => selectTrip(tid));
      elements.tripGrid.appendChild(btn);
    });
  }
}

/** Select a trip by ID */
function selectTrip(tripId) {
  state.selectedTripId = tripId;

  // Update button active states
  document.querySelectorAll(".trip-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.textContent === tripId);
  });

  logMessage("INFO", `Trip selected: ${tripId}`);
}

/** Execute Evaluation against API or Local Engine */
async function executeEvaluation() {
  if (state.isProcessing) return;
  state.isProcessing = true;
  elements.loadingOverlay.style.display = "flex";

  const requestPayload = {
    trip_id: state.selectedTripId,
    model_name: state.selectedModel,
    include_geometry: true,
    include_ground_truth: true,
    include_road_arcs: elements.toggleArcs.checked,
  };

  const startTime = performance.now();
  logMessage(
    "INFO",
    `Triggering evaluation: Trip=${requestPayload.trip_id}, Model=${requestPayload.model_name}, Mode=${state.mode}`
  );

  const queryParams = new URLSearchParams({
    mode: state.mode,
    backend_url: state.backendUrl,
  });

  try {
    const res = await fetch(`${apiBase}/api/evaluate?${queryParams.toString()}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestPayload),
    });

    const elapsedMs = Math.round(performance.now() - startTime);

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || `Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    state.lastResponse = data;

    logMessage(
      "SUCCESS",
      `HTTP 200 OK | Trip ${data.trip_id} processed in ${elapsedMs} ms (Model inference: ${data.metrics.inference_time_ms != null ? data.metrics.inference_time_ms : '?'} ms)`
    );

    // Update UI components
    updateMetricsDashboard(data);
    renderMapData(data);
    updateJsonInspector(data);
    updateSummaryTable(data);
  } catch (err) {
    logMessage("ERROR", `Evaluation failed: ${err.message}`);
    if (state.mode === "live") {
      logMessage("WARN", "Laptop 1 unreachable. Automatically falling back to local simulation mode...");
      setMode("mock");
      setTimeout(() => executeEvaluation(), 100);
      return;
    }
  } finally {
    state.isProcessing = false;
    elements.loadingOverlay.style.display = "none";
  }
}

/** Update Metric Cards */
function updateMetricsDashboard(data) {
  const m = data.metrics;

  // Top-1 Accuracy — handle null values gracefully
  const decTop1 = m.decoded_top1_accuracy != null ? m.decoded_top1_accuracy : m.top1_accuracy;
  const rawTop1 = m.top1_accuracy != null ? m.top1_accuracy : 0;
  if (decTop1 != null) {
    const decTop1Pct = (decTop1 * 100).toFixed(2);
    const rawTop1Pct = (rawTop1 * 100).toFixed(2);
    const diffPct = (decTop1 - rawTop1) * 100;
    elements.metricDecodedTop1.textContent = `${decTop1Pct}%`;
    if (rawTop1 > 0 && rawTop1 !== decTop1) {
      elements.metricTop1Sub.textContent = `Raw: ${rawTop1Pct}% (${diffPct >= 0 ? "+" : ""}${diffPct.toFixed(2)}% via HMM)`;
    } else {
      elements.metricTop1Sub.textContent = `Viterbi-decoded prediction`;
    }
  } else {
    elements.metricDecodedTop1.textContent = "N/A";
    elements.metricTop1Sub.textContent = "No server-side ground truth";
  }

  // Continuity — handle null values gracefully
  const decCont = m.decoded_continuity != null ? m.decoded_continuity : m.route_continuity;
  const rawCont = m.route_continuity != null ? m.route_continuity : 0;
  if (decCont != null) {
    const decContPct = (decCont * 100).toFixed(2);
    const rawContPct = (rawCont * 100).toFixed(2);
    elements.metricContinuity.textContent = `${decContPct}%`;
    if (rawCont > 0 && rawCont !== decCont) {
      elements.metricContinuitySub.textContent = `Raw: ${rawContPct}% | Graph coherent`;
    } else {
      elements.metricContinuitySub.textContent = `Topologically coherent`;
    }
  } else {
    elements.metricContinuity.textContent = "N/A";
    elements.metricContinuitySub.textContent = "Not available";
  }

  // Macro F1
  elements.metricF1.textContent = m.macro_f1 != null ? m.macro_f1.toFixed(4) : "N/A";

  // Log Loss
  elements.metricLoss.textContent = m.log_loss != null ? m.log_loss.toFixed(4) : "N/A";

  // Latency
  const infMs = m.inference_time_ms != null ? m.inference_time_ms : 0;
  elements.metricLatency.textContent = `${infMs.toFixed(1)} ms`;
  elements.metricLatencySub.textContent = `Scoring & Viterbi decode`;

  // Points & Route Arcs
  const numArcs = data.predicted_arc_ids ? data.predicted_arc_ids.length : 0;
  const totalPts = m.total_points != null ? m.total_points : (data.raw_gps ? data.raw_gps.length : 0);
  elements.metricPoints.textContent = `${totalPts.toLocaleString()}`;
  elements.metricPointsSub.textContent = `${numArcs} reconstructed arcs`;
}

/** Render GPS observations, predicted route, and ground truth on Leaflet */
function renderMapData(data) {
  // Clear previous layers
  layerGroups.gps.clearLayers();
  layerGroups.predicted.clearLayers();
  layerGroups.groundTruth.clearLayers();
  layerGroups.roadArcs.clearLayers();

  const allBoundsPoints = [];

  // 1. Render Sampled Road Network Arcs (Grey)
  if (data.road_arcs_coords && data.road_arcs_coords.length > 0) {
    data.road_arcs_coords.forEach((segment) => {
      L.polyline(segment, {
        color: "#64748b",
        weight: 1.5,
        opacity: 0.45,
      }).addTo(layerGroups.roadArcs);
    });
  }

  // 2. Render Ground Truth Route (Green)
  if (data.ground_truth_coords && data.ground_truth_coords.length > 0) {
    const gtPolyline = L.polyline(data.ground_truth_coords, {
      color: "#22c55e",
      weight: 5,
      opacity: 0.85,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(layerGroups.groundTruth);

    gtPolyline.bindPopup(
      `<strong>Ground Truth Reference Route</strong><br>Total Vertices: ${data.ground_truth_coords.length}`
    );
    data.ground_truth_coords.forEach((pt) => allBoundsPoints.push(pt));
  }

  // 3. Render Predicted Route (Vibrant Purple/Indigo with glow)
  if (data.predicted_route_coords && data.predicted_route_coords.length > 0) {
    // Halo line for visibility
    L.polyline(data.predicted_route_coords, {
      color: "#4338ca",
      weight: 7,
      opacity: 0.5,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(layerGroups.predicted);

    // Primary route line
    const predPolyline = L.polyline(data.predicted_route_coords, {
      color: "#a855f7",
      weight: 4,
      opacity: 0.95,
      lineCap: "round",
      lineJoin: "round",
    }).addTo(layerGroups.predicted);

    predPolyline.bindPopup(
      `<strong>PathSnap Reconstructed Trajectory</strong><br>Model: ${data.model_used}<br>Arcs: ${
        data.predicted_arc_ids ? data.predicted_arc_ids.length : "N/A"
      }<br>Decoded Continuity: ${(data.metrics.decoded_continuity * 100).toFixed(2)}%`
    );
    data.predicted_route_coords.forEach((pt) => allBoundsPoints.push(pt));
  }

  // 4. Render Raw GPS Observations (Blue circle markers)
  if (data.raw_gps && data.raw_gps.length > 0) {
    // To ensure fluid 60fps rendering, sample circle markers if over 2000 points
    const step = Math.max(1, Math.floor(data.raw_gps.length / 1500));
    for (let i = 0; i < data.raw_gps.length; i += step) {
      const pt = data.raw_gps[i];
      allBoundsPoints.push([pt.lat, pt.lon]);

      const marker = L.circleMarker([pt.lat, pt.lon], {
        radius: 3.5,
        fillColor: "#38bdf8",
        color: "#0369a1",
        weight: 1,
        opacity: 0.8,
        fillOpacity: 0.7,
      }).addTo(layerGroups.gps);

      marker.bindPopup(`
        <strong>GPS Observation #${pt.index}</strong><br>
        Lat: ${pt.lat.toFixed(6)}<br>
        Lon: ${pt.lon.toFixed(6)}<br>
        Timestamp: ${pt.timestamp.toFixed(1)}s
      `);
    }
  }

  // 5. Auto-fit Map Extent
  if (data.bounds) {
    map.fitBounds(
      [
        [data.bounds[0], data.bounds[1]],
        [data.bounds[2], data.bounds[3]],
      ],
      { padding: [40, 40] }
    );
  } else if (allBoundsPoints.length > 0) {
    map.fitBounds(L.latLngBounds(allBoundsPoints), { padding: [40, 40] });
  }
}

/** Update JSON Inspector */
function updateJsonInspector(data) {
  elements.jsonContent.textContent = JSON.stringify(data, null, 2);
}

/** Update Summary Table */
function updateSummaryTable(data) {
  const m = data.metrics;
  const tbody = document.getElementById("summaryTableBody");
  if (!tbody) return;

  tbody.innerHTML = `
    <tr><td>Trip Identifier</td><td><code>${data.trip_id}</code> (${data.partition})</td></tr>
    <tr><td>Model Architecture</td><td><strong>${data.model_used}</strong></td></tr>
    <tr><td>Raw Top-1 Accuracy</td><td>${m.top1_accuracy != null ? (m.top1_accuracy * 100).toFixed(2) + '%' : 'N/A (no ground truth)'}</td></tr>
    <tr><td>Decoded Top-1 Accuracy (Viterbi)</td><td><strong>${m.decoded_top1_accuracy != null ? (m.decoded_top1_accuracy * 100).toFixed(2) + '%' : 'N/A (no ground truth)'}</strong></td></tr>
    <tr><td>Raw Route Continuity</td><td>${m.route_continuity != null ? (m.route_continuity * 100).toFixed(2) + '%' : 'N/A'}</td></tr>
    <tr><td>Decoded Route Continuity</td><td><strong>${m.decoded_continuity != null ? (m.decoded_continuity * 100).toFixed(2) + '%' : 'N/A'}</strong></td></tr>
    <tr><td>Macro-Averaged F1 Score</td><td>${m.macro_f1 != null ? m.macro_f1.toFixed(4) : "N/A"}</td></tr>
    <tr><td>Candidate Log Loss</td><td>${m.log_loss != null ? m.log_loss.toFixed(4) : "N/A"}</td></tr>
    <tr><td>Server Inference Latency</td><td>${m.inference_time_ms != null ? m.inference_time_ms.toFixed(2) + ' ms' : 'N/A'}</td></tr>
    <tr><td>Total GPS Observations</td><td>${m.total_points != null ? m.total_points.toLocaleString() : 'N/A'}</td></tr>
    <tr><td>Unambiguous Observation Points</td><td>${m.clear_points != null ? m.clear_points.toLocaleString() : 'N/A'}</td></tr>
    <tr><td>Candidate Shortlist Misses</td><td>${m.candidate_misses != null ? m.candidate_misses : 'N/A'}</td></tr>
    <tr><td>Execution Mode</td><td><code>${data.execution_info ? data.execution_info.mode : "standard"}</code></td></tr>
  `;
}

/** Add entry to live terminal log */
function logMessage(level, text) {
  const now = new Date();
  const timeStr = now.toTimeString().split(" ")[0] + "." + String(now.getMilliseconds()).padStart(3, "0");

  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `
    <span class="log-time">[${timeStr}]</span>
    <span class="log-level ${level}">[${level}]</span>
    <span class="log-msg">${escapeHtml(text)}</span>
  `;

  elements.terminalLogs.appendChild(entry);
  elements.terminalLogs.scrollTop = elements.terminalLogs.scrollHeight;
}

function switchTab(tabName) {
  elements.tabLogsBtn.classList.toggle("active", tabName === "logs");
  elements.tabJsonBtn.classList.toggle("active", tabName === "json");
  elements.tabSummaryBtn.classList.toggle("active", tabName === "summary");

  elements.tabLogs.style.display = tabName === "logs" ? "block" : "none";
  elements.tabJson.style.display = tabName === "json" ? "block" : "none";
  elements.tabSummary.style.display = tabName === "summary" ? "block" : "none";
}

/** Export reconstructed route and points as GeoJSON */
function exportGeoJSON() {
  if (!state.lastResponse) {
    alert("No evaluation results to export. Run evaluation first.");
    return;
  }

  const d = state.lastResponse;
  const features = [];

  // Add Predicted Route LineString
  if (d.predicted_route_coords && d.predicted_route_coords.length > 0) {
    features.push({
      type: "Feature",
      properties: {
        layer: "predicted_route",
        trip_id: d.trip_id,
        model: d.model_used,
        decoded_top1: d.metrics.decoded_top1_accuracy,
        decoded_continuity: d.metrics.decoded_continuity,
      },
      geometry: {
        type: "LineString",
        // GeoJSON uses [lon, lat]
        coordinates: d.predicted_route_coords.map((pt) => [pt[1], pt[0]]),
      },
    });
  }

  // Add Ground Truth LineString
  if (d.ground_truth_coords && d.ground_truth_coords.length > 0) {
    features.push({
      type: "Feature",
      properties: {
        layer: "ground_truth",
        trip_id: d.trip_id,
      },
      geometry: {
        type: "LineString",
        coordinates: d.ground_truth_coords.map((pt) => [pt[1], pt[0]]),
      },
    });
  }

  const geoJson = {
    type: "FeatureCollection",
    properties: {
      trip_id: d.trip_id,
      exported_at: new Date().toISOString(),
      system: "PathSnap Distributed Evaluator",
    },
    features: features,
  };

  const blob = new Blob([JSON.stringify(geoJson, null, 2)], { type: "application/geo+json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `pathsnap_trip_${d.trip_id}_route.geojson`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  logMessage("SUCCESS", `Exported GeoJSON file for trip ${d.trip_id}`);
}

function escapeHtml(text) {
  const map = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" };
  return String(text).replace(/[&<>"']/g, (m) => map[m]);
}
