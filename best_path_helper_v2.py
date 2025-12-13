# air_aware_routing.py
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import math
import os
from live_aq_pred import predict_current, grab_recent_measurements




# ---------- helpers ----------

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(PROJECT_ROOT, "code", "data")

def gtfs_path(filename):
    return os.path.join(DATA_DIR, "gtfs", filename)

'''
determine distance from start to end to be used as weights
'''
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (math.sin(dphi/2)**2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

'''
builds the graph of the subways
return a networkx graph of network
'''
def build_subway_graph(
    routes_path=None,
    trips_path=None,
    stops_path=None,
    stop_times_path=None,
):
    if routes_path is None:
        routes_path = gtfs_path("routes.txt")
    if trips_path is None:
        trips_path = gtfs_path("trips.txt")
    if stops_path is None:
        stops_path = gtfs_path("stops.txt")
    if stop_times_path is None:
        stop_times_path = gtfs_path("stop_times.txt")

    routes = pd.read_csv(routes_path)
    trips = pd.read_csv(trips_path)
    stops = pd.read_csv(stops_path)
    stop_times = pd.read_csv(stop_times_path)

    # 1. Filter to SUBWAY ROUTES ONLY: route_type == 1 (NYC subway)
    subway_routes = routes[routes["route_type"] == 1]["route_id"]

    # 2. Filter trips to subway trips
    subway_trips = trips[trips["route_id"].isin(subway_routes)]["trip_id"]

    # 3. Filter stop_times to subway stop_times
    st = stop_times[stop_times["trip_id"].isin(subway_trips)].copy()

    # 4. Keep only stops referenced by subway stop_times
    used_stop_ids = st["stop_id"].unique()
    stops = stops[stops["stop_id"].isin(used_stop_ids)][["stop_id", "stop_name", "stop_lat", "stop_lon"]]

    # 5. Merge st with route_id for graph building
    st = st.merge(trips[["trip_id", "route_id"]], on="trip_id", how="left")

    # --- NEW: figure out which routes serve each stop_id ---
    # map route_id -> route_short_name (or fallback to route_id)
    if "route_short_name" in routes.columns:
        rid_to_short = dict(zip(routes["route_id"], routes["route_short_name"]))
    else:
        rid_to_short = dict(zip(routes["route_id"], routes["route_id"]))

    # for each stop_id, what route_ids serve it?
    stop_to_route_ids = (
        st.groupby("stop_id")["route_id"]
        .apply(lambda s: set(s.dropna()))
        .to_dict()
    )

    # build a display name: "{stop_name} ({line1/line2/...})"
    display_names = []
    for _, row in stops.iterrows():
        sid = row["stop_id"]
        base_name = row["stop_name"]
        route_ids = stop_to_route_ids.get(sid, set())

        line_names = sorted({str(rid_to_short.get(rid, rid)) for rid in route_ids})
        if line_names:
            disp = f"{base_name} ({'/'.join(line_names)})"
        else:
            disp = base_name

        display_names.append(disp)

    # after you compute display_names
    stops = stops.copy()
    stops["base_name"] = stops["stop_name"]      # raw GTFS name
    stops["display_name"] = display_names        # e.g., "86 St (4/5/6)"

    # -------------------------------------------------------

    G = nx.Graph()

    # 6. Add nodes (use display_name instead of raw stop_name)
    for _, row in stops.iterrows():
        G.add_node(
            row["stop_id"],
            name=row["display_name"],  # <-- this is what shows up in describe_route
            base_name=row["base_name"],
            lat=row["stop_lat"],
            lon=row["stop_lon"]
        )

    # 7. Add intra-station edges to connect N/S platforms
    # (still group by raw stop_name so complexes are linked as before)
    add_intra_station_edges(G, stops)

    # 8. Add edges between consecutive stops in each trip
    grouped = st.sort_values(["trip_id", "stop_sequence"]).groupby("trip_id")

    for trip_id, g in grouped:
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            s1 = g.loc[i, "stop_id"]
            s2 = g.loc[i + 1, "stop_id"]

            if s1 not in G.nodes or s2 not in G.nodes:
                continue

            lat1, lon1 = G.nodes[s1]["lat"], G.nodes[s1]["lon"]
            lat2, lon2 = G.nodes[s2]["lat"], G.nodes[s2]["lon"]
            dist_km = haversine(lat1, lon1, lat2, lon2)

            route_id = g.loc[i, "route_id"]

            if G.has_edge(s1, s2):
                G[s1][s2]["routes"].add(route_id)
                G[s1][s2]["weight"] = min(G[s1][s2]["weight"], dist_km)
            else:
                G.add_edge(s1, s2, weight=dist_km, routes={route_id})

    return G, stops


'''
connect station stops and ends
'''
def add_intra_station_edges(G, stops_df):
    # use base_name if available, otherwise stop_name
    group_col = "base_name" if "base_name" in stops_df.columns else "stop_name"

    groups = stops_df.groupby(group_col)["stop_id"].apply(list)

    for _, ids in groups.items():
        if len(ids) > 1:
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    G.add_edge(
                        ids[i],
                        ids[j],
                        weight=0.00001,
                        routes={"transfer"},
                    )



"""
    For each subway station, find the nearest air-quality sensor.
"""
def assign_stations_to_sensors(stops_df, sensors_df,
                               sensor_lat_col="Latitude",
                               sensor_lon_col="Longitude"):
    # Keep only sensors with valid coordinates
    sensors = sensors_df.dropna(subset=[sensor_lat_col, sensor_lon_col]).copy()

    sensor_records = sensors[["SiteName", sensor_lat_col, sensor_lon_col]].to_dict("records")

    cliques = {rec["SiteName"]: [] for rec in sensor_records}
    nearest_site_list = []
    nearest_dist_list = []

    for _, stop in stops_df.iterrows():
        s_lat = stop["stop_lat"]
        s_lon = stop["stop_lon"]

        best_site = None
        best_dist = float("inf")

        for rec in sensor_records:
            d = haversine(s_lat, s_lon, rec[sensor_lat_col], rec[sensor_lon_col])
            if d < best_dist:
                best_dist = d
                best_site = rec["SiteName"]

        nearest_site_list.append(best_site)
        nearest_dist_list.append(best_dist)

        if best_site is not None:
            cliques[best_site].append(stop["stop_id"])

    stops_with_assignment = stops_df.copy()
    stops_with_assignment["nearest_site"] = nearest_site_list
    stops_with_assignment["nearest_dist_km"] = nearest_dist_list

    return cliques, stops_with_assignment

"""
    - Assign each station to nearest AQ sensor
    - If that sensor's BadAir == 1, remove all its stations from the network.
      Optionally only remove stations within max_radius_km of the sensor.
"""
def prune_graph_by_air_quality(G, stops_df, sensors_df,
                               bad_col="BadAir",
                               sensor_lat_col="Latitude",
                               sensor_lon_col="Longitude",
                               max_radius_km=None):
    cliques, stops_assigned = assign_stations_to_sensors(
        stops_df,
        sensors_df,
        sensor_lat_col=sensor_lat_col,
        sensor_lon_col=sensor_lon_col,
    )

    # Which sensors show bad air
    bad_sites = set(sensors_df[sensors_df[bad_col] == 1]["SiteName"])

    bad_stop_ids = set()

    for site in bad_sites:
        if site not in cliques:
            continue
        for sid in cliques[site]:
            if max_radius_km is not None:
                dist = float(
                    stops_assigned.loc[stops_assigned["stop_id"] == sid, "nearest_dist_km"].iloc[0]
                )
                if dist > max_radius_km:
                    continue
            bad_stop_ids.add(sid)

    G_clean = G.copy()
    G_clean.remove_nodes_from(bad_stop_ids)

    return G_clean, bad_stop_ids, cliques, stops_assigned


"""
    Find a stop_id given a (partial) stop name.
    Returns the first match.
"""
def find_stop_id_by_name(stops_df, partial_name):
    col = "display_name" if "display_name" in stops_df.columns else "stop_name"
    mask = stops_df[col].str.contains(partial_name, case=False, na=False)
    matches = stops_df[mask]
    if matches.empty:
        raise ValueError(f"No station found matching '{partial_name}'")
    return matches.iloc[0]["stop_id"]



def describe_route(G, path, routes_path=None):
    """
    Given:
      - G: subway graph with edge attribute 'routes'
      - path: list of (stop_id, stop_name)

    Returns human-readable transfer instructions.
    """
    if not path or len(path) < 2:
        return []

    # Load route_id -> route_short_name mapping
    if routes_path is None:
        routes_path = gtfs_path("routes.txt")
    routes_df = pd.read_csv(routes_path)

    if "route_short_name" in routes_df.columns:
        id_to_name = dict(zip(routes_df["route_id"], routes_df["route_short_name"]))
    else:
        id_to_name = dict(zip(routes_df["route_id"], routes_df["route_id"]))

    # Extract ids + names
    path_ids = [sid for sid, _ in path]
    path_names = {sid: name for sid, name in path}

    # Get routes for each edge
    edge_routes = []
    for i in range(len(path_ids) - 1):
        u = path_ids[i]
        v = path_ids[i + 1]
        data = G.get_edge_data(u, v, default={})
        routes = data.get("routes", set())
        edge_routes.append(set(routes))

    # Build segments
    segments = []
    current_routes = None
    seg_start_idx = 0

    for i in range(len(edge_routes)):
        er = edge_routes[i]

        if current_routes is None:
            current_routes = set(er)
            seg_start_idx = i
        else:
            common = current_routes & er
            if common:
                current_routes = common
            else:
                # End previous segment at THIS edge's starting node (i)
                segments.append((seg_start_idx, i, current_routes.copy()))
                current_routes = set(er)
                seg_start_idx = i

    # Close final segment safely — end at the last node
    segments.append((seg_start_idx, len(path_ids) - 1, current_routes.copy()))

    # Convert segments into directions
    directions = []
    for seg_start_idx, seg_end_idx, routes in segments:

        start_stop_id = path_ids[seg_start_idx]
        end_stop_id   = path_ids[seg_end_idx]    # <-- FIXED LINE

        start_name = path_names[start_stop_id]
        end_name   = path_names[end_stop_id]

        num_stops = seg_end_idx - seg_start_idx

        if not routes:
            route_label = "Unknown route"
        else:
            route_names = sorted(id_to_name.get(rid, rid) for rid in routes)
            route_label = "/".join(route_names)

        directions.append(
            f"Take {route_label} from {start_name} to {end_name} ({num_stops} stops)."
        )

    return directions


def compute_shortest_path(
    G,
    stops_df,
    origin_name,
    dest_name,
    distance_weight="weight",
    transfer_penalty_km=1000.0,
):
    # Look up stop_ids
    origin_id = find_stop_id_by_name(stops_df, origin_name)
    dest_id   = find_stop_id_by_name(stops_df, dest_name)

    # Check if either has been removed from the graph
    if origin_id not in G.nodes:
        print(f"Origin station '{origin_name}' was removed (bad air).")
        return None

    if dest_id not in G.nodes:
        print(f"Destination station '{dest_name}' was removed (bad air).")
        return None

    # Custom edge weight: distance + big penalty for transfers
    def edge_weight(u, v, attrs):
        base = float(attrs.get(distance_weight, 0.0))
        routes = attrs.get("routes", set()) or set()

        # Any intra-station "transfer" edge gets a huge penalty
        if "transfer" in routes:
            return base + transfer_penalty_km
        return base

    try:
        path_ids = nx.shortest_path(
            G,
            source=origin_id,
            target=dest_id,
            weight=edge_weight,
        )
    except nx.NetworkXNoPath:
        print("No valid route available due to bad-air pruning.")
        return None

    # Convert stop IDs to human-friendly names
    path = [(sid, G.nodes[sid]["name"]) for sid in path_ids]
    return path


def print_route_stats(G, path):
    """
    Prints:
      - Number of stations traveled
      - Number of line transfers
    """
    if not path or len(path) < 2:
        print("No route available.")
        return

    # Number of stations traveled = number of nodes in path
    num_stations = len(path)

    # Extract route sets for each edge
    path_ids = [sid for sid, _ in path]
    edge_routes = []

    for i in range(len(path_ids) - 1):
        u = path_ids[i]
        v = path_ids[i + 1]
        data = G.get_edge_data(u, v, default={})
        edge_routes.append(set(data.get("routes", set())))

    # Count transfers (route set changes)
    transfers = 0
    current_routes = edge_routes[0]

    for r in edge_routes[1:]:
        if not (current_routes & r):  # no overlap → transfer
            transfers += 1
            current_routes = r
        else:
            current_routes = current_routes & r

    print(f"Stations traveled: {num_stations}")
    print(f"Transfers: {transfers}")

######################################################
#########       Compute Path Method          #########     
######################################################

'''
1) grab the networkx of the subway network
2) grab labels of the air sensors and predict the current PM2.5 measurements
3) prune network and remove stations that are within 0.5km of measurements that are too high
4) compute shortest path from start to stop on pruned network
'''
def compute_path(month, year, start_name, dest_name):
    G_rough, stops = build_subway_graph()

    labels = pd.read_csv(os.path.join(DATA_DIR, "AQ_2024", "labels.csv"))
    pred = predict_current(month, year)
    aq_sensors = pred.merge(
        labels[["SiteName", "Latitude", "Longitude"]],
        on="SiteName",
        how="left"
    )

    G, bad_stop_ids, cliques, stops_assigned = prune_graph_by_air_quality(
        G_rough,
        stops,
        aq_sensors,
        bad_col="BadAir",
        sensor_lat_col="Latitude",
        sensor_lon_col="Longitude",
        max_radius_km=0.5
    )

    # Uses the new transfer-penalized shortest path
    path = compute_shortest_path(G, stops, start_name, dest_name)
    return path, G

    

    



