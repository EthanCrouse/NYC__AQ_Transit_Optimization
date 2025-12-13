from best_path_helper import compute_path, describe_route, print_route_stats
from datetime import datetime

if __name__ == "__main__":
    now = datetime.now()
    month = now.month - 1
    year = now.year

    month = 3
    year = 2025

    start, dest = "Franklin St", "72 St"
    path, G = compute_path(month, year, start, dest)
    print("Clean Air Path:")

    if path is None:
        print("Origin or Destination are in Bad Air Quality Zones or no safe route exists.")
    else:
        print("Air-aware path (stations):")
        for sid, name in path:
            print(f"  {sid}: {name}")

        print("\nDirections:")
        directions = describe_route(G, path)
        for step in directions:
            print(" -", step)
        print_route_stats(G, path)
        