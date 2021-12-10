
import os
from sys import argv
import results_analysis as ra


def detect_marker_positions(results_path):
    (markers, tspan, dt) = ra.load_results(results_path)

    objects = ra.separate_connected(markers)
    
    constellations = []
    
    for obj in objects:
        center = ra.locate_center(obj)
        marker_constellation = ra.get_marker_constellation(obj, center)
        constellations.append(marker_constellation)
    return constellations


def main():
    dirs = os.walk("results")
    dirs = [d for d in dirs if len(d[2]) > 0 and "spins" in d[0]]
    
    constellations = []

    for d in dirs:
        constellations += detect_marker_positions(d[0])

    for i, constellation in enumerate(constellations):
        p = dirs[i][0]
        path = dirs[i][0][len("results\\"):]
        ra.save_constellation(constellation, f"analyzed_results/constellations/{path}.tsv")



if __name__ == "__main__":
    main()






