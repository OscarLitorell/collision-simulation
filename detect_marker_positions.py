
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
    puck_spins = [
        "spin1",
        "spin2"
    ]

    constellations = []

    for spin in puck_spins:
        constellations += detect_marker_positions(f"results/{spin}")

    for i, constellation in enumerate(constellations):
        ra.save_constellation(constellation, f"analyzed_results/objects/{i}.tsv")



if __name__ == "__main__":
    main()






