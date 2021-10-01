
import os
from sys import argv
import matplotlib.pyplot as plt
import results_analysis as ra


def main():
    results_path = "results/collision"
    constellations_path = "analyzed_results/objects"

    (markers, tspan, dt) = ra.load_results(results_path)
    
    objects = ra.separate_connected(markers)
    
    files_in_dir = os.listdir(constellations_path)

    constellations = [ra.load_constellation(constellations_path + "/" + file) for file in files_in_dir if file.endswith('.tsv')]

    positions = []
    rotations = []

    for obj in objects:
        constellation = ra.match_marker_constellation(obj, constellations)
        center, rotation = ra.get_center_and_rotation(obj, constellation)

        positions.append(center)
        rotations.append(rotation)



    

if __name__ == "__main__":
    main()


