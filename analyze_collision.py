
import os
from sys import argv
import matplotlib.pyplot as plt
import results_analysis as ra
import scipy.signal as signal



def main(plot=True):
    results_path = "results/collision"
    constellations_path = "analyzed_results/objects"

    (markers, tspan, dt) = ra.load_results(results_path)
    
    objects = ra.separate_connected(markers)
    
    files_in_dir = os.listdir(constellations_path)

    constellations = [ra.load_constellation(constellations_path + "/" + file) for file in files_in_dir if file.endswith('.tsv')]

    positions = []
    rotations = []

    for obj in objects:
        # constellation = ra.match_marker_constellation(obj, constellations)
        center, rotation = ra.get_center_and_rotation(obj, constellations)

        positions.append(center)
        rotations.append(rotation)

        p1 = obj[0]
        p2 = obj[1]
        p3 = obj[2]


        if plot:    
            plt.plot(center[:,0], center[:,1])
            plt.plot(p1[0], p1[1])
            plt.plot(p2[0], p2[1])
            plt.plot(p3[0], p3[1])

    if plot:
        plt.axis('equal')
        plt.show()




    

if __name__ == "__main__":
    main()


