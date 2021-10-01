
import os
from sys import argv
import matplotlib.pyplot as plt
import results_analysis as ra
import scipy.signal as signal
import numpy as np



def main(simulation_name, plot=True):
    results_path = f"results/{simulation_name}"
    constellations_path = "analyzed_results/constellations"

    (markers, tspan, dt) = ra.load_results(results_path)
    
    objects = ra.separate_connected(markers)
    
    files_in_dir = os.listdir(constellations_path)

    constellations = [ra.load_constellation(constellations_path + "/" + file) for file in files_in_dir if file.endswith('.tsv')]

    object_movements = [] # (position, rotation, constellation_index)

    for obj in objects:
        center, rotation, constellation_index = ra.get_center_and_rotation(obj, constellations)

        object_movements.append((center, rotation, constellation_index))

        p1 = obj[:,:,0]
        p2 = obj[:,:,1]
        p3 = obj[:,:,2]

        if plot:    
            plt.plot(center[:,0], center[:,1])
            plt.plot(p1[:,0], p1[:,1])
            plt.plot(p2[:,0], p2[:,1])
            plt.plot(p3[:,0], p3[:,1])

    if plot:
        plt.axis('equal')
        plt.show()
    
    for position, rotation, constellation_index in object_movements:
        save_position_and_rotation(position, rotation, simulation_name, constellation_index)



def save_position_and_rotation(position, rotation, simulation_name, object_index):
    directory = f"analyzed_results/movement/{simulation_name}/object_{object_index}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savetxt(f"{directory}/position.tsv", position[:,:,0], delimiter="\t")
    np.savetxt(f"{directory}/rotation.tsv", rotation[:,0], delimiter="\t")
    
    


    

if __name__ == "__main__":
    try:
        path = argv[1]
        plot = "-p" in argv
        main(path, plot)
    except IndexError:
        print("Please specify an experiment name")
        exit(1)

