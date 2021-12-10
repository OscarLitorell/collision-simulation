
import os
from sys import argv
import matplotlib.pyplot as plt
import results_analysis as ra
import scipy.signal as signal
import numpy as np
import glob


def main(simulation_name, plot=True):
    results_path = f"results/{simulation_name}"
    group_name = os.path.normpath(simulation_name).split(os.path.sep)[0]
    constellations_path = f"analyzed_results/constellations/{group_name}"

    (markers, tspan, dt) = ra.load_results(results_path)

    print("Grouping markers...")    
    objects = ra.separate_connected(markers)
    print("Found {} groups".format(len(objects)))

    files_in_dir = [file for file in glob.iglob(f"{constellations_path}/**/*.tsv", recursive=True)]
    # files_in_dir = [[os.path.join(d[0], f) for f in d[2]] for d in os.walk(constellations_path)]
    # files_in_dir = [file for file in os.listdir(constellations_path) if file.endswith(".tsv")]

    constellations = [ra.load_constellation(file) for file in files_in_dir]

    object_movements = [] # (position, rotation, constellation_index)

    for obj in objects:
        center, rotation, constellation_index = ra.get_center_and_rotation(obj, constellations)

        object_movements.append((center, rotation, constellation_index))

        marker_count = obj.shape[2]

        if plot:    
            plt.plot(center[:,0], center[:,1])

            for i in range(marker_count):
                p = obj[:,:,i]
                plt.plot(p[:,0], p[:,1], alpha=0.3)

    if plot:
        plt.axis('equal')
        plt.show()
    
    for position, rotation, constellation_index in object_movements:
        save_position_and_rotation(position, rotation, simulation_name, os.path.split(files_in_dir[constellation_index][:-4])[-1])
        save_time_data(dt, simulation_name)

def save_time_data(dt, simulation_name):
    directory = os.path.join("analyzed_results", "movement", simulation_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(f"{directory}/dt.txt", "w") as fw:
        fw.write(str(dt))

def save_position_and_rotation(position, rotation, simulation_name, object_name):
    directory = os.path.join("analyzed_results", "movement", simulation_name, object_name)
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

