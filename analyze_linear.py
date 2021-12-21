import os
import numpy as np

import analyze_collision
import scipy.signal as signal
import matplotlib.pyplot as plt


def convert_qualisys(files_path):

    groups = os.listdir(files_path)

    for group in groups:
        group_path = os.path.join(files_path, group)
        files = [file for file in os.listdir(group_path) if file.endswith(".tsv")]
        for file in files:
            with open(os.path.join(group_path, file), "r") as text:
                lines = text.readlines()
                lines = [line[:-1].split("\t") for line in lines]

                metadata_list = lines[:10]
                metadata = {}
                for line in metadata_list:
                    key = line[0].lower()
                    value = line[1:]
                    metadata[key] = value

                header = lines[10]
                data_text = lines[11:]
                data = np.array(data_text).astype(float)
                marker_count = (len(header) - 3) / 3
                marker_names = metadata["marker_names"]
                timestamps = data[:,1]
                marker_trajectories = {}
                marker_ranges = []
                for i, marker in enumerate(marker_names):
                    if len(marker) == 0:
                        continue
                    marker_trajectory = data[:,2+3*i : 2+3*i+3]
                    marker_trajectories[marker] = marker_trajectory
                    marker_range = np.any(marker_trajectory != 0, axis=1)
                    marker_ranges.append(marker_range)

                marker_ranges = np.all(np.array(marker_ranges), axis=0)
                marker_trajectories = {k: v[marker_ranges] for k, v in marker_trajectories.items()}
                timestamps = timestamps[marker_ranges]
                if not os.path.exists(f"results_linear/{group}/{file[:-4]}"):
                    os.makedirs(f"results_linear/{group}/{file[:-4]}")
                for marker, trajectory in marker_trajectories.items():
                    data = np.hstack((timestamps.reshape(len(timestamps), 1), trajectory))
                    np.savetxt(f"results_linear/{group}/{file[:-4]}/{marker}.tsv", data, delimiter="\t")
                

def filter(data, axis=0):
    return signal.savgol_filter(data, 21, 1, axis=axis)


def main():
    convert_qualisys("qualisys_linear")

    g = []
    name = []
    mass_0 = []
    mass_1 = []
    v_0_before = []
    v_1_before = []
    v_0_after = []
    v_1_after = []


    groups = os.listdir("results_linear")
    for group in groups:

        subdirs = os.listdir(os.path.join("results_linear", group))
        for subdir in subdirs:
            dir = os.path.join(group, subdir)
            markers = os.listdir(f"results_linear/{dir}")
            
            objects = []

            for marker in markers:
                data = np.loadtxt(f"results_linear/{dir}/{marker}", delimiter="\t")
                timestamps = data[:,0]
                pos = data[:,1:] * 0.001

                dt = timestamps[1] - timestamps[0]

                mass = float(marker[:-4]) * 0.001

                pos_f = filter(pos, axis=0)
                vel = np.gradient(pos_f, axis=0) / dt
                vel_f = filter(vel, axis=0)
                acc = np.gradient(vel_f, axis=0) / dt
                acc_f = filter(acc, axis=0)
                acc_mag_sq = np.sum(acc_f**2, axis=1)
                max_acc_index = np.argmax(acc_mag_sq)

                pos_before = pos[:max_acc_index]
                pos_after = pos[max_acc_index:]
                pos_before_f = filter(pos_before, axis=0)
                pos_after_f = filter(pos_after, axis=0)
                vel_before = np.gradient(pos_before_f, axis=0) / dt
                vel_after = np.gradient(pos_after_f, axis=0) / dt
                vel_before_f = filter(vel_before, axis=0)
                vel_after_f = filter(vel_after, axis=0)
                objects.append({
                    "v0": vel_before_f[-1],
                    "v1": vel_after_f[0],
                    "m": mass
                })

            g.append(group)
            name.append(dir)
            mass_0.append(objects[0]["m"])
            mass_1.append(objects[1]["m"])
            v_0_before.append(objects[0]["v0"])
            v_1_before.append(objects[1]["v0"])
            v_0_after.append(objects[0]["v1"])
            v_1_after.append(objects[1]["v1"])
    
    g = np.array(g)
    name = np.array(name)
    mass_0 = np.array(mass_0)
    mass_1 = np.array(mass_1)
    v_0_before = np.array(v_0_before)
    v_1_before = np.array(v_1_before)
    v_0_after = np.array(v_0_after)
    v_1_after = np.array(v_1_after)

    d_v_before = np.linalg.norm(v_0_before - v_1_before, axis=1)
    d_v_after = np.linalg.norm(v_0_after - v_1_after, axis=1)
    e = d_v_after / d_v_before

    momentum_0_before = mass_0.reshape((len(mass_0), 1)) * v_0_before
    momentum_1_before = mass_1.reshape((len(mass_1), 1)) * v_1_before
    momentum_0_after = mass_0.reshape((len(mass_0), 1)) * v_0_after
    momentum_1_after = mass_1.reshape((len(mass_1), 1)) * v_1_after
    
    total_momentum_before = momentum_0_before + momentum_1_before
    total_momentum_after = momentum_0_after + momentum_1_after

    rel_momentum_before = momentum_1_before - momentum_0_before
    rel_momentum_after = momentum_1_after - momentum_0_after

    momentum_diff = total_momentum_after - total_momentum_before
    normalized_momentum_diff = momentum_diff #/ np.linalg.norm(rel_momentum_after - rel_momentum_before, axis=1).reshape((momentum_diff.shape[0],1)) * 2
    abs_normalized_momentum_diff = np.linalg.norm(normalized_momentum_diff, axis=1)


    plt.grid()
    plt.scatter(g, abs_normalized_momentum_diff)
    plt.xlabel("Diskar")
    plt.ylabel("Relativ förändring av total rörelsemängd")
    plt.show()

    plt.scatter(g, e)
    plt.xlabel("Diskar")
    plt.ylabel("Elasticitetskoefficient")
    plt.title("Elasticitetskoefficient")
    plt.show()


    # for group in groups:
    #     mask = g == group
    #     print(group)

    #     x = d_v_before[mask]
    #     y = e[mask]

    #     plt.plot(x, y, "o")
    #     plt.xlabel("d_v_before")
    #     plt.ylabel("e")
    #     plt.show()


    input("Press enter to continue")




if __name__ == "__main__":
    main()


