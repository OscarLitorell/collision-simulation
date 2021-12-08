import os
import numpy as np

import analyze_collision
import scipy.signal as signal


def convert_qualisys(files_path):
    
    files = [file for file in os.listdir(files_path) if file.endswith(".tsv")]
    for file in files:
        if not os.path.exists(f"results_linear/{file[:-4]}"):
            os.mkdir(f"results_linear/{file[:-4]}")
        with open(f"{files_path}/{file}", "r") as text:
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
            data = np.array(data_text).astype(np.float)
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
            for marker, trajectory in marker_trajectories.items():
                data = np.hstack((timestamps.reshape(len(timestamps), 1), trajectory))
                np.savetxt(f"results_linear/{file[:-4]}/{marker}.tsv", data, delimiter="\t")
            

def filter(data, axis=0):
    return signal.savgol_filter(data, 21, 1, axis=axis)


def main():
    convert_qualisys("qualisys_linear")

    dirs = os.listdir("results_linear")
    for dir in dirs:
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
        d_v0 = np.linalg.norm(objects[0]["v0"] - objects[1]["v0"])
        d_v1 = np.linalg.norm(objects[0]["v1"] - objects[1]["v1"])
        e = d_v1 / d_v0
        print(d_v0, d_v1, e)



if __name__ == "__main__":
    main()


