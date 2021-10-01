from sys import argv
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import results_analysis as ra


def filter(data, axis=0):
    return signal.savgol_filter(data, 51, 1, axis=axis)

def main(name):
    directory = f"analyzed_results/movement/{name}"
    objects = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    collision_info = []

    for object in objects:
        pos = np.loadtxt(f"{directory}/{object}/position.tsv", delimiter="\t")
        rot = np.loadtxt(f"{directory}/{object}/rotation.tsv", delimiter="\t")
        rot = np.unwrap(rot)
        dt = 0.01 # Needs a real input

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

        rot_before = rot[:max_acc_index]
        rot_after = rot[max_acc_index:]
        rot_before_f = filter(rot_before, axis=0)
        rot_after_f = filter(rot_after, axis=0)
        rot_vel_before = np.gradient(rot_before_f, axis=0) / dt
        rot_vel_after = np.gradient(rot_after_f, axis=0) / dt
        rot_vel_before_f = filter(rot_vel_before, axis=0)
        rot_vel_after_f = filter(rot_vel_after, axis=0)


        collision_position = pos[max_acc_index]
        collision_rotation = rot[max_acc_index]

        info = {
            "pos": collision_position,
            "vel": (vel_before_f[-1], vel_after_f[0]),
            "rot_vel": (rot_vel_before_f[-1], rot_vel_after_f[0]),
        }
        # Eventuellt problem:
        # Svårt att veta exakta hastigheter precis före och efter
        # när man filtrerar. Bäst resultat verkar ges av 1:a 
        # ordningens Savitsky-Golay-filter.

        collision_info.append(info)

    print(collision_info)


if __name__ == '__main__':
    try:
        name = argv[1]
        main(name)
    except IndexError:
        print("Please specify an experiment name")
        exit(1)
