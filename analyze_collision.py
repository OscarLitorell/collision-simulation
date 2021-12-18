from sys import argv
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import results_analysis as ra
import json
import glob

def filter(data, axis=0):
    return signal.savgol_filter(data, 13, 1, axis=axis)

def main(name, do_print=False):
    directory = f"analyzed_results/movement/{name}"
    objects = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    group = os.path.normpath(name).split(os.path.sep)[0]

    disk_data_paths = f"qualisys_2d/{group}"

    all_disks = dict((os.path.split(file)[-1][:-5], file) for file in glob.iglob(f"{disk_data_paths}/**/*.json", recursive=True))


    collision_info = []

    accs = []
    motion = []
    obj_infos = []

    for object in objects:
        pos = np.loadtxt(f"{directory}/{object}/position.tsv", delimiter="\t")
        rot = np.loadtxt(f"{directory}/{object}/rotation.tsv", delimiter="\t")
        rot = np.unwrap(rot)
        motion.append((pos, rot))
        
        dt = np.loadtxt(f"{directory}/dt.txt", delimiter="\t")

        with open(all_disks[object]) as f:
            object_info = json.load(f)
            obj_infos.append(object_info)
        if do_print:
            print(object_info)

        pos_f = filter(pos, axis=0)
        vel = np.gradient(pos_f, axis=0) / dt
        vel_f = filter(vel, axis=0)
        acc = np.gradient(vel_f, axis=0) / dt
        acc_f = filter(acc, axis=0)
        acc_mag_sq = np.sum(acc_f**2, axis=1)

        accs.append(acc_mag_sq)
        coll = (acc_mag_sq > 1).astype(float)
        cd = np.diff(coll).clip(0, 1)

        hits = np.sum(cd)
        if hits != 1 and do_print:
            plt.plot(np.sqrt(acc_mag_sq))
            plt.title(f"{name} {object}")
            print(name)
            plt.show()


    max_acc_index = np.argmax(accs[0] * accs[1])
    
    for i, object in enumerate(objects):
        pos, rot = motion[i]
        object_info = obj_infos[i]

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
            "mass": object_info["mass"],
            "radius": object_info["radius"],
            "height": object_info["height"]
        }
        # Eventuellt problem:
        # Svårt att veta exakta hastigheter precis före och efter
        # när man filtrerar. Bäst resultat verkar ges av 1:a 
        # ordningens Savitsky-Golay-filter.

        collision_info.append(info)
    
    normal = collision_info[0]["pos"] - collision_info[1]["pos"]
    normal = normal / np.linalg.norm(normal)
    tangent = np.array([-normal[1], normal[0]])
    rel_tangent_vel_before = np.dot(collision_info[0]["vel"][0] - collision_info[1]["vel"][0], tangent)
    rel_tangent_vel_before += collision_info[0]["radius"] * collision_info[0]["rot_vel"][0]
    rel_tangent_vel_before -= collision_info[1]["radius"] * collision_info[1]["rot_vel"][0]
    rel_tangent_vel_after = np.dot(collision_info[0]["vel"][1] - collision_info[1]["vel"][1], tangent)
    rel_tangent_vel_after += collision_info[0]["radius"] * collision_info[0]["rot_vel"][1]
    rel_tangent_vel_after -= collision_info[1]["radius"] * collision_info[1]["rot_vel"][1]
    rel_normal_vel_before = abs(np.dot(collision_info[0]["vel"][0] - collision_info[1]["vel"][0], normal))
    rel_normal_vel_after = abs(np.dot(collision_info[0]["vel"][1] - collision_info[1]["vel"][1], normal))

    momentum_0_before = collision_info[0]["mass"] * collision_info[0]["vel"][0]
    momentum_1_before = collision_info[1]["mass"] * collision_info[1]["vel"][0]
    momentum_0_after = collision_info[0]["mass"] * collision_info[0]["vel"][1]
    momentum_1_after = collision_info[1]["mass"] * collision_info[1]["vel"][1]
    momentum_before = momentum_0_before + momentum_1_before
    momentum_after = momentum_0_after + momentum_1_after
    impulse_0 = momentum_0_before - momentum_0_after
    impulse_1 = momentum_1_before - momentum_1_after
    impulse = (impulse_0 - impulse_1) / 2
    normal_impulse = np.dot(impulse, normal)
    tangent_impulse = np.dot(impulse, tangent)
    fric_coeff = abs(tangent_impulse / normal_impulse)

    rel_normal_vel_before = np.dot(collision_info[0]["vel"][0] - collision_info[1]["vel"][0], normal)
    rel_normal_vel_after = np.dot(collision_info[0]["vel"][1] - collision_info[1]["vel"][1], normal)

    
    e = abs(rel_normal_vel_after / rel_normal_vel_before)

    moment_of_inertia_0 = abs(tangent_impulse * collision_info[0]["radius"] / (collision_info[0]["rot_vel"][1] - collision_info[0]["rot_vel"][0]))
    moment_of_inertia_1 = abs(tangent_impulse * collision_info[1]["radius"] / (collision_info[1]["rot_vel"][1] - collision_info[1]["rot_vel"][0]))

    if do_print:
        print("Relative tangent velocity before:", rel_tangent_vel_before)
        print("Relative tangent velocity after:", rel_tangent_vel_after)
        print("Momentum before:", momentum_before)
        print("Momentum after:", momentum_after)
        print("Normal impulse:", normal_impulse)
        print("Tangent impulse:", tangent_impulse)
        print("Friction coefficient:", fric_coeff)
        print("Relative normal velocity before:", rel_normal_vel_before)
        print("Relative normal velocity after:", rel_normal_vel_after)
        print("Elasiticity coefficient:", e)

    collision_properties = {
        "name": name,
        "pos_0_x": collision_info[0]["pos"][0],
        "pos_0_y": collision_info[0]["pos"][1],
        "pos_0_x": collision_info[1]["pos"][0],
        "pos_0_y": collision_info[1]["pos"][1],
        "vel_0_before_x": collision_info[0]["vel"][0][0],
        "vel_0_before_y": collision_info[0]["vel"][0][1],
        "vel_1_before_x": collision_info[1]["vel"][0][0],
        "vel_1_before_y": collision_info[1]["vel"][0][1],
        "vel_0_after_x": collision_info[0]["vel"][1][0],
        "vel_0_after_y": collision_info[0]["vel"][1][1],
        "vel_1_after_x": collision_info[1]["vel"][1][0],
        "vel_1_after_y": collision_info[1]["vel"][1][1],
        "rot_vel_0_before": collision_info[0]["rot_vel"][0],
        "rot_vel_1_before": collision_info[1]["rot_vel"][0],
        "rot_vel_0_after": collision_info[0]["rot_vel"][1],
        "rot_vel_1_after": collision_info[1]["rot_vel"][1],
        "mass_0": collision_info[0]["mass"],
        "mass_1": collision_info[1]["mass"],
        "radius_0": collision_info[0]["radius"],
        "radius_1": collision_info[1]["radius"],
        "height_0": collision_info[0]["height"],
        "height_1": collision_info[1]["height"],
        "rel_tangent_vel_before": rel_tangent_vel_before,
        "rel_tangent_vel_after": rel_tangent_vel_after,
        "rel_normal_vel_before": rel_normal_vel_before,
        "rel_normal_vel_after": rel_normal_vel_after,
        "normal_impulse": normal_impulse,
        "tangent_impulse": tangent_impulse,
        "fric_coeff": fric_coeff,
        "e": e,
        "moment_of_inertia_0": moment_of_inertia_0,
        "moment_of_inertia_1": moment_of_inertia_1,
    }

    path = os.path.join("analyzed_collisions", os.path.split(name)[0])
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f"analyzed_collisions/{name}.json", "w") as f:
        json.dump(collision_properties, f)
   



if __name__ == '__main__':
    try:
        name = argv[1]
        main(name, "-p" in argv)
    except IndexError:
        print("Please specify an experiment name")
        exit(1)
