

import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    with open("analyzed_collisions.tsv", "r") as f:
        lines = f.readlines()
    lines = [line.strip().split("\t") for line in lines]
    header = lines[0]
    lines = lines[1:]

    rows = np.array(lines)
    cols = rows.T.tolist()


    name = np.array(cols[header.index("name")])
    pos_0_x = np.array(cols[header.index("pos_0_x")]).astype(float)
    pos_0_y = np.array(cols[header.index("pos_0_y")]).astype(float)
    pos_1_x = np.array(cols[header.index("pos_1_x")]).astype(float)
    pos_1_y = np.array(cols[header.index("pos_1_y")]).astype(float)
    vel_0_before_x = np.array(cols[header.index("vel_0_before_x")]).astype(float)
    vel_0_before_y = np.array(cols[header.index("vel_0_before_y")]).astype(float)
    vel_1_before_x = np.array(cols[header.index("vel_1_before_x")]).astype(float)
    vel_1_before_y = np.array(cols[header.index("vel_1_before_y")]).astype(float)
    vel_0_after_x = np.array(cols[header.index("vel_0_after_x")]).astype(float)
    vel_0_after_y = np.array(cols[header.index("vel_0_after_y")]).astype(float)
    vel_1_after_x = np.array(cols[header.index("vel_1_after_x")]).astype(float)
    vel_1_after_y = np.array(cols[header.index("vel_1_after_y")]).astype(float)
    rot_vel_0_before = np.array(cols[header.index("rot_vel_0_before")]).astype(float)
    rot_vel_1_before = np.array(cols[header.index("rot_vel_1_before")]).astype(float)
    rot_vel_0_after = np.array(cols[header.index("rot_vel_0_after")]).astype(float)
    rot_vel_1_after = np.array(cols[header.index("rot_vel_1_after")]).astype(float)
    mass_0 = np.array(cols[header.index("mass_0")]).astype(float)
    mass_1 = np.array(cols[header.index("mass_1")]).astype(float)
    radius_0 = np.array(cols[header.index("radius_0")]).astype(float)
    radius_1 = np.array(cols[header.index("radius_1")]).astype(float)
    height_0 = np.array(cols[header.index("height_0")]).astype(float)
    height_1 = np.array(cols[header.index("height_1")]).astype(float)
    rel_tangent_vel_before = np.array(cols[header.index("rel_tangent_vel_before")]).astype(float)
    rel_tangent_vel_after = np.array(cols[header.index("rel_tangent_vel_after")]).astype(float)
    rel_normal_vel_before = np.array(cols[header.index("rel_normal_vel_before")]).astype(float)
    rel_normal_vel_after = np.array(cols[header.index("rel_normal_vel_after")]).astype(float)
    normal_impulse = np.array(cols[header.index("normal_impulse")]).astype(float)
    tangent_impulse = np.array(cols[header.index("tangent_impulse")]).astype(float)
    fric_coeff = np.array(cols[header.index("fric_coeff")]).astype(float)
    e = np.array(cols[header.index("e")]).astype(float)
    moment_of_inertia_0 = np.array(cols[header.index("moment_of_inertia_0")]).astype(float)
    moment_of_inertia_1 = np.array(cols[header.index("moment_of_inertia_1")]).astype(float)

    total_momentum_before_x = mass_0 * vel_0_before_x + mass_1 * vel_1_before_x
    total_momentum_before_y = mass_0 * vel_0_before_y + mass_1 * vel_1_before_y
    total_momentum_after_x = mass_0 * vel_0_after_x + mass_1 * vel_1_after_x
    total_momentum_after_y = mass_0 * vel_0_after_y + mass_1 * vel_1_after_y
    momentum_diff_x = total_momentum_after_x - total_momentum_before_x
    momentum_diff_y = total_momentum_after_y - total_momentum_before_y
    momentum_diff = np.sqrt(momentum_diff_x**2 + momentum_diff_y**2)

    normal = np.array([pos_1_x - pos_0_x, pos_1_y - pos_0_y]).T
    normal /= np.linalg.norm(normal, axis=1).reshape((normal.shape[0], 1))
    tangent = np.array([-normal[1], normal[0]])

    momentum_before_0 = mass_0 * np.array([vel_0_before_x, vel_0_before_y])
    momentum_before_1 = mass_1 * np.array([vel_1_before_x, vel_1_before_y])
    momentum_after_0 = mass_0 * np.array([vel_0_after_x, vel_0_after_y])
    momentum_after_1 = mass_1 * np.array([vel_1_after_x, vel_1_after_y])
    rel_momentum_before = momentum_before_0 - momentum_before_1
    rel_momentum_after = momentum_after_0 - momentum_after_1
    rel_normal_momentum_before = np.dot(rel_momentum_before, normal)
    rel_tangent_momentum_before = np.dot(rel_momentum_before, tangent)
    rel_normal_momentum_after = np.dot(rel_momentum_after, normal)
    rel_tangent_momentum_after = np.dot(rel_momentum_after, tangent)

        
    group = [os.path.normpath(n).split(os.path.sep)[0] for n in name]
    groups = list(set(group))
    group = np.array(group)





    input("Press enter to continue...")

if __name__ == '__main__':
    main()