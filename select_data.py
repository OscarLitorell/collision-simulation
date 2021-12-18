

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
        
    group = [os.path.normpath(n).split(os.path.sep)[0] for n in name]
    groups = list(set(group))
    group = np.array(group)

        
    blacklist = [
        "aluBig-alu\\collisions\\T_aluBig_Y-alu_spin0031",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_5cm0013",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_5cm0016",
        "aluRub-aluBig\\collisions\\T_aluBig_Y-aluRub0020",
        "aluRub-aluThi\\collisions\\Y_aluThi_T-aluRub0011",
        "aluTunn-alu\\collisions\\Y_aluTunn_Y_alu0003",
        "alu-aluRub\\collisions\\T-alu_Y-aluRub_2cm0017",
        "alu-aluRub\\collisions\\T-alu_Y-aluRub_2cm0017",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_1cm0004",
        "aluBig-alu\\collisions\\T_aluBig_Y-alu_spin0031"
    ]

    sel = np.array([n not in blacklist for n in name])


    fig, ax = plt.subplots()
    ax.scatter(x=group[sel], y=momentum_diff[sel])
    plt.show()



    for g in groups:
        print(g)
        mask = np.all([group==g, sel], axis=0)
        x = abs(normal_impulse[mask])
        y = e[mask]
        plt.plot(x, y, "o")
        plt.title(g)
        plt.show()
        

    input("Press enter to continue...")

if __name__ == '__main__':
    main()