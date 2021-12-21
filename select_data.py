

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
    tangent = np.array([-normal[:, 1], normal[:, 0]]).T

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

        
    blacklist = [
        "aluBig-alu\\collisions\\T_aluBig_Y-alu_spin0031",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_5cm0013",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_5cm0016",
        "aluRub-aluBig\\collisions\\T_aluBig_Y-aluRub0020",
        "aluRub-aluThi\\collisions\\Y_aluThi_T-aluRub0011",
        "aluTunn-alu\\collisions\\Y_aluTunn_Y_alu0003",
        "alu-aluRub\\collisions\\T-alu_Y-aluRub_2cm0017",
        "alu-aluRub\\collisions\\T-alu_Y-aluRub_2cm0017",
        "aluBig-aluThi\\collisions\\Y_aluThi_T-aluBig_1cm0004"
    ]

    sel = np.array([n not in blacklist for n in name])

    def x_group_y_momentum_diff():
        plt.scatter(x=group[sel], y=momentum_diff[sel])
        plt.title("Skillnad i rörelsemängd mellan före och efter kollision")
        plt.show()


    def x_group_y_e():
        plt.scatter(x=group[sel], y=e[sel])
        plt.title("Elasticitetskoefficient")
        plt.show()


    def x_group_y_fric_coeff():
        plt.scatter(x=group[sel], y=fric_coeff[sel])
        plt.title("Friktionskoefficient")
        plt.show()


    def x_rel_tangent_vel_y_fric_coeff():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_tangent_vel_before[s] * sign
            y = fric_coeff[s]
            dx = (rel_tangent_vel_after[s] - rel_tangent_vel_before[s]) * sign
            dy = 0

            for i in range(len(x)):
                plt.arrow(x[i], y[i], dx[i], dy, head_width=0, head_length=0, fc='#888888', ec='#888888')
            plt.plot(x, y, '.')
            plt.plot(x+dx, y, '.')
            plt.grid()
            plt.title(f"Friktionskoefficient mot tangenthastighet, {name}")
            plt.xlabel("rel. tangenthastighet")
            plt.ylabel("Friktionskoefficient")
            plt.show()


    def x_rel_normal_vel_y_fric_coeff():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign = (rel_normal_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_normal_vel_before[s] * sign
            y = fric_coeff[s]

            plt.plot(x, y, '.')
            plt.grid()
            plt.title(f"Friktionskoefficient mot normalhastighet, {name}")
            plt.xlabel("rel. normalhastighet")
            plt.ylabel("Friktionskoefficient")
            plt.show()
    

    def x_rel_tangent_vel_y_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_tangent_vel_before[s] * sign
            y = e[s]

            plt.plot(x, y, '.')
            plt.grid()
            plt.title(f"Elasticitetskoefficient mot tangenthastighet, {name}")
            plt.xlabel("rel. tangenthastighet")
            plt.ylabel("Elasticitetskoefficient")
            plt.show()

    
    def x_rel_normal_vel_y_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign = (rel_normal_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_normal_vel_before[s] * sign
            y = e[s]

            plt.plot(x, y, '.')
            plt.grid()
            plt.title(f"Elasticitetskoefficient mot normalhastighet, {name}")
            plt.xlabel("rel. normalhastighet")
            plt.ylabel("Elasticitetskoefficient")
            plt.show()

    def x_rel_normal_vel_y_tangent_vel_z_fric_coeff():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign_n = (rel_normal_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_normal_vel_before[s] * sign_n
            sign_t = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            y = rel_tangent_vel_after[s] * sign_t
            z = fric_coeff[s]           

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z)
            plt.title(f"Friktionskoefficient mot normalhastighet och tangenthastighet, {name}")
            ax.set_xlabel("rel. normalhastighet")
            ax.set_ylabel("rel. tangenthastighet")
            ax.set_zlabel("Friktionskoefficient")
            plt.show()

    def x_rel_normal_vel_y_tangent_vel_z_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign_n = (rel_normal_vel_before[s] > 0).astype(int) * 2 - 1
            sign_t = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            x = rel_normal_vel_before[s] * sign_n
            y = rel_tangent_vel_before[s] * sign_t
            z = e[s]

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z)
            plt.title(f"Elasticitetskoefficient mot normalhastighet och tangenthastighet, {name}")
            ax.set_xlabel("rel. normalhastighet")
            ax.set_ylabel("rel. tangenthastighet")
            ax.set_zlabel("Elasticitetskoefficient")
            plt.show()




        
    # x_group_y_momentum_diff()
    # x_group_y_e()
    # x_group_y_fric_coeff()
    # x_rel_tangent_vel_y_fric_coeff()
    # x_rel_normal_vel_y_fric_coeff()
    # x_rel_tangent_vel_y_e()
    # x_rel_normal_vel_y_e()
    # x_rel_normal_vel_y_tangent_vel_z_fric_coeff()
    # x_rel_normal_vel_y_tangent_vel_z_e()

    input("Press enter to continue...")

if __name__ == '__main__':
    main()