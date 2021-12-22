

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

    common_vel_x = (mass_0 * vel_0_before_x + mass_1 * vel_1_before_x) / (mass_0 + mass_1)
    common_vel_y = (mass_0 * vel_0_before_y + mass_1 * vel_1_before_y) / (mass_0 + mass_1)
    common_vel_normal = normal[:, 0] * common_vel_x + normal[:, 1] * common_vel_y
    vel_0_before_normal = normal[:, 0] * vel_0_before_x + normal[:, 1] * vel_0_before_y
    vel_1_before_normal = normal[:, 0] * vel_1_before_x + normal[:, 1] * vel_1_before_y
    vel_0_after_normal = normal[:, 0] * vel_0_after_x + normal[:, 1] * vel_0_after_y
    vel_1_after_normal = normal[:, 0] * vel_1_after_x + normal[:, 1] * vel_1_after_y

    energy_before_0_normal = 0.5 * mass_0 * (vel_0_before_normal - common_vel_normal)**2
    energy_before_1_normal = 0.5 * mass_1 * (vel_1_before_normal - common_vel_normal)**2
    energy_after_0_normal = 0.5 * mass_0 * (vel_0_after_normal - common_vel_normal)**2
    energy_after_1_normal = 0.5 * mass_1 * (vel_1_after_normal - common_vel_normal)**2
    energy_before_normal = energy_before_0_normal + energy_before_1_normal
    energy_after_normal = energy_after_0_normal + energy_after_1_normal


    momentum_before_0 = mass_0.reshape((len(mass_0), 1)) * np.array([vel_0_before_x - common_vel_x, vel_0_before_y - common_vel_y]).T
    momentum_before_1 = mass_1.reshape((len(mass_1), 1)) * np.array([vel_1_before_x - common_vel_x, vel_1_before_y - common_vel_y]).T
    momentum_after_0 = mass_0.reshape((len(mass_0), 1)) * np.array([vel_0_after_x - common_vel_x, vel_0_after_y - common_vel_y]).T
    momentum_after_1 = mass_1.reshape((len(mass_1), 1)) * np.array([vel_1_after_x - common_vel_x, vel_1_after_y - common_vel_y]).T

    rel_momentum_before = momentum_before_0 - momentum_before_1
    rel_momentum_after = momentum_after_0 - momentum_after_1
    rel_normal_momentum_before = rel_momentum_before[:, 0] * normal[:, 0] + rel_momentum_before[:, 1] * normal[:, 1]
    rel_tangent_momentum_before = rel_momentum_before[:, 0] * tangent[:, 0] + rel_momentum_before[:, 1] * tangent[:, 1]
    rel_normal_momentum_after = rel_momentum_after[:, 0] * normal[:, 0] + rel_momentum_after[:, 1] * normal[:, 1]
    rel_tangent_momentum_after = rel_momentum_after[:, 0] * tangent[:, 0] + rel_momentum_after[:, 1] * tangent[:, 1]

    momentum_scale = np.linalg.norm(rel_momentum_before, axis=1)



    
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
        plt.scatter(x=group[sel], y=momentum_diff[sel] / momentum_scale[sel], c="#00f3")
        plt.title("Normerad skillnad i rörelsemängd mellan före och efter kollision")
        plt.xticks(rotation=45)
        plt.xlabel("Kollisionsgrupp")
        plt.ylabel("Normerad förändring av total rörelsemängd [1]")
        plt.show()


    def x_group_y_e():
        plt.scatter(x=group[sel], y=e[sel], c="#00f3")
        plt.title("Elasticitetskoefficient")
        plt.xticks(rotation=45)
        plt.show()


    def x_group_y_fric_coeff():
        plt.scatter(x=group[sel], y=fric_coeff[sel], c="#00f3")
        plt.title("Impulskvot")
        plt.xticks(rotation=45)
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
                plt.arrow(x[i], y[i], dx[i], dy, head_width=0, head_length=0, fc='#0004', ec='#0004')

            static = x + dx > 0.1
            plt.plot((x+dx)[static], y[static], 'r.')
            plt.plot((x+dx)[~static], y[~static], 'bx')


            plt.grid()
            plt.title(f"Impulskvot mot relativ tangenthastighet, {name}")
            plt.xlabel("Relativ tangenthastighet i kontaktpunkt ($\Delta u_{\\parallel}$), [m/s]")
            plt.ylabel("Impulskvot ($\mu_i$), [1]")
            plt.show()


    def x_normal_energy_y_fric_coeff():
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

            sign_t = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            static = rel_tangent_vel_after[s] * sign_t > 0.1
            plt.plot(x[static], y[static], 'r.')
            plt.plot(x[~static], y[~static], 'bx')
            plt.grid()
            plt.title(f"Impulskvot mot normalenergi, {name}")
            plt.xlabel("Rörelseenergi från hastighet i normalled ($E_{\\bot}$), [J]")
            plt.ylabel("Impulskvot ($\mu_i$), [1]")
            plt.show()
    

    def x_rel_tangent_vel_y_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        rub_sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
        ]
        alu_sels = [
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for sels in [rub_sels, alu_sels]:
            for name, s in sels:
                sign = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
                x = rel_tangent_vel_before[s] * sign
                y = e[s]

                if sels == alu_sels:
                    plt.plot(x, y, '.g')
                else:
                    plt.plot(x, y, '.')

            names = [name for name, _ in sels]
                    
            if len(names) > 1:
                plt.legend(names)

            plt.grid()
            plt.title(f"Elasticitetskoefficient mot tangenthastighet, {' och '.join(names)}")
            plt.xlabel("Relativ tangentiell hastighet i kontaktpunkt före stöten ($\Delta u_{\\parallel}$), [m/s]")
            plt.ylabel("Elasticitetskoefficient ($e$), [1]")
            plt.ylim(0, 1.1)
            plt.show()

    
    def x_normal_energy_y_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        rub_sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0))
        ]

        alu_sels = [
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for sels in [rub_sels, alu_sels]:
            for name, s in sels:
                sign = (rel_normal_vel_before[s] > 0).astype(int) * 2 - 1
                x = rel_normal_vel_before[s] * sign
                y = e[s]

                if sels == alu_sels:
                    plt.plot(x, y, '.g')
                else:
                    plt.plot(x, y, '.')

            names = [name for name, _ in sels]

            if len(names) > 1:
                plt.legend(names)

            plt.grid()
            plt.title(f"Elasticitetskoefficient mot normalenergi, {' och '.join(names)}")
            plt.xlabel("Rörelseenergi från hastighet i normalled ($E_{\\bot}$), [J]")
            plt.ylabel("Elasticitetskoefficient ($e$), [1]")
            plt.ylim(0, 1.1)

            plt.show()

    def x_normal_energy_y_tangent_vel_z_fric_coeff():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign_t = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            x = energy_before_normal[s]
            y = rel_tangent_vel_after[s] * sign_t
            z = fric_coeff[s]

            mask = y > 0.1
            x1 = x[mask]
            y1 = y[mask]
            z1 = z[mask]
            x2 = x[~mask]
            y2 = y[~mask]
            z2 = z[~mask]


            fig = plt.figure()
            ax = plt.axes(projection='3d')
            if len(x1) > 0:
                ax.scatter3D(x1, y1, z1, c='r', marker='o')
            ax.scatter3D(x2, y2, z2, c='b', marker='o')
            plt.title(f"Impulskvot mot normalenergi och tangenthastighet, {name}")
            ax.set_xlabel("Rörelseenergi från hastighet i normalled ($E_{\\bot}$), [J]")
            ax.set_ylabel("Relativ tangentiell hastighet i kontaktpunkt ($\Delta u_{\\parallel}$), [m/s]")
            ax.set_zlabel("Impulskvot ($\mu_i$), [1]")
            plt.show()

    def x_normal_energy_y_tangent_vel_z_e():
        rub_rub = np.array(["aluRub-aluRub" in g for g in group])
        rub = np.all([np.array(["Rub" in g for g in group]), ~rub_rub], axis=0)
        sels = [
            ("båda gummi", np.all([rub_rub, sel], axis=0)),
            ("ena gummi", np.all([rub, sel], axis=0)),
            ("aluminium", np.all([~rub, ~rub_rub, sel], axis=0))
        ]

        for name, s in sels:
            sign_t = (rel_tangent_vel_before[s] > 0).astype(int) * 2 - 1
            x = energy_before_normal[s]
            y = rel_tangent_vel_before[s] * sign_t
            z = e[s]

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.scatter3D(x, y, z)
            plt.title(f"Elasticitetskoefficient mot normalenergi och tangenthastighet, {name}")
            ax.set_xlabel("Rörelseenergi från hastighet i normalled ($E_{\\bot}$), [J]")
            ax.set_ylabel("Relativ tangentiell hastighet i kontaktpunkt ($\Delta u_{\\parallel}$), [m/s]")
            ax.set_zlabel("Elasticitetskoefficient ($e$), [1]")
            plt.show()




    
    # x_group_y_momentum_diff()
    # x_group_y_e()
    # x_group_y_fric_coeff()
    # x_rel_tangent_vel_y_fric_coeff()
    # x_normal_energy_y_fric_coeff()
    # x_rel_tangent_vel_y_e()
    # x_normal_energy_y_e()
    # x_normal_energy_y_tangent_vel_z_fric_coeff()
    # x_normal_energy_y_tangent_vel_z_e()

    input("Press enter to continue...")

if __name__ == '__main__':
    main()