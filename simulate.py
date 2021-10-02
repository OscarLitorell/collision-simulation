import os
from sys import argv
import numpy as np
import json

from numpy import linalg
from numpy.linalg.linalg import norm

gravity = 9.82

class Component:

    def __init__(self, radius, markers, mass, friction_coeff = 0.1):
        self.radius = radius
        self.markers = markers
        self.mass = mass
        self.friction_coeff = friction_coeff
    
    def moment_of_inertia(self):
        return self.mass * self.radius**2 / 2
    
    @staticmethod
    def import_from_file(filename, unit_multiplier = 0.001):
        with open(filename, 'r') as f:
            data = json.load(f)
        radius = data['radius'] * unit_multiplier
        markers = np.array(data['markers']) * unit_multiplier
        mass = data['mass']
        friction_coeff = data['friction_coeff']
        return Component(radius, markers, mass, friction_coeff)


class SimObject:

    def __init__(self, component, p0, v0, theta0, omega0, t0=0):
        self.component = component
        self.pos = p0
        self.v = v0
        self.theta = theta0
        self.omega = omega0
        self.t = t0
        self.history = []
    
    def update(self, dt, noise_level=0):
        self.pos = self.pos + self.v * dt
        self.theta = self.theta + self.omega * dt

        if self.v[0] != 0 or self.v[1] != 0:
            self.v = self.v * max(1 - dt * gravity * self.component.friction_coeff / linalg.norm(self.v), 0)

        self.omega = self.omega * (1 - self.component.friction_coeff * dt)
        self.t += dt
        self.history.append([self.t, *self.get_marker_positions(noise_level)])

    def get_marker_positions(self, noise_level=0):
        rotation_matrix = np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta),  np.cos(self.theta)]
        ])

        positions = np.array([self.pos + rotation_matrix @ self.component.markers[i] for i in range(len(self.component.markers))])
        noise = (np.random.random(size=positions.shape) * 2 - 1) * noise_level

        return positions + noise



class Sim:

    def __init__(
            self, 
            dt, 
            t_max, 
            objects,
            noise_level=0,
            elasticity_coeff=None,
            collision_friction_coeff=None):

        self.dt = dt
        self.t_max = t_max
        self.objects = objects
        self.noise_level = noise_level
        self.elasticity_coeff = elasticity_coeff
        self.collision_friction_coeff = collision_friction_coeff

    def simulate(self):
        for i in range(int(self.t_max / self.dt)):
            for obj in self.objects:
                obj.update(self.dt, self.noise_level)

            collisions = self.check_collissions()
            for collision in collisions:
                self.collide(self.objects[collision[0]], self.objects[collision[1]])
  
    def check_collissions(self):
        collissions = []
        for i in range(len(self.objects)):
            p1 = self.objects[i].pos
            r1 = self.objects[i].component.radius
            for j in range(i + 1, len(self.objects)):
                p2 = self.objects[j].pos
                r2 = self.objects[j].component.radius
                if np.linalg.norm(p1 - p2) < r1 + r2:
                    collissions.append((i, j))
        return collissions

    
    def collide(self, obj1, obj2):
        p1 = obj1.pos
        p2 = obj2.pos
        r1 = obj1.component.radius
        r2 = obj2.component.radius
        v1 = obj1.v
        v2 = obj2.v
        m1 = obj1.component.mass
        m2 = obj2.component.mass
        
        normal = p2 - p1
        normal /= np.linalg.norm(normal)
        tangent = np.array([-normal[1], normal[0]])

        v1n = np.dot(v1, normal)
        v2n = np.dot(v2, normal)

        e = self.elasticity_coeff # Change later

        m = np.array([
            [ m1-e*m2, (1+e)*m2],
            [(1+e)*m2,  m2-e*m1]
        ]) / (m1+m2)
        
        [v1np, v2np] = m @ np.array([v1n, v2n])
        
        J = m1 * (v1n - v1np) # Impulse on each object

        v1tp, v2tp, omega1p, omega2p = self.collision_tangential(obj1, obj2, J)

        v1p = v1np * normal + v1tp * tangent
        v2p = v2np * normal + v2tp * tangent
        
        obj1.v = v1p
        obj2.v = v2p
        obj1.omega = omega1p
        obj2.omega = omega2p


        # Correct for overlap:
        overlap = r1 + r2 - np.linalg.norm(p2 - p1)
        obj1.pos = p1 - overlap * normal * 0.5
        obj2.pos = p2 + overlap * normal * 0.5


    def collision_tangential(self, obj1, obj2, J):
        p1 = obj1.pos
        p2 = obj2.pos
        r1 = obj1.component.radius
        r2 = obj2.component.radius
        m1 = obj1.component.mass
        m2 = obj2.component.mass

        nh = (p2 - p1) / np.linalg.norm(p2 - p1) # N-hat (normal unit vector)
        th = np.array([-nh[1], nh[0]]) # T-hat (tangent unit vector)

        v1 = np.dot(obj1.v, th)
        v2 = np.dot(obj2.v, th)

        omega1 = obj1.omega
        omega2 = obj2.omega
        I1 = obj1.component.moment_of_inertia()
        I2 = obj2.component.moment_of_inertia()
        v1t = omega1 * r1 + v1
        v2t = -omega2 * r2 + v2

        u = self.collision_friction_coeff # Friction coeff

        # Konstant glid:
        Ju = J * u * sign(v2t - v1t)
        d_v1 = Ju / m1
        d_v2 = -Ju / m2
        d_omega1 = Ju * r1 / I1
        d_omega2 = Ju * r2 / I2

        v1p = v1 + d_v1
        v2p = v2 + d_v2
        omega1p = omega1 + d_omega1
        omega2p = omega2 + d_omega2

        v1tp = omega1p * r1 + v1p
        v2tp = -omega2p * r2 + v2p

        if (v2t - v1t) * (v2tp - v1tp) < 0:
            A = np.array([
                [m1, m2,      0,      0],
                [ 0,  0,  I1/r1, -I2/r2],
                [m1,  0, -I1/r1,      0],
                [ 1, -1,     r1,     r2]
            ])
            a = v2 - v1 - r1 * omega1 - r2 * omega2
            B = np.array([0, 0, 0, a])

            x = np.linalg.solve(A, B)
            d_v1 = x[0]
            d_v2 = x[1]
            d_omega1 = x[2]
            d_omega2 = x[3]

            v1p = v1 + d_v1
            v2p = v2 + d_v2
            omega1p = omega1 + d_omega1
            omega2p = omega2 + d_omega2


        return [v1p, v2p, omega1p, omega2p]

    def export_to_file(self, save_path, unit_multiplier=0.001):

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        i = 0
        for obj in self.objects:
            lines = obj.history
            for marker_index, _ in enumerate(obj.component.markers):

                with open(f"{save_path}/marker{i}.tsv", "w") as f:
                    for line in lines:
                        t = line[0]
                        pos = line[marker_index + 1] / unit_multiplier
                        x = pos[0]
                        y = pos[1]

                        f.write("\t".join(str(round(a, 3)) for a in [t, x, y]) + "\n")
                i += 1

    @staticmethod
    def import_from_file(filepath, unit_multiplier=0.001):

        with open(filepath, "r") as f:
            setup = json.load(f)
            timespan = setup["timespan"]
            dt = setup["dt"]
            noise_level = setup["noise_level"] * unit_multiplier
            rigidbodies = setup["rigidbodies"]
            elasticity_coeff = setup["elasticity_coeff"] if "elasticity_coeff" in setup else 1
            collision_friction_coeff = setup["collision_friction_coeff"] if "collision_friction_coeff" in setup else 0
        objects = []

        configs_dir = os.path.dirname(filepath)
        sim_dir = os.path.dirname(configs_dir)
        objects_dir = os.path.join(sim_dir, "objects")

        for rigidbody in rigidbodies:
            component = Component.import_from_file(os.path.join(objects_dir, rigidbody['object'] + ".json"), unit_multiplier)
            position = np.array(rigidbody["position"]) * unit_multiplier
            velocity = np.array(rigidbody["velocity"]) * unit_multiplier
            theta = rigidbody["theta"]
            omega = rigidbody["omega"]
            objects.append(SimObject(component, position, velocity, theta, omega))

        return Sim(dt, timespan, objects, noise_level, elasticity_coeff, collision_friction_coeff)


def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0

def main(sim_path, plot=False):

    configs = os.listdir(os.path.join(sim_path, "configs"))

    for config in configs:
        if not config.endswith(".json"):
            continue
        path = os.path.join(sim_path, "configs", config)
        name = os.path.splitext(config)[0]

        print(f"Simulating {name}")
        sim = Sim.import_from_file(path)
        sim.simulate()
        print(f"Simulation {name} done")
        sim.export_to_file(f"results/{name}")
        print(f"Simulation {name} exported\n")




if __name__ == "__main__":
    try:
        path = argv[1]
        main(path)
    except IndexError:
        print("Please specify path to simulation setup")
        exit(1)
        

