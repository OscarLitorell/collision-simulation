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

        if self.v[0] != 0 and self.v[1] != 0:
            self.v = self.v * max(1 - dt * gravity * self.component.friction_coeff / linalg.norm(self.v), 0)

        self.v = self.v * (1 - self.component.friction_coeff * dt)
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

    def __init__(self, dt, t_max, objects, noise_level=0):
        self.dt = dt
        self.t_max = t_max
        self.objects = objects
        self.noise_level = noise_level
    

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
            for j in range(i + 1, len(self.objects)):
                p1 = self.objects[i].pos
                p2 = self.objects[j].pos
                r1 = self.objects[i].component.radius
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
        
        normal = p1 - p2
        normal /= np.linalg.norm(normal)
        tangent = np.array([-normal[1], normal[0]])

        v1n = np.dot(v1, normal)
        v2n = np.dot(v2, normal)
        v1t = v1 - v1n * normal
        v2t = v2 - v2n * normal

        e = 1 # Change later

        m = np.array([
            [e*m1-m2, (1+e)*m2],
            [(1+e)*m2, e*m2-m1]
        ]) / (e*(m1+m2))
        
        [v1np, v2np] = m @ np.array([v1n, v2n])

        [v1tp, v2tp] = [v1t, v2t] # No friction

        v1p = v1np * normal + v1tp
        v2p = v2np * normal + v2tp
        
        obj1.v = v1p
        obj2.v = v2p
        
        overlap = r1 + r2 - np.linalg.norm(p1 - p2)
        obj1.pos = p1 + overlap * normal * 0.5
        obj2.pos = p2 - overlap * normal * 0.5


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

        return Sim(dt, timespan, objects, noise_level)


def main(sim_path):

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
        

