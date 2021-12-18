
import convert_qualisys_2d
import detect_marker_positions
import calculate_movement
import analyze_collision
import compile_analyzed_collisions_2d
from sys import argv

import os

def wait(step_through, message, func):
    if step_through:
        print(message)
        ans = input("Press enter to continue, or 's' to skip\n")
        if ans == "s":
            print("Skipping...")
        else:
            func()
    else:
        func()

def calculate_movement_and_analyze_collisions():
    collision_names = [p[0] for p in os.walk("results") if "collisions" in p[0].split(os.path.sep) and len(p[2]) > 0]
    collision_names = [name[len("results/"):] for name in collision_names]
    for c in collision_names:
        calculate_movement.main(c, False)
        analyze_collision.main(c)

    

def main(step_through):

    wait(step_through, "Converting Qualisys 2D data", convert_qualisys_2d.main)
    wait(step_through, "Detecting marker positions", detect_marker_positions.main)
    wait(step_through, "Calculating movement and collisions", calculate_movement_and_analyze_collisions)
    wait(step_through, "Compiling analyzed collisions", compile_analyzed_collisions_2d.main)

    print("Done")

if __name__ == '__main__':
    main("-s" in argv)