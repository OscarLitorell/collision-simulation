
import convert_qualisys_2d
import detect_marker_positions
import calculate_movement
import analyze_collision
import compile_analyzed_collisions_2d
from sys import argv

import os

def wait(yes):
    if yes:
        input("Press enter to continue...")

def main(step_through):
    print("Converting qualisys data...")
    convert_qualisys_2d.main()
    print("Detecting marker positions...")
    wait(step_through)
    detect_marker_positions.main()
    collision_names = [p[0] for p in os.walk("results") if "collisions" in p[0].split(os.path.sep) and len(p[2]) > 0]
    collision_names = [name[len("results/"):] for name in collision_names]
    print("Calculating movement and analyzing collisions...")
    wait(step_through)
    for c in collision_names:
        calculate_movement.main(c, False)
        analyze_collision.main(c)
    print("Compiling analyzed collisions...")
    wait(step_through)
    compile_analyzed_collisions_2d.main()
    print("Done")

if __name__ == '__main__':
    main("-s" in argv)