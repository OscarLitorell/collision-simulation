

import os
from sys import argv
import numpy as np
from numpy.lib.function_base import average
from scipy import signal
import matplotlib.pyplot as plt 


filter_range = 51

# Detta program innehåller funktioner för att analysera
# resultatdata från experimentet.

# Format på matriser och vektorer:
# matris.shape = (antal tidssteg, rader, kolumner).
# vektor.shape = (antal tidssteg, rader, 1).
# Vektorsamling.shape = (antal tidssteg, rader, kolumner=antal vektorer).
# Denna konvention bör alltid användas.

# Obs: Numpys transponat .T bör därför ej användas, använd 
# istället .swapaxes(1,2).

def separate_connected(markers, threshold = 0.01):
    """
    Grupperar alla punkter vars anstånd till varandra varierar mindre än threshold.
    """
    steps = markers.shape[0]
    marker_count = markers.shape[2]

    distances = get_distances(markers)
    avg = np.mean(distances, axis=0).reshape(1, marker_count, marker_count)
    
    max_deviations = np.abs(distances - avg).max(axis=0)
    connected = max_deviations < threshold

    object_count = np.linalg.matrix_rank(connected)

    object_markers = []
    for i in range(marker_count):
        row = connected[i].tolist()
        if row not in object_markers:
            object_markers.append(row)
        if len(object_markers) == object_count:
            break

    objects = []
    for object_marker in object_markers:
        objects.append(markers[:,:,np.array(object_marker)])

    
    return objects


def match_marker_constellation(markers, constellations):
    """
    Obs: bättre att använda get_center_and_rotation direkt.
    Hittar konstellationen som bäst matchar den markörerna.
    Obs: Kan inte skilja bra mellan spegelvända konstellationer.
    """

    distances = get_ordered_distances(markers)
    constellations = [c for c in constellations if c.shape[0] == markers.shape[0]]

    diffs = []
    for constellation in constellations:
        constellation_distances = get_ordered_distances(constellation.reshape(constellation.shape[0], 2, 1))
        diff_sq = (distances - constellation_distances)**2
        diffs.append(np.mean(diff_sq))
    index = np.argmin(diffs)
    return constellations[index]

def get_ordered_distances(markers):
    marker_count = markers.shape[2]
    distances = get_distances(markers)
    distances = np.sort(np.mean(distances, axis=0).flatten())[marker_count:]
    return distances

def get_distances(markers):
    steps = markers.shape[0]
    marker_count = markers.shape[2]
    distances = np.zeros((steps, marker_count, marker_count))
    for i in range(marker_count):
        for j in range(i):
            distance = np.sqrt(np.sum((markers[:,:,i] - markers[:,:,j])**2, axis=1))
            distances[:,i,j] = distance
    return distances + distances.swapaxes(1,2)


def get_marker_constellation(obj, center=None):
    steps = obj.shape[0]
    if center is None:
        center = np.mean(obj, axis=2).reshape(steps, 2, 1)
    rel_positions = obj - center
    distances = np.sqrt(np.sum(rel_positions**2, axis=1))
    avg_distances = np.mean(distances, axis=0)
    largest_distance_index = np.argmax(avg_distances)
    angles = np.arctan2(
        rel_positions[:,1,largest_distance_index],
        rel_positions[:,0,largest_distance_index]
    )
    rotation_matrices = np.array([
        [np.cos(angles), np.sin(angles)],
        [-np.sin(angles), np.cos(angles)]
    ]).swapaxes(1,2).swapaxes(0,1)
    
    rotated_positions = rotation_matrices @ rel_positions
    average_rotated_positions = np.mean(rotated_positions, axis=0).T

    return average_rotated_positions


def load_constellation(path):
    c = np.loadtxt(path, delimiter="\t")
    return c.reshape(c.shape[0], 2, 1).T


def save_constellation(markers, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    np.savetxt(path, markers, delimiter="\t")


def locate_center(markers):
    """
    Hittar masscentrum för ett objekt.
    Obs: Kräver att objektet roterar för bästa resultat.
    """ 
    
    # Se beräkning (masscentrum) för variabelnamn.
    steps = markers.shape[0]

    cr = np.array([[0,-1],[1,0]])

    # Filtrera positioner
    markers = signal.savgol_filter(markers, filter_range, 3, axis=0)

    p1, p2 = farthest_points(markers)
    
    v1 = np.gradient(p1, axis=0)
    v2 = np.gradient(p2, axis=0)
    v1 = signal.savgol_filter(v1, filter_range, 3, axis=0)
    v2 = signal.savgol_filter(v2, filter_range, 3, axis=0)    
    a1 = np.gradient(v1, axis=0)
    a2 = np.gradient(v2, axis=0)

    delta_p = p2 - p1

    delta_a = a2 - a1
    delta_a_sq = np.sum(delta_a**2, axis=1).reshape((steps, 1, 1))

    delta_ax = delta_a[:,0]
    delta_ay = delta_a[:,1]

    M = np.array([
        [delta_ax, delta_ay],
        [-delta_ay, delta_ax]
    ]).reshape((2, 2, steps)).swapaxes(1,2).swapaxes(0,1)

    ab = - M @ a1 / delta_a_sq
    
    ab_mean = np.mean(ab[filter_range:-filter_range], axis=0)

    a = ab_mean[0]
    b = ab_mean[1]

    center = p1 + a * delta_p + b * cr @ delta_p

    return center



def get_center_and_rotation(obj, constellations):
    marker_count = obj.shape[2]

    obj_normalized = normalized_rotation_and_position(obj)

    deviations = []
    for constellation in constellations:
        for i in range(2):
            c_normalized = normalized_rotation_and_position(constellation, i==1)
            all_points = np.concatenate((obj_normalized, c_normalized), axis=2)
            max_deviation = get_ordered_distances(all_points)[2*marker_count-1]
            reverse_direction = i == 1
            deviations.append((constellation, max_deviation, reverse_direction))
    
    constellation, max_deviation, reverse_direction = min(deviations, key=lambda x: x[1])

    center, rotation = match_transform(obj, constellation, reverse_direction)
    return center, rotation


def farthest_points(obj, reverse_direction=False):
    """
    Punkterna längst ifrån varandra
    obj.shape = (steps, 2, marker_count)
    """
    steps = obj.shape[0]
    marker_count = obj.shape[2]
    distances = np.mean(get_distances(obj), axis=0)

    argmax = np.argmax(distances)
    i = argmax % marker_count
    j = argmax // marker_count

    if reverse_direction:
        i, j = j, i 

    p1 = obj[:,:,i].reshape(steps, 2, 1)
    p2 = obj[:,:,j].reshape(steps, 2, 1)

    return p1, p2

def match_transform(obj, constellation, reverse_direction=False):
    p1, p2 = farthest_points(obj)
    c_p1, c_p2 = farthest_points(constellation, reverse_direction)
    delta_p = p2 - p1
    delta_c_p = c_p2 - c_p1
    obj_rotation = np.arctan2(delta_p[:,1], delta_p[:,0])
    c_rotation = np.arctan2(delta_c_p[:,1], delta_c_p[:,0])
    rotation = obj_rotation - c_rotation

    rot_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ]).swapaxes(1,2).swapaxes(0,1)[:,:,:,0]
    c_p1_rotated = rot_matrix @ c_p1
    center = p1 - c_p1_rotated
    return center, rotation


def normalized_rotation_and_position(obj, reverse_direction=False):
    """
    obj.shape = (steps, 2, marker_count)
    """
    steps = obj.shape[0]
    marker_count = obj.shape[2]
    p1, p2 = farthest_points(obj, reverse_direction)

    delta_p = p2 - p1
    rotation = np.arctan2(delta_p[:,1], delta_p[:,0])

    rot_matrix = np.array([
        [ np.cos(rotation), np.sin(rotation)],
        [-np.sin(rotation), np.cos(rotation)]
    ]).swapaxes(1,2).swapaxes(0,1)[:,:,:,0]

    shifted = obj - p1
    rotated = rot_matrix @ shifted

    rotated_mean = np.mean(rotated, axis=0).reshape(1, 2, marker_count)
    return rotated_mean





def load_results(results_path):
    """
    Ladda resultaten från en .tsv-fil.
    """
    
    files = [file for file in os.listdir(results_path) if file.endswith(".tsv")]
    first_file = True
    markers = []

    for file in files:
        data = np.genfromtxt(os.path.join(results_path, file), delimiter="\t")
        x = data[:,1] * 0.001
        y = data[:,2] * 0.001

        pos = np.array([[x], [y]]).swapaxes(1,2).swapaxes(0,1)

        markers.append(pos)

        if first_file:
            t = data[:,0]
            first_file = False

    tspan = t[-1] - t[0]
    dt = t[1] - t[0]
        
    markers = np.array(markers)[:,:,:,0].swapaxes(0,1).swapaxes(1,2)
    

    return markers, tspan, dt



