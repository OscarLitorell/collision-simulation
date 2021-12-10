
import os
from types import TracebackType
import numpy as np



def convert_qualisys_2d(file_path):
    dirs = os.walk(file_path)

    for directory in dirs:
        files = directory[2]
        path = directory[0][len(file_path) + 1:]
        for file in files:
            if file.endswith('.tsv'):
                new_fp = os.path.join("results", path, file[:-4])
                if not os.path.exists(new_fp):
                    os.makedirs(new_fp)
                file_name = os.path.join(file_path, path, file)
                converted_files = convert_file(file_name)
                for i, converted_file in enumerate(converted_files):
                    np.savetxt(os.path.join(new_fp,f"marker{i}.tsv"), converted_file, delimiter="\t")


def convert_file(filepath):
    with open(filepath, "r") as text:
        lines = text.readlines()
        lines = [line[:-1].split("\t") for line in lines]

        metadata_list = lines[:10]
        metadata = {}
        for line in metadata_list:
            key = line[0].lower()
            value = line[1:]
            metadata[key] = value

        header = lines[10]
        data_text = lines[11:]
        data = np.array(data_text).astype(float)
        marker_count = (len(header) - 3) / 3
        marker_names = metadata["marker_names"]
        timestamps = data[:,1]
        marker_trajectories = []
        marker_ranges = []
        for i, marker in enumerate(marker_names):
            marker_trajectory = data[:,2+3*i : 2+3*i+3]
            marker_trajectories.append(marker_trajectory)
            marker_range = np.any(marker_trajectory != 0, axis=1)
            marker_ranges.append(marker_range)

        marker_ranges = np.all(np.array(marker_ranges), axis=0)
        marker_trajectories = [mt[marker_ranges] for mt in marker_trajectories]
        timestamps = timestamps[marker_ranges]

        trajectories_with_time = []

        marker_trajectories = xy_coords(marker_trajectories)
        if len(marker_trajectories) not in [3, 6]:
            print(len(marker_trajectories), filepath)
            return []

        for mt in marker_trajectories:
            trajectories_with_time.append(np.hstack((timestamps.reshape(len(timestamps), 1), mt)))
        return trajectories_with_time

def xy_coords(trajectories):
    # points.shape = (point_count, 3)
    trajectory_centers = [t.mean(axis=0) for t in trajectories]
    centered_trajectories = [t - trajectory_center for t, trajectory_center in zip(trajectories, trajectory_centers)]
    all_points = np.vstack(centered_trajectories)
    normal = find_normal(all_points)
    rot_mat = rotation_matrix(normal, np.array([0, 0, 1]))

    return [(rot_mat @ trajectory.T).T for trajectory in trajectories]


def rotation_matrix(vector_from, vector_to):
    v = (vector_from - vector_to)
    v_h = v / np.linalg.norm(v)
    vp = v_h - np.dot(v_h, vector_to) * vector_to
    vp_h = vp / np.linalg.norm(vp)
    A1 = np.eye(3) - 2 * np.outer(v_h, v_h)
    A2 = np.eye(3) - 2 * np.outer(vp_h, vp_h)
    return A2 @ A1


def find_normal(points):
    b =  points.T @ points
    eig_vals, eig_vecs = np.linalg.eig(b)
    eig_vals = np.real(eig_vals)
    eig_vecs = np.real(eig_vecs)
    idx = np.argsort(eig_vals)[0]
    return eig_vecs[:, idx]



def main():
    file_path = "qualisys_2d"
    convert_qualisys_2d(file_path)

if __name__ == '__main__':
    main()
