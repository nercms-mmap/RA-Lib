import csv

from scipy.io import loadmat


def get_first_key(file_path):
    mat_data = loadmat(file_path)
    keys = list(mat_data.keys())
    filtered_keys = [key for key in keys if not key.startswith('__')]
    return filtered_keys[0]


if __name__ == '__main__':
    file_d = "..\\full_lists\\ans\\rank-result-simulation-dataset-iRANK.mat"
    file_rel = "..\\full_lists\\data\\simulation_test_rel.mat"

    key_d = get_first_key(file_d)
    key_rel = get_first_key(file_rel)

    d = loadmat(file_d)[key_d]
    rel = loadmat(file_rel)[key_rel]

    results = []
    # d是二维数组(query， item) = rank, 转为csv文件，每行[query, item, rank]
    with open("data\\trans_mat_results_rankbased.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for query in range(d.shape[0]):
            for item in range(d.shape[1]):
                row = [query, item, d[query, item]]
                writer.writerow(row)

    with open("data\\trans_mat_rel.csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        for query in range(rel.shape[0]):
            for item in range(rel.shape[1]):
                row = [query, 0, item, rel[query, item]]
                writer.writerow(row)
