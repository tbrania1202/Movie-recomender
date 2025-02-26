from typing import List, Tuple
from copy import deepcopy
import numpy as np


def owd1(X):
    def is_dominated(p1, p2):
        return all(a <= b for a, b in zip(p1, p2))

    x = deepcopy(X)
    P = []
    dominance_count = 0
    empty_row = tuple([float('-inf')] * len(x[0]))

    for i in range(len(x)):
        if x[i] == empty_row:
            continue
        else:
            current_point = x[i]
            current_index = x.index(current_point)
            found_dominance = False

            for j in range(i, len(x)):
                if x[j] == empty_row:
                    continue
                elif is_dominated(current_point, x[j]):
                    x[j] = empty_row
                    dominance_count += 1
                elif is_dominated(x[j], current_point):
                    x[current_index] = empty_row
                    dominance_count += 1
                    current_point = x[j]
                    current_index = x.index(current_point)
                    found_dominance = True

            if current_point == empty_row:
                continue
            if current_point in P:
                continue
            else:
                P.append(current_point)
            if not found_dominance:
                x[current_index] = empty_row

    return P, dominance_count


def owd2(X):
    def is_dominated(p1, p2):
        return all(a <= b for a, b in zip(p1, p2))

    x = deepcopy(X)
    P = []
    dominance_count = 0
    empty_row = tuple([float('-inf')] * len(x[0]))

    for i in range(len(x)):
        if x[i] == empty_row:
            continue

        current_point = x[i]
        current_index = x.index(current_point)

        for j in range(i, len(x)):
            if x[j] == empty_row:
                continue
            elif is_dominated(current_point, x[j]):
                x[j] = empty_row
                dominance_count += 1
            elif is_dominated(x[j], current_point):
                x[current_index] = empty_row
                dominance_count += 1
                current_point = x[j]
                current_index = x.index(current_point)

        if current_point == empty_row:
            continue
        if current_point in P:
            continue
        else:
            P.append(current_point)

        for m in range(len(x)):
            if x[m] == empty_row:
                continue
            elif is_dominated(current_point, x[m]):
                x[m] = empty_row
                dominance_count += 1

        x[current_index] = empty_row

        non_empty_indices = [x.index(i) for i in x if i[0] > float('-inf')]
        if len(non_empty_indices) == 1:
            P.append(x[non_empty_indices[0]])
            break

    return P, dominance_count


def owd3(X):
    x = deepcopy(X)
    P = []
    minimum = []
    empty_row = tuple([float('-inf')] * len(x[0]))
    sup = []
    D = []
    J = []
    dominance_count = 0

    for i in range(len(empty_row)):
        min_point = min(x, key=lambda k: k[i])
        minimum.append(min_point[i])

    for j in range(len(x)):
        d = sum((a - b) ** 2 for a, b in zip(minimum, x[j]))
        sup.append((d, j))

    sup = sorted(sup, key=lambda item: item[0])

    for k in range(len(sup)):
        D.append(sup[k][0])
        J.append(sup[k][1])

    M = len(x)
    m = 0

    while m < M:
        current_point = x[J[m]]

        if current_point == empty_row:
            m = m + 1
            continue

        for i in range(M):
            if all(a <= b for a, b in zip(current_point, x[i])):
                x[i] = empty_row
                dominance_count += 1

        P.append(current_point)
        x[J[m]] = empty_row
        m = m + 1

    return P, dominance_count


def czebyszew(matrix):
    m, n = matrix.shape
    v = np.zeros((m, n))
    v_max = np.zeros(n)
    for i in range(n):
        v_max[i] = max(matrix[:, i])
    for i in range(m):
        for j in range(n):
            v[i][j] = matrix[i][j] / v_max[j]
    return v

def euclides(matrix):
    m, n = matrix.shape
    v = np.zeros((m, n))
    v_euc = np.zeros(n)
    for i in range(n):
        v_euc[i] = np.sqrt(np.sum(matrix[:, i]) ** 2)
    for i in range(m):
        for j in range(n):
            v[i][j] = matrix[i][j] / v_euc[j]
    return v


def topsis(matrix, weights, criteria, norm):
    matrix_r = deepcopy(matrix)
    if norm == 'czebyszew':
        matrix_norm = czebyszew(matrix_r)
    elif norm == 'euclides':
        matrix_norm = euclides(matrix_r)

    m, n = matrix.shape

    w = deepcopy(weights)
    w = w / np.sum(weights)
    for i in range(m):
        for j in range(n):
            matrix_norm[i][j] *= w[j]

    worst_alt = []
    best_alt = []
    for i in range(n):
        if criteria[i]:
            worst_el = min(matrix_norm[:, i])
            worst_alt.append(worst_el)
            best_el = max(matrix_norm[:, i])
            best_alt.append(best_el)
        else:
            worst_el = max(matrix_norm[:, i])
            worst_alt.append(worst_el)
            best_el = min(matrix_norm[:, i])
            best_alt.append(best_el)

    worst_distance = []
    best_distance = []
    if norm == 'czebyszew':
        for i in range(m):
            best_distance.append(max([(abs(norm - best)) for norm, best in zip(matrix_norm[i], best_alt)]))
            worst_distance.append(max([(abs(norm - worst)) for norm, worst in zip(matrix_norm[i], worst_alt)]))
    elif norm == 'euclides':
        for i in range(m):
            best_distance.append(np.sqrt(sum([(_v - _u) ** 2 for _v, _u in zip(matrix_norm[i], best_alt)])))
            worst_distance.append(np.sqrt(sum([(_v - _l) ** 2 for _v, _l in zip(matrix_norm[i], worst_alt)])))

    best_distance = np.array(best_distance).astype(float)
    worst_distance = np.array(worst_distance).astype(float)

    si = worst_distance / (worst_distance + best_distance)

    el = sorted(list(enumerate(si)), key=lambda r: r[1])
    el = el[::-1]
    rank = [[i[0], el.index(i)] for i in el]
    best_rank = [i[1] + 1 for i in sorted(rank)]
    return si, best_rank


def normalization(matrix):
    m, n = matrix.shape
    new = np.zeros(n)
    for i in range(m):
        for j in range(n):
            new[j] += (matrix[i, j])**2
    for k in range(len(new)):
        new[k] = np.sqrt(new[k])
    return new


def weight_normalization(matrix, weights):
    m = matrix.shape[0]
    v = 0
    for i in range(m):
        p = (matrix[i] - weights[i])**2
        v += p
    v = np.sqrt(v)
    return v


def rsm(matrix, weights, points, status_quo):

    m, n = matrix.shape
    m_copy = deepcopy(matrix)
    #normalizacja macierzy
    m_norm = normalization(m_copy)
    for i in range(m):
        for j in range(n):
            m_copy[i, j] = m_copy[i, j] / m_norm[j]

    #wyznaczenie ważonej znormalizowanej macierzy
    w_copy = deepcopy(weights)
    w_copy = w_copy / np.sum(weights)
    for i in range(m):
        for j in range(n):
            m_copy[i, j] *= w_copy[j]

    #wyznaczenie punktów docelowych oraz punktów status quo
    target = np.zeros(n)
    new_status_quo = np.zeros(n)
    for i in range(n):
        sum_points = np.sum(points[i]) #sum_points = np.sum(points[:, i]) 
        target[i] = sum_points / m
        sum_status_quo = np.sum(status_quo[i]) #sum_status_quo = np.sum(status_quo[:, i])
        new_status_quo[i] = sum_status_quo / m

    #wyznaczenia najlepszych i najgorszych punktów
    best = np.zeros(m)
    worst = np.zeros(m)
    for i in range(m):
        best[i] = weight_normalization(m_copy[i], target)
        worst[i] = weight_normalization(m_copy[i], new_status_quo)

    #obliczanie podobieństwa do najgorszego punktu
    si = np.zeros(m)
    for i in range(m):
        si[i] = worst[i] / (worst[i] + best[i])

    #rankingowanie
    el = sorted(list(enumerate(si)), key=lambda r: r[1])
    el = el[::-1]
    rank = [[i[0], el.index(i)] for i in el]
    best_rank = [i[1] + 1 for i in sorted(rank)]
    return si, best_rank


def uta_star(data_matrix, weights, criteria, lambda_value=0.5):
    num_alternatives, num_criteria = data_matrix.shape
    num_experts = len(weights)

    # Uwzględnienie współczynników wiarygodności ekspertów
    expert_weights = np.ones(num_experts) / num_experts  # Załóżmy, że są równe
    normalized_data = np.zeros_like(data_matrix, dtype=float)

    for j in range(num_criteria):
        normalized_data[:, j] = data_matrix[:, j] * weights[j] * criteria[j]

    # Obliczenia dla każdej alternatywy
    best_rank = np.zeros(num_alternatives)

    for i in range(num_alternatives):
        for j in range(num_criteria):
            criterion_values = normalized_data[:, j]
            sorted_indices = np.argsort(criterion_values)

            for k, alternative_index in enumerate(sorted_indices):
                best_rank[alternative_index] += expert_weights[j] * (k + 1)

    # Współczynnik lambda
    best_rank = lambda_value * best_rank

    # Sortowanie wyników
    sorted_indices = np.argsort(best_rank)
    final_rank = [i + 1 for i in sorted_indices]

    # Obliczenie podobieństwa
    max_rank = max(final_rank)
    si = [1 - (rank / (max_rank + 1)) for rank in final_rank]

    return si, final_rank




if __name__ == "__main__":
    X1 = [(5,5), (3,6), (4,4), (5,3), (3,3), (1,8), (3,4), (4,5), (3,10), (6,6), (4,1), (3,5)]
    print(owd1(X1))
    print(owd2(X1))
    print(owd3(X1))

    matrix = np.array([[13, 8299], [17, 7399], [13, 3200], [4, 4549], [17, 4049], [12, 8199], [9.5, 2699], [11, 5399],
                       [6, 7299], [14, 5099], [10, 7699], [10, 5099], [14, 5099], [15, 5699], [20, 10499], [4, 4399]])
    weights = np.array([4, 3])
    points = np.array([[16, 3000], [18, 5500], [19, 4000], [20, 4500]])
    status_quo = np.array([[10, 5500], [11, 7500], [12, 6500], [13, 5000]])


    print("\nRSM")
    si, best_rank = rsm(matrix, weights, points, status_quo)
    for i, (similarity, rank) in enumerate(zip(si, best_rank)):
        print(f"Alternatywa {i+1}: Podobieństwo = {similarity}, Pozycja = {rank}")

    matrix = np.array([
        [8, 15, 15, 5],
        [2, 1, 3, 3],
        [7, 4, 9, 13],
        [7, 8, 7, 3],
        [16, 6, 7, 1]
    ])
    weights = np.array([2, 3, 1, 4])
    criteria = np.array([5, 3, 1, 2])

    print("\nTopsis")
    si, best_rank = topsis(matrix, weights, criteria, 'euclides')
    for i, (similarity, rank) in enumerate(zip(si, best_rank)):
        print(f"Alternatywa {i+1}: Podobieństwo = {similarity}, Pozycja = {rank}")

    print("\nUTA star")
    si, best_rank = uta_star(matrix, weights, criteria)
    for i, (similarity, rank) in enumerate(zip(si, best_rank)):
        print(f"Alternatywa {i+1}: Podobieństwo = {similarity}, Pozycja = {rank}")