import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from pyitlib import discrete_random_variable as drv


def return_group(groups, k):
    for n, j in groups.items():
        for f in j:
            if f == k:
                return n
    raise Exception("Group not found.")


def return_cost(groups, selected_groups, group_costs, k):
    group_id = return_group(groups, k)
    if group_id in selected_groups:
        return 0
    else:
        return group_costs[group_id]


def return_normalized_cost(groups, selected_groups, group_costs, k):
    min_cost = min(group_costs.values())
    max_cost = max(group_costs.values())

    normalized_costs = {}
    for i, v in group_costs.items():
        normalized_costs[i] = (1 - 0.1) * (v - min_cost) / (max_cost - min_cost) + 0.1

    group_id = return_group(groups, k)
    if group_id in selected_groups:
        return 0
    else:
        return normalized_costs[group_id]


def information_mutual_combined(X1, X2, Y, base):
    # I(Y, (X1, X2)) =
    # I(Y, X1) + I(Y, X1) + II(Y,X1,X2) =
    # I(Y, X1) + I(Y, X1) + I(Y,X1|X2) - I(Y,X1) =
    # I(Y, X1) + I(Y,X1|X2)
    a = drv.information_mutual(Y, X2, base=base)
    b = drv.information_mutual_conditional(X=Y, Y=X1, Z=X2, base=base)

    return a + b


def information_mutual_conditional_fixed_arguments_order(Z, X, Y, base):
    return drv.information_mutual_conditional(X, Y, Z, base=base)


def relevance_group_criterion(X_candidate: np.ndarray, y: np.ndarray, X_prev: np.ndarray = None):
    assert X_candidate.shape[0] == y.shape[0]

    a = 0
    b = 0

    if X_candidate.shape[1] == 1:
        a += relevance_criterion(X_candidate.reshape(len(y)), y, X_prev)
    else:
        for i in range(X_candidate.shape[1]):
            a += np.apply_along_axis(
                information_mutual_combined,
                axis=0,
                arr=np.delete(X_candidate, i, axis=1),
                X2=X_candidate[:, i],
                Y=y,
                base=np.exp(1)
            ).sum()

    if X_prev is not None:
        assert X_prev.shape[0] == X_candidate.shape[0]
        for i in range(X_candidate.shape[1]):
            b += np.apply_along_axis(
                information_mutual_combined,
                axis=0,
                arr=X_prev,
                X2=X_candidate[:, i],
                Y=y,
                base=np.exp(1)
            ).sum()

    return (a + b)


def relevance_criterion(x_candidate: np.ndarray, y: np.ndarray, X_prev: np.ndarray = None):
    assert x_candidate.shape[0] == y.shape[0]

    if X_prev is None:
        a = drv.information_mutual(
            X=x_candidate,
            Y=y,
            base=np.exp(1)
        )
    else:
        a = np.apply_along_axis(
            information_mutual_conditional_fixed_arguments_order,
            axis=0,
            arr=X_prev,
            X=x_candidate,
            Y=y,
            base=np.exp(1)
        ).sum()

    return (a)


def find_best_feature(X: np.ndarray, y: np.ndarray, groups: dict, selected_groups: set, group_costs: dict,
                      lmb: float = None, prev_var_idx: list = None):
    if prev_var_idx is not None:
        cand_var_idx = list(set(range(X.shape[1])).difference(set(prev_var_idx)))
    else:
        cand_var_idx = list(range(X.shape[1]))
    candidates_relevance_scores = []

    for k in cand_var_idx:
        relevance_criterion_value = relevance_criterion(
            x_candidate=X[:, k],
            y=y,
            X_prev=X[:, prev_var_idx] if (prev_var_idx is not None) else None
        )
        candidates_relevance_scores.append(relevance_criterion_value)

    if lmb is not None:
        candidates_relevance_cost_scores = [0] * len(cand_var_idx)
        for i, k in enumerate(cand_var_idx):
            candidates_relevance_cost_scores[i] = candidates_relevance_scores[i] - lmb * return_normalized_cost(groups,
                                                                                                                selected_groups,
                                                                                                                group_costs,
                                                                                                                k)
    else:
        candidates_relevance_cost_scores = candidates_relevance_scores

    k = np.argmax(candidates_relevance_cost_scores)

    selected_feature = cand_var_idx[k]
    selected_relevance_score = candidates_relevance_scores[k]
    selected_cost = return_cost(groups, selected_groups, group_costs, selected_feature)

    return selected_feature, selected_relevance_score, selected_cost


def find_best_group(X: np.ndarray, y: np.ndarray, groups: dict, selected_groups: set, group_costs: dict,
                    lmb: float = None, prev_var_idx: list = None):
    min_cost = min(group_costs.values())
    max_cost = max(group_costs.values())

    normalized_costs = {}
    for k, v in group_costs.items():
        normalized_costs[k] = (1 - 0.1) * (v - min_cost) / (max_cost - min_cost) + 0.1

    if prev_var_idx is not None:
        cand_group_idx = list(set(groups.keys()).difference(selected_groups))
    else:
        cand_group_idx = list(list(groups.keys()))
    cand_groups_relevance_scores = []

    for k in cand_group_idx:
        relevance_criterion_value = relevance_group_criterion(
            X_candidate=X[:, groups[k]],
            y=y,
            X_prev=X[:, prev_var_idx] if (prev_var_idx is not None) else None
        )
        cand_groups_relevance_scores.append(relevance_criterion_value)

    if lmb is not None:
        cand_groups_relevance_cost_scores = [0] * len(cand_group_idx)
        for i, k in enumerate(cand_group_idx):
            cand_groups_relevance_cost_scores[i] = cand_groups_relevance_scores[i] - lmb * normalized_costs.get(k)
    else:
        cand_groups_relevance_cost_scores = cand_groups_relevance_scores

    k = np.argmax(cand_groups_relevance_cost_scores)

    selected_group = cand_group_idx[k]
    selected_relevance_score = cand_groups_relevance_scores[k]
    selected_cost = group_costs.get(selected_group)

    return selected_group, selected_relevance_score, selected_cost


def find_best_features_subset(X, y, groups, group_costs, budget, lmb=None, verbose=True, cache_dict=None,
                              cache_dir=None):
    number_of_features = X.shape[1]

    if cache_dict is not None:
        S = cache_dict.get('feature_order')
        U = [i for i in range(number_of_features)]
        selected_groups = set()

        for i in S:
            U.remove(i)
            selected_groups.add(return_group(groups, i))
        number_of_features = number_of_features - len(S)

        total_costs = cache_dict.get('total_costs')
        total_cost = total_costs[-1]

        print(
            f'Read cached results with total cost of {total_cost} and selected {len(S)} features and {len(selected_groups)} groups with {number_of_features} features left.')
    else:
        S = []
        U = [i for i in range(number_of_features)]
        selected_groups = set()
        total_costs = []
        total_cost = 0

    for _ in range(number_of_features):
        k, _, cost = find_best_feature(
            X=X,
            y=y,
            groups=groups,
            selected_groups=selected_groups,
            group_costs=group_costs,
            prev_var_idx=S if len(S) > 0 else None,
            lmb=lmb
        )
        if total_cost + cost >= budget:
            break

        total_cost = total_cost + cost
        total_costs.append(total_cost)
        # relevance_scores.append(relevance_score)
        selected_groups.add(return_group(groups, k))
        U.remove(k)
        S.append(k)

        if (cost > 0) and (verbose):
            print('Num of features: ', len(S))
            print('Total cost     : ', total_cost)

        if cache_dir is not None:
            if lmb is None:
                filename = f'{cache_dir}traditional_a1_B_{math.ceil(total_cost)}.csv'
            else:
                filename = f'{cache_dir}cs_a1_B_{math.ceil(total_cost)}_lopt.csv'

            pd.DataFrame({
                'feature_order': S,
                'total_cost': total_costs
            }).to_csv(filename, index=False)

    return S, total_costs


def backward_step(X: np.ndarray, y: np.ndarray, tau: float, cur_var_idx: list, prev_var_idx: list = None):
    if prev_var_idx is None:
        r_var_idx = cur_var_idx
    else:
        r_var_idx = cur_var_idx + prev_var_idx

    var_to_delete_idx = []

    for r in r_var_idx:
        for j in r_var_idx:
            a = drv.information_mutual(X[:, r], X[:, j], base=np.exp(1)) / drv.entropy(X[:, r], base=np.exp(1))
            b = drv.information_mutual(y, X[:, r], base=np.exp(1))
            c = drv.information_mutual(y, X[:, j], base=np.exp(1))

            if (a > tau) and (b < c):
                if r not in var_to_delete_idx:
                    var_to_delete_idx.append(r)
    return var_to_delete_idx


def find_best_group_subset(X, y, groups, group_costs, budget, tau, lmb=None, verbose=True, cache_dict=None,
                           cache_dir=None):
    number_of_groups = len(groups)
    if cache_dict is not None:
        tau = cache_dict.get('tau')
        S = cache_dict.get('feature_order')
        selected_groups = set()

        for i in S:
            selected_groups.add(return_group(groups, i))
        number_of_groups = number_of_groups - len(selected_groups)

        total_costs = cache_dict.get('total_costs')
        total_cost = total_costs[-1]

        print(
            f'Read cached results with total cost of {total_cost} and selected {len(S)} features and {len(selected_groups)} groups with {number_of_groups} groups left.')
    else:

        S = []
        number_of_groups = len(groups)
        selected_groups = set()
        total_costs = []
        total_cost = 0

    for _ in range(number_of_groups):
        k, _, cost = find_best_group(
            X=X,
            y=y,
            groups=groups,
            selected_groups=selected_groups,
            group_costs=group_costs,
            prev_var_idx=S if len(S) > 0 else None,
            lmb=lmb
        )
        G = groups.get(k)
        D = backward_step(X, y, tau, cur_var_idx=G, prev_var_idx=S)

        if total_cost + cost >= budget:
            break

        if len(G) != len(D):
            total_cost = total_cost + cost
        selected_groups.add(k)

        for f in G:
            if f not in D:
                S.append(f)
                total_costs.append(total_cost)

        if (cost > 0) and (verbose):
            print('Num of f, c : ', len(S), ', ', len(total_costs))
            print('Total cost  : ', total_cost)

        if cache_dir is not None:
            if lmb is None:
                filename = f'{cache_dir}traditional_a2_B_{math.ceil(total_cost)}.csv'
            else:
                filename = f'{cache_dir}cs_a2_B_{math.ceil(total_cost)}_lopt_group_tau_{str(tau).replace(".", "")}.csv'

            pd.DataFrame({
                'feature_order': S,
                'total_cost': total_costs
            }).to_csv(filename, index=False)

    return S, total_costs


def score_cv(X, y, cur_vars):
    model = LogisticRegression(max_iter=2000)
    cv_score = cross_val_score(model, X[:, cur_vars], y, cv=3, scoring='roc_auc').mean()
    return cv_score


def lmb_max(X, y, group_costs):
    I_max = np.apply_along_axis(
        drv.information_mutual,
        axis=0,
        arr=X,
        Y=y,
        base=np.exp(1)
    ).max()

    I_min = np.apply_along_axis(
        drv.information_mutual,
        axis=0,
        arr=X,
        Y=y,
        base=np.exp(1)
    ).min()

    costs = list(set(group_costs.values()))
    costs.sort()

    c_1 = 0
    c_2 = costs[0]

    lmb_max = (I_max - I_min) / (c_2 - c_1)

    return lmb_max


def lmb_opt(X, y, groups, group_costs, normalized_costs, budget, n=10, m=5000):
    l_max = lmb_max(X, y, normalized_costs)

    subset_idx = np.random.choice(X.shape[0], m, replace=False)
    X = X[subset_idx, :]
    y = y[subset_idx]

    l_values = np.linspace(0 + l_max / n, l_max, num=n)
    cv_scores = []
    for idx, i in enumerate(l_values):
        print(f'L: {idx}/{n}')
        S, _, _ = find_best_features_subset(
            X,
            y,
            groups,
            group_costs,
            budget,
            i,
            False)

        cv_scores.append(score_cv(X, y, S))
    k = np.argmax(cv_scores)

    return l_values[k]


def lmb_max_group(X, y, groups, group_costs):
    group_scores = []
    for k in groups.keys():
        relevance_criterion_value = relevance_group_criterion(
            X_candidate=X[:, groups[k]],
            y=y,
            X_prev=None
        )
        group_scores.append(relevance_criterion_value)

    I_max = max(group_scores)
    I_min = min(group_scores)

    costs = list(set(group_costs.values()))
    costs.sort()

    c_1 = 0
    c_2 = costs[0]

    lmb_max = (I_max - I_min) / (c_2 - c_1)

    return lmb_max


def lmb_opt_group(X, y, groups, group_costs, normalized_costs, budget, n=10, m=5000):
    l_max = lmb_max_group(X, y, groups, normalized_costs)

    subset_idx = np.random.choice(X.shape[0], m, replace=False)
    X = X[subset_idx, :]
    y = y[subset_idx]

    l_values = np.linspace(0 + l_max / n, l_max, num=n)
    cv_scores = []
    for idx, i in enumerate(l_values):
        print(f'L: {idx}/{n}')
        S, _ = find_best_group_subset(
            X=X,
            y=y,
            groups=groups,
            group_costs=group_costs,
            budget=budget,
            tau=0.8,
            lmb=i,
            verbose=False
        )

        cv_scores.append(score_cv(X, y, S))
    k = np.argmax(cv_scores)

    return l_values[k]

