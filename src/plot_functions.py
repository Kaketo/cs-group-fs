from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt

def score(X_train, y_train, X_test, y_test, feature_order):
    total_scores = []
    model = LogisticRegression(max_iter=2000)
    for i, _ in enumerate(feature_order):
        cur_vars = feature_order[0:i+1]
        model.fit(X_train.iloc[:,cur_vars], y_train)
        y_pred = model.predict_proba(X_test.iloc[:,cur_vars])[:,1]
        total_scores.append(roc_auc_score(y_test, y_pred))

    return total_scores


def plot_scores(results_df, budget, total_cost, plot_settings):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axvline(x=budget, linewidth=3, label=f'budget={budget:.2f}')

    for alg in results_df.keys():
        idx = results_df[alg]['total_cost'] < budget

        total_costr_cut = results_df[alg]['total_cost'][idx]
        scores_cut = results_df[alg]['score'][idx]

        plt.plot(
            total_costr_cut,
            scores_cut,
            linestyle=plot_settings.get('linestyles').get(alg),
            marker='o',
            color=plot_settings.get('colors').get(alg),
            label=plot_settings.get('names').get(alg),
            markersize=9,
            linewidth=3
        )

    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title(f'Budget = {round(budget / total_cost * 100)}% of total cost', fontsize=26)
    ax.set_xlabel('Cost', fontsize=26)
    ax.set_ylabel('AUC', fontsize=26)
    plt.grid(color='gray', linestyle='dotted', linewidth=1, alpha=0.4)
    plt.xlim(left=0)
    ax.legend(prop={"size": 12}, loc='lower right')


    plt.tight_layout()
    plt.show()