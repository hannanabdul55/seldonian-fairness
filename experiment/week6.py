from seldonian.objectives import *
from sklearn.metrics import *

import matplotlib.pyplot as plt
from scipy.stats import cumfreq

import pandas as pd
from tempeh.configurations import datasets

from sklearn import preprocessing

dataset = datasets['lawschool_passbar']()

# Load data
X_train, X_test = dataset.get_X(format=pd.DataFrame)
y_train, y_test = dataset.get_y(format=pd.Series)
A_train, A_test = dataset.get_sensitive_features(name='race', format=pd.Series)

# Combine all training data into a single data frame and glance at a few rows
all_train_raw = pd.concat([X_train, A_train, y_train], axis=1)
all_test_raw = pd.concat([X_test, A_test, y_test], axis=1)

all_data = pd.concat([all_train_raw, all_test_raw], axis=0)

X = all_data[['lsat', 'ugpa', 'race']]
y = all_data[['pass_bar']]
le = preprocessing.LabelEncoder()
X['race'] = le.fit_transform(X['race'])
scaler = preprocessing.StandardScaler()
X[['lsat', 'ugpa']] = scaler.fit_transform(X[['lsat', 'ugpa']])
A = X['race']
A_idx = 'race'


def compare_cdfs(data, A, num_bins=100):
    cdfs = {}
    assert len(np.unique(A)) == 2

    limits = (min(data), max(data))
    s = 0.5 * (limits[1] - limits[0]) / (num_bins - 1)
    limits = (limits[0] - s, limits[1] + s)

    for a in np.unique(A):
        subset = data[A == a]

        cdfs[a] = cumfreq(subset, numbins=num_bins, defaultreallimits=limits)

    lower_limits = [v.lowerlimit for _, v in cdfs.items()]
    bin_sizes = [v.binsize for _, v in cdfs.items()]
    actual_num_bins = [v.cumcount.size for _, v in cdfs.items()]

    assert len(np.unique(lower_limits)) == 1
    assert len(np.unique(bin_sizes)) == 1
    assert np.all([num_bins == v.cumcount.size for _, v in cdfs.items()])

    xs = lower_limits[0] + np.linspace(0, bin_sizes[0] * num_bins, num_bins)

    disparities = np.zeros(num_bins)
    for i in range(num_bins):
        cdf_values = np.clip([v.cumcount[i] / len(data[A == k]) for k, v in cdfs.items()], 0, 1)
        disparities[i] = max(cdf_values) - min(cdf_values)

    return xs, cdfs, disparities


def plot_and_compare_cdfs(data, A, num_bins=100, loc='best'):
    xs, cdfs, disparities = compare_cdfs(data, A, num_bins)

    for k, v in cdfs.items():
        plt.plot(xs, v.cumcount / len(data[A == k]), label=k)

    assert disparities.argmax().size == 1
    d_idx = disparities.argmax()

    xs_line = [xs[d_idx], xs[d_idx]]
    counts = [v.cumcount[d_idx] / len(data[A == k]) for k, v in cdfs.items()]
    ys_line = [min(counts), max(counts)]

    plt.plot(xs_line, ys_line, 'o--')
    disparity_label = "max disparity = {0:.3f}\nat {1:0.3f}".format(disparities[d_idx], xs[d_idx])
    plt.text(xs[d_idx], 1, disparity_label, ha="right", va="top")

    plt.xlabel(data.name)
    plt.ylabel("cumulative frequency")
    plt.legend(loc=loc)
    plt.show()


all_train_grouped = all_data.groupby('race')

counts_by_race = all_train_grouped[['lsat']].count().rename(
    columns={'lsat': 'count'})

quartiles_by_race = all_train_grouped[['lsat', 'ugpa']].quantile([.25, .50, .75]).rename(
    index={0.25: "25%", 0.5: "50%", 0.75: "75%"}, level=1).unstack()

rates_by_race = all_train_grouped[['pass_bar']].mean().rename(
    columns={'pass_bar': 'pass_bar_rate'})

summary_by_race = pd.concat([counts_by_race, quartiles_by_race, rates_by_race], axis=1)

A_p = np.sum(A) / A.shape[0]
print(f"{A_p * 100}% of the labels for 'race' is 'white'")


# pass bar rate should be equal for race='white' or race='balck'
def create_tpr_ghat(A_idx, A_val):
    def tpr_ab(X, y_true, y_pred, delta, n=None):
        tp_a = tpr_rate(A_idx, 1)(X, y_true, y_pred)
        tp = tpr_rate(A_idx, 0)(X, y_true, y_pred)
        if method == 'ttest':
            bound = abs(ttest_bounds(tp, delta, n) - ttest_bounds(tp_a, delta, n))
        else:
            bound = abs(hoeffdings_bounds(tp, delta, n) - hoeffdings_bounds(tp_a, delta, n))
        return bound.upper - 0.2

    return tpr_ab


A_idx = 2
tpr_ab = create_tpr_ghat(A_idx, 1)

# Construct the ghat
ghats = []
ghats.append({
    'fn': tpr_ab,
    'delta': 0.05
})

method = 'ttest'
op_method = 'CG'

estimator = LogisticRegressionSeldonianModel(X.to_numpy(), y.to_numpy().flatten(), g_hats=ghats).fit(
    opt=op_method)
print(f"Accuracy: {balanced_accuracy_score(y, estimator.predict(X))}\n")

print("Unconstrainedoptimization")
