from scipy.stats import t
from sklearn.linear_model import LogisticRegression

from bounds import *
import numpy as np

from objectives import *
import pickle
import matplotlib.pyplot as plt


def week2_demo():
    # Basic example showing bound calculation for 2 variables with defined upper and lwoer bounds
    a = RandomVariable(4, lower=-1, upper=6)
    b = RandomVariable(5, lower=-3, upper=7)

    # Calculate bounds after doing a * b
    print(f"a={a}")
    print(f"b={b}")
    print(f"a+b = {a + b}\n")

    # The bounds default to the value passed if not given
    c = RandomVariable(4)
    d = RandomVariable(5)

    print(f"c = {c} \t d = {d}\n")
    # print bounds of c * d
    print(f"c*d = {c * d}\n")

    # Bounds for constants are again just defaulted to the input value
    e = RandomVariable(40, lower=30, upper=50)
    print(f"e={e}")
    print(f"e*3={e * 3}")

    # Bounds for division
    f = RandomVariable(40, lower=30, upper=50)
    g = RandomVariable(2, lower=0.5, upper=4)
    print(f"f={f}\tg={g}")
    print(f"f/g={f / g}")

    # Bounds for division with \infinity in the resultant bound
    h = RandomVariable(40, lower=30, upper=50)
    i = RandomVariable(2, lower=-2, upper=4)
    print(f"h={h}\ti={i}")
    print(f"h/i={h / i}")

    # Bounds for division with \infinity in the resultant bound
    j = RandomVariable(40, lower=30, upper=50)
    k = RandomVariable(2, lower=-2, upper=4)
    print(f"j={j}\tk={k}")
    print(f"|j*k|={abs(j * k)}")

    # Bounds for division with \infinity in the resultant bound
    j = RandomVariable(-1.5, lower=-2, upper=-1)
    k = RandomVariable(3, lower=0, upper=5)
    print(f"j={j}\tk={k}")
    print(f"j/k={j / k}")


# example with a gHat


def ttest_bounds(samples, delta):
    if not (isinstance(samples, numbers.Number) or isinstance(samples, np.ndarray)):
        raise ValueError(f"`samples` argument should be a numpy array")
    samples = np.array(samples)
    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array)")
    n = samples.size
    # print(f"n={n}")
    dev = (samples.std() / np.sqrt(n)) * t.ppf(1 - delta, n - 1)
    sample_mean = samples.mean()
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)


def hoeffdings_bounds(samples, delta):
    if not (isinstance(samples, numbers.Number) or isinstance(samples, np.ndarray)):
        raise ValueError(f"`samples` argument should be a numpy array")
    samples = np.array(samples)
    if samples.ndim > 1:
        raise ValueError(f"`samples` should be a vector (1-D array)")
    n = samples.size
    # print(f"n={n}")
    dev = 2 * np.sqrt(np.log(1 / delta) / (2 * n))
    sample_mean = samples.mean()
    return RandomVariable(sample_mean, lower=sample_mean - dev, upper=sample_mean + dev)


# Calculate TP rate for X[a_idx] = a_val and also return overall tp
def tp_rate_diff(X, y_pred, y_true, a_val, a_idx):
    tp_a = np.sum((X[:, a_idx] == a_val) & ((y_true == 1) & (y_pred == 1)))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    return tp_a, tp


# g_hat = (TP(X[a_idx] = a_val) / TP()) - threshold
def gHat_tprate_diff(X, y_pred, y_true, a_val, a_idx, threshold, delta=0.05, method='ttest'):
    tp_a, tp = tp_rate_diff(X, y_pred, y_true, a_val, a_idx)
    # print(f"tp_a={tp_a}\ttp={tp}")
    if method == 'ttest':
        TP_A, TP = map(lambda x: ttest_bounds(x, delta, X.shape[0]), [tp_a, tp])
    else:
        TP_A, TP = map(lambda x: hoeffdings_bounds(x, delta, X.shape[0]), [tp_a, tp])
    # print(f"TP_A={TP_A}\tTP={TP}")
    return (TP_A / TP) - threshold
    pass


# g_hat = (TP(X[a_idx] = a_val) / TP(X[b_idx] = b_val)) - threshold
def gHat_tp_rate_ab(X, y_pred, y_true, a_val, a_idx, b_val, b_idx, threshold):
    tp_a = np.sum((X[:, a_idx] == a_val) & ((y_true == 1) & (y_pred == 1)))
    tp_b = np.sum((X[:, b_idx] == b_val) & ((y_true == 1) & (y_pred == 1)))
    return (tp_a / tp_b) - threshold


# g_hat = (FP(X[a_idx] = a_val) / FP()) - threshold
def gHat_fp_rate_diff(X, y_pred, y_true, a_val, a_idx, threshold):
    fp_a = np.sum((X[:, a_idx] == a_val) & ((y_true == 0) & (y_pred == 1)))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return (fp_a / fp) - threshold


# g_hat = (FP(X[a_idx] = a_val) / FP(X[b_idx] = b_val)) - threshold
def gHat_fp_rate_ab(X, y_pred, y_true, a_val, a_idx, b_val, b_idx, threshold):
    fp_a = np.sum((X[:, a_idx] == a_val) & ((y_true == 0) & (y_pred == 1)))
    fp_b = np.sum((X[:, b_idx] == b_val) & ((y_true == 0) & (y_pred == 1)))
    return (fp_a / fp_b) - threshold


X_t = np.array([[0.5, 0, 1, 0],
                [0.5, 0, 1, 1],
                [0, 1, 1, 1],
                [0.5, 0, 0, 0],
                [0.5, 1, 1, 1]
                ])
y_t = np.array([1, 1, 1, 0, 1])
y_pred = np.array([1, 1, 1, 1, 1])


def week3_demo(n=1000, d=100):
    print(f"\nExperiment for {n} samples of dimensions {d}")
    X = np.random.randn(n, d)
    A_idx = np.random.randint(0, d)
    X[:, A_idx] = np.random.binomial(2, 0.2, n)
    y = np.random.binomial(1, 0.7, n)
    # print(np.unique(X[:, A_idx], return_counts=True))
    estimator = LogisticRegression().fit(X, y)
    y_pred = estimator.predict(X)
    tpr_a = tpr_rate(A_idx, 2)(X, y, y_pred)
    tpr = tpr_rate()(X, y, y_pred)
    t_bound = ttest_bounds(tpr_a, 0.05) / ttest_bounds(tpr, 0.05)
    print(f"Bounds using t-test: {t_bound}\n")
    h_bound = hoeffdings_bounds(tpr_a, 0.05) / hoeffdings_bounds(tpr, 0.05)
    print(f"Bounds using Hoeffdings: {h_bound}")
    return t_bound, h_bound


d = 500
ns = np.linspace(10, 100000, 40).astype(int)
t_bounds = []
h_bounds = []

# Run experiment for various number of samples
# for n in ns:
#     t_b, h = week3_demo(n, d)
#     t_bounds.append(t_b)
#     h_bounds.append(h)

# pickle.dump(h_bounds, open('h_bounds.p', 'wb'))
# pickle.dump(t_bounds, open('t_bounds.p', 'wb'))
h_bounds = pickle.load(open('h_bounds.p', 'rb'))
t_bounds = pickle.load(open('t_bounds.p', 'rb'))
pickle.dump(ns, open('ns.p', 'wb'))
s = 5
plt.plot(ns[s:], list(map(lambda x: x.upper, t_bounds))[s:], label='Upper Bound')
plt.plot(ns[s:], list(map(lambda x: x.lower, t_bounds))[s:], label='Lower Bound')
plt.plot(ns[s:], list(map(lambda x: x.value, t_bounds))[s:], label='Value')
plt.title('T-test bound for TPR Rate diff as gHat')
plt.xlabel('Number of data points')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.plot(ns[s:], list(map(lambda x: x.upper, h_bounds))[s:], label='Upper Bound')
plt.plot(ns[s:], list(map(lambda x: x.lower, h_bounds))[s:], label='Lower Bound')
plt.plot(ns[s:], list(map(lambda x: x.value, h_bounds))[s:], label='Value')
plt.title('Hoeffdings bound for TPR Rate diff as gHat')
plt.xlabel('Number of data points')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.plot(ns[s:], list(map(lambda x: x.upper - x.lower, h_bounds))[s:], label='delta - Hoeffdings')
plt.plot(ns[s:], list(map(lambda x: x.upper - x.lower, t_bounds))[s:], label='delta - t-test')
# plt.plot(ns[s:], list(map(lambda x: x.value, h_bounds))[s:], label='Value')
plt.title('upper_bound - lower_bound(delta) for TPR Rate diff as gHat')
plt.xlabel('Number of data points')
plt.ylabel('Value')
plt.legend()
plt.show()


# week3_demo(n, d)

# week3_demo(n * 100, d * 10)
#
# week3_demo(n * 1000, d * 20)
