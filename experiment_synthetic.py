from synthetic import *
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from datetime import datetime
from time import time
exp_config = {}

exp_config['N'] = np.linspace(500, 100000, 10).astype(np.int)

exp_config['trials'] = 10
exp_config['methods'] = ['ttest', 'hoeffdings']
exp_config['D'] = 20
exp_config['tprs'] = [0.4, 0.6]
exp_config['test_size'] = 0.5
exp_config['opt'] = 'Powell'
result = []
uc_result = []

checkpoint = time()


def save_res(obj):
    pickle.dump(obj, open(f"./results_week6/results-{checkpoint}.p", 'wb'))


def run_experiment(exp):
    for n in exp['N']:
        failure_rate = []
        accuracy = []
        mean_ghat = []
        uc_failure_rate = []
        uc_accuracy = []
        uc_mean_ghat = []

        for t in np.arange(exp['trials']):
            X, y, A_idx = make_synthetic(n, exp['D'], *exp['tprs'])
            ghats = []
            ghats.append({
                'fn': ghat_tpr_diff(A_idx, threshold=abs(exp['tprs'][0] - exp['tprs'][1]) - 0.1),
                'delta': 0.05
            })
            # est = SeldonianAlgorithmLogRegCMAES(X, y, test_size=exp['test_size'], g_hats=ghats,
            #                                     verbose=True)
            est = LogisticRegressionSeldonianModel(X, y, test_size=exp['test_size'], g_hats=ghats,
                                                   verbose=True)
            est.fit(opt=exp['opt'])

            acc = accuracy_score(y, est.predict(X))
            safe = est.safetyTest()
            mean_ghat.append(safe)
            failure_rate.append(1 if safe > 0 else 0)
            accuracy.append(acc)

            uc_est = LogisticRegression(penalty='none').fit(X, y)
            uc_acc = accuracy_score(y, uc_est.predict(X))
            y_preds = uc_est.predict(X)
            ghat_val = ghats[0]['fn'](X, y, y_preds, ghats[0]['delta'])
            uc_failure_rate.append(1 if ghat_val > 0 else 0)
            uc_accuracy.append(uc_acc)
            uc_mean_ghat.append(ghat_val)
        result.append({
            'failure_rate': np.mean(failure_rate),
            'failure_rate_std': np.std(failure_rate),
            'accuracy': np.mean(accuracy),
            'ghat': np.mean(mean_ghat),
            'uc_failure_rate': np.mean(uc_failure_rate),
            'uc_failure_rate_std': np.std(uc_failure_rate),
            'uc_accuracy': np.mean(uc_accuracy),
            'uc_ghat':np.mean(uc_mean_ghat)
        })
        save_res(result)
        print(f"Result for config: N = {n}")
        print(f"{result[-1]!r}")


run_experiment(exp_config)
