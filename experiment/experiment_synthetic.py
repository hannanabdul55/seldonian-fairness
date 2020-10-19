from seldonian.synthetic import *
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from time import time

import argparse
import json

parser = argparse.ArgumentParser(description='Process some config values')
parser.add_argument("--config")
parser.add_argument("checkpoint", nargs='?', default='results-'+str(time()))

args = parser.parse_args()
if args.config:
    exp_config = json.load(open(args.config, "r"))
else:
    exp_config = {'N': np.geomspace(10e2, 10e6, 20).astype(np.int), 'trials': 4,
                  'methods': ['ttest', 'hoeffdings'], 'D': 20, 'tprs': [0.3, 0.6],
                  'test_size': 0.4, 'opt': ['Powell']}

uc_result = []
results = {}

if args.checkpoint:
    checkpoint = f"./results_2/{args.checkpoint}.p"
else:
    checkpoint = './results_2/results-' + str(time()) + '.p'


def save_res(obj):
    pickle.dump(obj, open(checkpoint, 'wb'))


def run_experiment(exp):
    for n in exp['N']:
        X, y, A_idx = make_synthetic(n, exp['D'], *exp['tprs'])
        for opt in exp['opt']:
            if opt not in results:
                results[opt] = []
            failure_rate = []
            accuracy = []
            mean_ghat = []
            uc_failure_rate = []
            uc_accuracy = []
            uc_mean_ghat = []
            for t in np.arange(exp['trials']):
                ghats = []
                ghats.append({
                    'fn': ghat_tpr_diff(A_idx,
                                        threshold=abs(exp['tprs'][0] - exp['tprs'][1]) / 4),
                    'delta': 0.05
                })
                if opt == 'CMAES':
                    est = SeldonianAlgorithmLogRegCMAES(X, y, test_size=exp['test_size'],
                                                        g_hats=ghats,
                                                        verbose=True)
                else:
                    est = LogisticRegressionSeldonianModel(X, y, test_size=exp['test_size'],
                                                           g_hats=ghats,
                                                           verbose=True)
                est.fit()

                acc = accuracy_score(y, est.predict(X))
                safe = est.safetyTest(predict=False)
                mean_ghat.append(safe)
                failure_rate.append(1 if safe > 0.0 else 0)
                accuracy.append(acc)

                uc_est = LogisticRegression(penalty='none').fit(X, y)
                uc_acc = accuracy_score(y, uc_est.predict(X))
                y_preds = uc_est.predict(X)
                ghat_val = ghats[0]['fn'](X, y, y_preds, ghats[0]['delta'],predict=False)
                uc_failure_rate.append(1 if ghat_val > 0.0 else 0)
                uc_accuracy.append(uc_acc)
                uc_mean_ghat.append(ghat_val)
            results[opt].append({
                'failure_rate': np.mean(failure_rate),
                'failure_rate_std': np.std(failure_rate),
                'accuracy': np.mean(accuracy),
                'ghat': np.mean(mean_ghat),
                'uc_failure_rate': np.mean(uc_failure_rate),
                'uc_failure_rate_std': np.std(uc_failure_rate),
                'uc_accuracy': np.mean(uc_accuracy),
                'uc_ghat': np.mean(uc_mean_ghat)
            })
            save_res(results)
            print(f"Result for config: N = {n}, opt= {opt}")
            print(f"{results[opt][-1]!r}")


if __name__ == '__main__':
    print(f"Running experiment with checkpoint: {checkpoint}")
    run_experiment(exp_config)
