import numpy as np
import csv
import glob
import re

bin_path = 'experiment_results/bin/'
csv_path = 'experiment_results/csv/'

def get_existing_experiment_numbers():
	result_files       = glob.glob(bin_path + 'results*.npz')
	experiment_numbers = [re.search('.*results([0-9]*).*', fn, re.IGNORECASE) for fn in result_files]
	experiment_numbers = [int(i.group(1)) for i in experiment_numbers]
	experiment_numbers.sort()
	return experiment_numbers

def genFilename(n):	
	return bin_path + 'results%d.npz' % n

def addMoreResults(newFileId, ms, seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_failures_g2,
	LS_solutions_found, LS_fs, LS_failures_g1, LS_failures_g2):

	newFile = np.load(genFilename(newFileId))
	new_ms                        = newFile['ms']
	new_seldonian_solutions_found = newFile['seldonian_solutions_found']
	new_seldonian_fs              = newFile['seldonian_fs']
	new_seldonian_failures_g1     = newFile['seldonian_failures_g1']
	new_seldonian_failures_g2     = newFile['seldonian_failures_g2']
	new_LS_solutions_found        = newFile['LS_solutions_found']
	new_LS_fs                     = newFile['LS_fs']
	new_LS_failures_g1            = newFile['LS_failures_g1']
	new_LS_failures_g2            = newFile['LS_failures_g2']

	if type(ms)==type(None):
		return [new_ms, new_seldonian_solutions_found, new_seldonian_fs, new_seldonian_failures_g1, new_seldonian_failures_g2,
				new_LS_solutions_found, new_LS_fs, new_LS_failures_g1, new_LS_failures_g2]
	else:
		#ms                         = np.vstack([ms,                        new_ms])
		seldonian_solutions_found  = np.vstack([seldonian_solutions_found, new_seldonian_solutions_found])
		seldonian_fs               = np.vstack([seldonian_fs,              new_seldonian_fs])
		seldonian_failures_g1      = np.vstack([seldonian_failures_g1,     new_seldonian_failures_g1])
		seldonian_failures_g2      = np.vstack([seldonian_failures_g2,     new_seldonian_failures_g2])
		LS_solutions_found         = np.vstack([LS_solutions_found,        new_LS_solutions_found])
		LS_fs                      = np.vstack([LS_fs,                     new_LS_fs])
		LS_failures_g1             = np.vstack([LS_failures_g1,            new_LS_failures_g1])
		LS_failures_g2             = np.vstack([LS_failures_g2,            new_LS_failures_g2])

		return [ms, seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_failures_g2, 
				LS_solutions_found, LS_fs, LS_failures_g1, LS_failures_g2]


def stderror(v):
	non_nan = np.count_nonzero(~np.isnan(v))        # number of valid (non NaN) elements in the vector
	return np.nanstd(v, ddof=1) / np.sqrt(non_nan)

def saveToCSV(ms, resultsQSA, resultsLS, filename):
	nCols = resultsQSA.shape[1]


	# The output CSV file will have columns corresponding to: 
	#     (1) m; (2) QSA mean value; (3) QSA standard error bar size; (4) LS mean value; (5) LS standard error bar size
	with open(filename, mode='w') as file:
		writer = csv.writer(file, delimiter=',')

		# There will be one column per value of m (amount of training data)
		for col in range(nCols):

			cur_m          = ms[col]
			seldonian_data = resultsQSA[:,col]
			LS_data        = resultsLS[:,col]

			non_nan = np.count_nonzero(~np.isnan(seldonian_data))
			if non_nan > 0:
				seldonian_mean     = np.nanmean(seldonian_data)
				seldonian_stderror = stderror(seldonian_data)
			else:
				seldonian_mean     = 'NaN'
				seldonian_stderror = 'NaN'
			
			LS_mean     = np.mean(LS_data)
			LS_stderror = stderror(LS_data)

			writer.writerow([cur_m, seldonian_mean, seldonian_stderror, LS_mean, LS_stderror])


def gather_results():
	ms                        = None
	seldonian_solutions_found = None
	seldonian_fs              = None
	seldonian_failures_g1     = None
	seldonian_failures_g2     = None
	LS_solutions_found        = None
	LS_fs                     = None
	LS_failures_g1            = None
	LS_failures_g2            = None

	experiment_numbers = get_existing_experiment_numbers()

	for file_idx in experiment_numbers:
		res = addMoreResults(file_idx, 
			ms, 
			seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_failures_g2, 
			LS_solutions_found, LS_fs, LS_failures_g1, LS_failures_g2)	
		[ms, 
		seldonian_solutions_found, seldonian_fs, seldonian_failures_g1, seldonian_failures_g2, 
		LS_solutions_found, LS_fs, LS_failures_g1, LS_failures_g2] = res


	saveToCSV(ms, -seldonian_fs,              -LS_fs,              csv_path+'fs.csv') # here, negative to return MSE rather than negative MSE
	saveToCSV(ms,  seldonian_solutions_found,  LS_solutions_found, csv_path+'solutions_found.csv')
	saveToCSV(ms,  seldonian_failures_g1,      LS_failures_g1,     csv_path+'failures_g1.csv')
	saveToCSV(ms,  seldonian_failures_g2,      LS_failures_g2,     csv_path+'failures_g2.csv')

