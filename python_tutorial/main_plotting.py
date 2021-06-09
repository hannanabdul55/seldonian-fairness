from helper import *        # Basic helper functions
import timeit               # To time the execution of ours experiments
import ray                  # To allow us to execute experiments in parallel
ray.init()    
from numba import jit       # Just-in-Time (JIT) compiler to accelerate Python code

# Folder where the experiment results will be saved
bin_path = 'experiment_results/bin/'


# Generate numPoints data points
@jit(nopython=True)
def generateData(numPoints):
	X =     np.random.normal(0.0, 1.0, numPoints) # Sample x from a standard normal distribution
	Y = X + np.random.normal(0.0, 1.0, numPoints) # Set y to be x, plus noise from a standard normal distribution
	return (X,Y)


# Uses the weights in theta to predict the output value, y, associated with the provided x.
# This function assumes we are performing linear regression, so that theta has
# two elements: the y-intercept (first parameter) and slope (second parameter)
@jit(nopython=True)
def predict(theta, x):
	return theta[0] + theta[1] * x


# Estimator of the primary objective, in this case, the negative sample mean squared error
@jit(nopython=True)
def fHat(theta, X, Y):
	n = X.size          # Number of points in the data set
	res = 0.0           # Used to store the sample MSE we are computing
	for i in range(n):  # For each point X[i] in the data set ...
		prediction = predict(theta, X[i])                # Get the prediction using theta
		res += (prediction - Y[i]) * (prediction - Y[i]) # Add the squared error to the result
	res /= n            # Divide by the number of points to obtain the sample mean squared error
	return -res         # Returns the negative sample mean squared error


# Returns unbiased estimates of g_1(theta), computed using the provided data
@jit(nopython=True)
def gHat1(theta, X, Y):
	n = X.size          # Number of points in the data set
	res = np.zeros(n)   # We will get one estimate per point; initialize res to store these estimates
	for i in range(n):
		prediction = predict(theta, X[i])                   # Compute the prediction for the i-th data point
		res[i] = (prediction - Y[i]) * (prediction - Y[i])  # Compute the squared error for the i-th data point
	res = res - 2.0     # We want the MSE to be less than 2.0, so g(theta) = MSE-2.0
	return res


# Returns unbiased estimates of g_2(theta), computed using the provided data
@jit(nopython=True)
def gHat2(theta, X, Y):
	n = X.size          # Number of points in the data set
	res = np.zeros(n)   # We will get one estimate per point; initialize res to store these estimates
	for i in range(n):
		prediction = predict(theta, X[i])                   # Compute the prediction for the i-th data point
		res[i] = (prediction - Y[i]) * (prediction - Y[i])  # Compute the squared error for the i-th data point
	res = 1.25 - res    # We want the MSE to be at least 1.25, so g(theta) = 1.25-MSE
	return res


# Run ordinary least squares linear regression on data (X,Y)
def leastSq(X, Y):
	X = np.expand_dims(X, axis=1) # Places the input  data in a matrix
	Y = np.expand_dims(Y, axis=1) # Places the output data in a matrix
	reg = LinearRegression().fit(X, Y)
	theta0 = reg.intercept_[0]   # Gets theta0, the y-intercept coefficient
	theta1 = reg.coef_[0][0]     # Gets theta0, the slope coefficient
	return np.array([theta0, theta1])


# Our Quasi-Seldonian linear regression algorithm operating over data (X,Y).
# The pair of objects returned by QSA is the solution (first element) 
# and a Boolean flag indicating whether a solution was found (second element).
def QSA(X, Y, gHats, deltas):
	# Put 40% of the data in candidateData (D1), and the rest in safetyData (D2)
	candidateData_len = 0.40
	candidateData_X, safetyData_X, candidateData_Y, safetyData_Y = train_test_split(
								X, Y, test_size=1-candidateData_len, shuffle=False)
	
	# Get the candidate solution
	candidateSolution = getCandidateSolution(candidateData_X, candidateData_Y, gHats, deltas, safetyData_X.size)

	# Run the safety test
	passedSafety      = safetyTest(candidateSolution, safetyData_X, safetyData_Y, gHats, deltas)

	# Return the result and success flag
	return [candidateSolution, passedSafety]


# Run the safety test on a candidate solution. Returns true if the test is passed.
#   candidateSolution: the solution to test. 
#   (safetyData_X, safetyData_Y): data set D2 to be used in the safety test.
#   (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
def safetyTest(candidateSolution, safetyData_X, safetyData_Y, gHats, deltas):

	for i in range(len(gHats)):	# Loop over behavioral constraints, checking each
		g         = gHats[i]	# The current behavioral constraint being checked
		delta     = deltas[i]	# The confidence level of the constraint

		# This is a vector of unbiased estimates of g(candidateSolution)
		g_samples = g(candidateSolution, safetyData_X, safetyData_Y) 

		# Check if the i-th behavioral constraint is satisfied
		upperBound = ttestUpperBound(g_samples, delta) 

		if upperBound > 0.0: # If the current constraint was not satisfied, the safety test failed
			return False

	# If we get here, all of the behavioral constraints were satisfied			
	return True


# The objective function maximized by getCandidateSolution.
#     thetaToEvaluate: the candidate solution to evaluate.
#     (candidateData_X, candidateData_Y): the data set D1 used to evaluated the solution.
#     (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#     safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
def candidateObjective(thetaToEvaluate, candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize):	

	# Get the primary objective of the solution, fHat(thetaToEvaluate)
	result = fHat(thetaToEvaluate, candidateData_X, candidateData_Y)

	predictSafetyTest = True     # Prediction of what the safety test will return. Initialized to "True" = pass
	for i in range(len(gHats)):  # Loop over behavioral constraints, checking each
		g         = gHats[i]       # The current behavioral constraint being checked
		delta     = deltas[i]      # The confidence level of the constraint

		# This is a vector of unbiased estimates of g_i(thetaToEvaluate)
		g_samples = g(thetaToEvaluate, candidateData_X, candidateData_Y)

		# Get the conservative prediction of what the upper bound on g_i(thetaToEvaluate) will be in the safety test
		upperBound = predictTTestUpperBound(g_samples, delta, safetyDataSize)

		# We don't think the i-th constraint will pass the safety test if we return this candidate solution
		if upperBound > 0.0:

			if predictSafetyTest:
				# Set this flag to indicate that we don't think the safety test will pass
				predictSafetyTest = False  

				# Put a barrier in the objective. Any solution that we think will fail the safety test will have a
				# large negative performance associated with it
				result = -100000.0    

			# Add a shaping to the objective function that will push the search toward solutions that will pass 
			# the prediction of the safety test
			result = result - upperBound

	# Negative because our optimizer (Powell) is a minimizer, but we want to maximize the candidate objective
	return -result  


# Use the provided data to get a candidate solution expected to pass the safety test.
#    (candidateData_X, candidateData_Y): data used to compute a candidate solution.
#    (gHats, deltas): vectors containing the behavioral constraints and confidence levels.
#    safetyDataSize: |D2|, used when computing the conservative upper bound on each behavioral constraint.
def getCandidateSolution(candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize):
	
	# Chooses the black-box optimizer we will use (Powell)
	minimizer_method = 'Powell'
	minimizer_options={'disp': False}

	# Initial solution given to Powell: simple linear fit we'd get from ordinary least squares linear regression
	initialSolution  = leastSq(candidateData_X, candidateData_Y)

	# Use Powell to get a candidate solution that tries to maximize candidateObjective
	res = minimize(candidateObjective, x0=initialSolution, method=minimizer_method, options=minimizer_options, 
		args=(candidateData_X, candidateData_Y, gHats, deltas, safetyDataSize))

	# Return the candidate solution we believe will pass the safety test
	return res.x


@ray.remote
def run_experiments(worker_id, nWorkers, ms, numM, numTrials, mTest):

	# Results of the Seldonian algorithm runs
	seldonian_solutions_found = np.zeros((numTrials, numM)) # Stores whether a solution was found (1=True,0=False)
	seldonian_failures_g1     = np.zeros((numTrials, numM)) # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1
	seldonian_failures_g2     = np.zeros((numTrials, numM)) # Stores whether solution was unsafe, (1=True,0=False), for the 2nd constraint, g_2
	seldonian_fs              = np.zeros((numTrials, numM)) # Stores the primary objective values (fHat) if a solution was found

	# Results of the Least-Squares (LS) linear regression runs
	LS_solutions_found = np.ones((numTrials, numM))  # Stores whether a solution was found. These will all be true (=1)
	LS_failures_g1     = np.zeros((numTrials, numM)) # Stores whether solution was unsafe, (1=True,0=False), for the 1st constraint, g_1
	LS_failures_g2     = np.zeros((numTrials, numM)) # Stores whether solution was unsafe, (1=True,0=False), for the 2nd constraint, g_2
	LS_fs              = np.zeros((numTrials, numM)) # Stores the primary objective values (f) if a solution was found

	# Prepares file where experiment results will be saved
	experiment_number = worker_id
	outputFile = bin_path + 'results%d.npz' % experiment_number
	print("Writing output to", outputFile)
	
	# Generate the data used to evaluate the primary objective and failure rates
	np.random.seed( (experiment_number+1) * 9999 )	
	(testX, testY) = generateData(mTest) 

	for trial in range(numTrials):
		for (mIndex, m) in enumerate(ms):

			# Generate the training data, D
			base_seed         = (experiment_number * numTrials)+1
			np.random.seed(base_seed+trial) # done to obtain common random numbers for all values of m			
			(trainX, trainY)  = generateData(m)

			# Run the Quasi-Seldonian algorithm
			(result, passedSafetyTest) = QSA(trainX, trainY, gHats, deltas)
			if passedSafetyTest:
				seldonian_solutions_found[trial, mIndex] = 1
				trueMSE = -fHat(result, testX, testY)                               # Get the "true" mean squared error using the testData
				seldonian_failures_g1[trial, mIndex] = 1 if trueMSE > 2.0  else 0   # Check if the first behavioral constraint was violated
				seldonian_failures_g2[trial, mIndex] = 1 if trueMSE < 1.25 else 0	# Check if the second behavioral constraint was violated
				seldonian_fs[trial, mIndex] = -trueMSE                              # Store the "true" negative mean-squared error
				print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] A solution was found: [{result[0]:.10f}, {result[1]:.10f}]\tfHat over test data: {trueMSE:.10f}")
			else:
				seldonian_solutions_found[trial, mIndex] = 0             # A solution was not found
				seldonian_failures_g1[trial, mIndex]     = 0             # Returning NSF means the first constraint was not violated
				seldonian_failures_g2[trial, mIndex]     = 0             # Returning NSF means the second constraint was not violated
				seldonian_fs[trial, mIndex]              = None          # This value should not be used later. We use None and later remove the None values
				print(f"[(worker {worker_id}/{nWorkers}) Seldonian trial {trial+1}/{numTrials}, m {m}] No solution found")

			# Run the Least Squares algorithm
			theta = leastSq(trainX, trainY)                              # Run least squares linear regression
			trueMSE = -fHat(theta, testX, testY)                         # Get the "true" mean squared error using the testData
			LS_failures_g1[trial, mIndex] = 1 if trueMSE > 2.0  else 0   # Check if the first behavioral constraint was violated
			LS_failures_g2[trial, mIndex] = 1 if trueMSE < 1.25 else 0   # Check if the second behavioral constraint was violated
			LS_fs[trial, mIndex] = -trueMSE                              # Store the "true" negative mean-squared error
			print(f"[(worker {worker_id}/{nWorkers}) LeastSq   trial {trial+1}/{numTrials}, m {m}] LS fHat over test data: {trueMSE:.10f}")
		print()

	np.savez(outputFile, 
			 ms=ms, 
			 seldonian_solutions_found=seldonian_solutions_found,
			 seldonian_fs=seldonian_fs, 
			 seldonian_failures_g1=seldonian_failures_g1, 
			 seldonian_failures_g2=seldonian_failures_g2,
			 LS_solutions_found=LS_solutions_found,
			 LS_fs=LS_fs,
			 LS_failures_g1=LS_failures_g1,
			 LS_failures_g2=LS_failures_g2)



if __name__ == "__main__":
  
	# Create the behavioral constraints: each is a gHat function and a confidence level delta
	gHats  = [gHat1, gHat2]
	deltas = [0.1, 0.1]

	if len(sys.argv) < 2:
		print("\nUsage: python main_plotting.py [number_threads]")
		print("       Assuming the default: 16")
		nWorkers = 16                # Workers is the number of threads running experiments in parallel
	else:
		nWorkers = int(sys.argv[1])  # Workers is the number of threads running experiments in parallel
	print(f"Running experiments on {nWorkers} threads")

	# We will use different amounts of data, m. The different values of m will be stored in ms.
	# These values correspond to the horizontal axis locations in all three plots we will make.
	# We will use a logarithmic horizontal axis, so the amounts of data we use shouldn't be evenly spaced.
	ms   = [2**i for i in range(5, 17)]  # ms = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
	numM = len(ms)

	# How many trials should we average over?
	numTrials = 70 # We pick 70 because with 70 trials per worker, and 16 workers, we get >1000 trials for each value of m
	
	# How much data should we generate to compute the estimates of the primary objective and behavioral constraint function values 
	# that we call "ground truth"? Each candidate solution deemed safe, and identified using limited training data, will be evaluated 
	# over this large number of points to check whether it is really safe, and to compute its "true" mean squared error.
	mTest = ms[-1] * 100 # about 5,000,000 test samples

	# Start 'nWorkers' threads in parallel, each one running 'numTrials' trials. Each thread saves its results to a file
	tic = timeit.default_timer()
	_ = ray.get([run_experiments.remote(worker_id, nWorkers, ms, numM, numTrials, mTest) for worker_id in range(1,nWorkers+1)])
	toc = timeit.default_timer()
	time_parallel = toc - tic # Elapsed time in seconds
	print(f"Time ellapsed: {time_parallel}")
