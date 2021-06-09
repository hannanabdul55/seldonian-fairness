from helper import *


# Generate numPoints data points
def generateData(numPoints):
	X =     np.random.normal(0.0, 1.0, numPoints) # Sample x from a standard normal distribution
	Y = X + np.random.normal(0.0, 1.0, numPoints) # Set y to be x, plus noise from a standard normal distribution
	return (X,Y)


# Uses the weights in theta to predict the output value, y, associated with the provided x.
# This function assumes we are performing linear regression, so that theta has
# two elements: the y-intercept (first parameter) and slope (second parameter)
def predict(theta, x):
	return theta[0] + theta[1] * x


# Estimator of the primary objective, in this case, the negative sample mean squared error
def fHat(theta, X, Y):
	n = X.size          # Number of points in the data set
	res = 0.0           # Used to store the sample MSE we are computing
	for i in range(n):  # For each point X[i] in the data set ...
		prediction = predict(theta, X[i])                # Get the prediction using theta
		res += (prediction - Y[i]) * (prediction - Y[i]) # Add the squared error to the result
	res /= n            # Divide by the number of points to obtain the sample mean squared error
	return -res         # Returns the negative sample mean squared error


# Returns unbiased estimates of g_1(theta), computed using the provided data
def gHat1(theta, X, Y):
	n = X.size          # Number of points in the data set
	res = np.zeros(n)   # We will get one estimate per point; initialize res to store these estimates
	for i in range(n):
		prediction = predict(theta, X[i])                   # Compute the prediction for the i-th data point
		res[i] = (prediction - Y[i]) * (prediction - Y[i])  # Compute the squared error for the i-th data point
	res = res - 2.0     # We want the MSE to be less than 2.0, so g(theta) = MSE-2.0
	return res


# Returns unbiased estimates of g_2(theta), computed using the provided data
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




if __name__ == "__main__":

	np.random.seed(0)  # Create the random number generator to use, with seed zero
	numPoints = 5000   # Let's use 5000 points

	(X,Y)  = generateData(numPoints)  # Generate the data

	# Create the behavioral constraints - each is a gHat function and a confidence level delta
	gHats  = [gHat1, gHat2] # The 1st gHat requires MSE < 2.0. The 2nd gHat requires MSE > 1.25
	deltas = [0.1, 0.1]

	(result, found) = QSA(X, Y, gHats, deltas) # Run the Quasi-Seldonian algorithm
	if found:
		print("A solution was found: [%.10f, %.10f]" % (result[0], result[1]))
		print("fHat of solution (computed over all data, D):", fHat(result, X, Y))
	else:
		print("No solution found")
  