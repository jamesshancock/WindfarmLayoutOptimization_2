from includes import *
from helpers import *
from windfarmQUBO import *

def ExpectedValue(counts, shots):
    """
    Calculates the expected value of a Hamiltonian given the counts from a quantum circuit.
    """
    value = 0
    for key, count in counts.items():
        sign = 1
        for i in key:
            if i == '1':
                sign *= -1
        value += sign * count / shots
    return value

def ParametricCircuitPCE(circ, n, theta):
    """
    Adds parameterized Ry rotations and CNOTs to a circuit for PCE.
    """
    counter = 0
    totalParas = len(theta)
    nLayers = totalParas // n
    for _ in range(nLayers):
        for qubit in range(n):
            circ.ry(theta[counter], qubit)
            counter += 1
        for qubit in range(n - 1):
            circ.cx(qubit, qubit + 1)
    return circ

def create_parametric_circuitPCE(n, nPara, nLayers):
    """
    Creates a parameterized circuit for PCE with Ry and Rx layers and CNOT entanglement.
    """
    theta_params = [Parameter(f'theta_{i}') for i in range(nPara)]
    circ = QuantumCircuit(n, n)
    counter = 0
    for _ in range(nLayers):
        for qubit in range(n):
            circ.ry(theta_params[counter], qubit)
            counter += 1
        for qubit in range(n):
            circ.rx(theta_params[counter], qubit)
            counter += 1
        for qubit in range(n - 1):
            circ.cx(qubit, qubit + 1)
    return circ, theta_params

def NParaPCE(n, K):
    """
    Returns the number of parameters and layers for PCE.
    """
    k = 1
    while 2 * n * k <= 2 * n * K:
        k += 1
    return 2 * n * k, k

def Sim(circ, shots):
    """
    Runs the quantum circuit on a local Aer simulator and returns the counts.
    """
    backend = AerSimulator()
    transpiled_circ = transpile(circ, backend, optimization_level=3)
    job = backend.run(transpiled_circ, shots=shots)
    result = job.result()
    return result.get_counts(circ)

def ExpectedValueForKey(theta, quantumParameters, key, parametric_circuit, theta_params):
    """
    Calculates the expected value of the Hamiltonian for a given key and parameters.
    """
    n = quantumParameters['nPCE']
    shots = quantumParameters['shots']
    circ = parametric_circuit.assign_parameters({theta_params[i]: theta[i] for i in range(len(theta))})
    for j in range(n):
        if key[j] == '3':
            circ.measure(j, j)
        elif key[j] == '1':
            circ.h(j)
            circ.measure(j, j)
        elif key[j] == '2':
            circ.sdg(j)
            circ.h(j)
            circ.measure(j, j)
    counts = Sim(circ, shots)
    return np.tanh(quantumParameters['talpha'] * ExpectedValue(counts, shots))

def PCE(theta, parameters, keys, Jprime, hPCE, parametric_circuit, theta_params):
    """
    Calculates the PCE cost function.
    """
    store = [ExpectedValueForKey(theta, parameters, key, parametric_circuit, theta_params) for key in keys]
    cost = 0
    for i in range(Jprime.shape[0]):
        for j in range(i + 1, Jprime.shape[0]):
            cost += Jprime[i, j] * store[i] * store[j]
        cost += hPCE[i] * store[i]
    return cost

def findCombinations(hKeys):
    """
    Finds compatible sets of measurement keys for parallel measurement.
    """
    n = len(hKeys[0])
    measureLocs = {key: [i for i, char in enumerate(key) if char != '0'] for key in hKeys}
    compatables = {}
    for key in hKeys:
        locs = measureLocs[key]
        compatables[key] = []
        for keyTest in hKeys:
            if key != keyTest:
                locsTest = measureLocs[keyTest]
                if not any(abs(loc - locTest) <= 1 for loc in locs for locTest in locsTest):
                    compatables[key].append(keyTest)
    compatible_sets = []
    keys_left = set(hKeys)
    while keys_left:
        key = keys_left.pop()
        current_set = {key}
        to_check = set(compatables[key])
        while to_check:
            test_key = to_check.pop()
            if test_key in keys_left and all(test_key in compatables[existing_key] for existing_key in current_set):
                current_set.add(test_key)
                keys_left.remove(test_key)
                to_check.update(compatables[test_key])
        compatible_sets.append(current_set)
    combinedOps = []
    for set1 in compatible_sets:
        string = '0' * n
        for i in range(n):
            for op in set1:
                if op[i] != '0':
                    string = string[:i] + op[i] + string[i + 1:]
        combinedOps.append(string)
    return compatible_sets, combinedOps, measureLocs

def combinedEVs(theta, quantumParameters, shots, circ, theta_params, n, combinedOps, compatible_sets, measureLocs):
    """
    Calculates expected values for combined measurement operators.
    """
    counter = 0
    termEvs = {}
    for key in combinedOps:
        if key != '0' * n:
            bound_circ = circ.assign_parameters({theta_params[i]: theta[i] for i in range(len(theta_params))})
            for j in range(n):
                if key[j] == '1':
                    bound_circ.h(j)
                    bound_circ.measure(j, j)
                elif key[j] == '2':
                    bound_circ.sdg(j)
                    bound_circ.h(j)
                    bound_circ.measure(j, j)
                if key[j] == '3':
                    bound_circ.measure(j, j)
            counts = Sim(bound_circ, shots)
            set2 = compatible_sets[counter]
            counter += 1
            for key2 in set2:
                locs = measureLocs[key2]
                termEv = 0
                for countsKey in counts:
                    count = counts[countsKey]
                    sign = 1
                    countsKey = countsKey[::-1]
                    for loc in locs:
                        if countsKey[loc] == '1':
                            sign *= -1
                    termEv += sign * count / shots
                termEvs[key2] = termEv
    return termEvs

def combinedPCE(theta, parameters, keys, Jprime, hPCE, circ, theta_params, combinedOps, compatible_sets, measureLocs):
    """
    Calculates the PCE cost function using combined measurement operators.
    """
    n = len(keys[0])
    shots = parameters['shots']
    termEVs = combinedEVs(theta, parameters, shots, circ, theta_params, n, combinedOps, compatible_sets, measureLocs)
    store = [np.tanh(parameters['talpha'] * termEVs[key]) for key in keys]
    cost = 0
    for i in range(Jprime.shape[0]):
        for j in range(i + 1, Jprime.shape[0]):
            cost += Jprime[i, j] * store[i] * store[j]
        cost += hPCE[i] * store[i]
    return cost

def thetaToSolutionPCE(theta, parameters, keys, parametric_circuit, theta_params):
    """
    Converts the parameters theta to a solution for the PCE algorithm.
    """
    store = []
    parameters['machine'] = 'simulator'
    for key in keys:
        EV = ExpectedValueForKey(theta, parameters, key, parametric_circuit, theta_params)
        store.append(1 if EV > 0 else 0)
    return store

def PCE_Solver(parameters):
    """
    Solves the wind farm layout optimization problem using the combinedPCE quantum algorithm.

    Parameters:
        parameters (dict): A dictionary containing wind farm parameters.

    Returns:
        solution (list): A list representing the wind farm layout.
        history (list): List of cost function values per iteration.
        timeTaken (float): Total optimization time in seconds.
    """
    # Setup problem size and circuit
    if parameters['fixedK'][0]:
        parameters['nPCE'] = chooseNfixedK(parameters['len_grid'], parameters['fixedK'][1])
        parameters['kPCE'] = parameters['fixedK'][1]
    else:
        parameters['nPCE'], parameters['kPCE'] = chooseNandK(parameters['len_grid'])
    parameters['nParaPCE'], parameters['nLayersPCE'] = NParaPCE(parameters['nPCE'], parameters['kPCE'])
    keys = generateBinaryStringsWithCopies(parameters['nPCE'], parameters['kPCE'])
    keys = keys[:parameters['len_grid'] * parameters['len_grid']]
    hPCE, Jprime = QuboToIsing(WindfarmQ(parameters))
    #print("Number of qubits required:", parameters['nPCE'])
    #print("PCE variables:")
    #print(keys)
    theta = [np.pi * np.random.uniform(0, 2) for _ in range(parameters['nParaPCE'])]
    parametric_circuit, theta_params = create_parametric_circuitPCE(parameters['nPCE'], parameters['nParaPCE'], parameters['nLayersPCE'])
    compatible_sets, combinedOps, measureLocs = findCombinations(keys)
    simsPerRun = len(combinedOps)
    tol = parameters['tol']
    L = parameters['L']
    # Optimization setup
    history = []
    paras = []
    timeTaken = 0
    timeTakenPerIter = []
    shortHistory = []
    start = [time.perf_counter()]  # Use a list for mutability
    endReason = "didnt store - work out"
    def cost_function(theta):
        cost = combinedPCE(theta, parameters, keys, Jprime, hPCE, parametric_circuit, theta_params, combinedOps, compatible_sets, measureLocs)
        history.append(cost)
        return cost

    def callback(xk, L=L, tol=tol):
        end = time.perf_counter()
        timeTakenPerIter.append(end - start[0])
        start[0] = end
        parameters['talpha'] *= 1e4
        theta = xk if not hasattr(xk, 'x') else xk.x
        parameters['talpha'] /= 1e4
        paras.append(theta)

        #print("cost:",history[-1])

        if len(history) % (parameters['nParaPCE'] + 1) == 0:
            #print("-"*10)
            shortHistory.append(history[-1])
            if len(shortHistory) >= parameters['maxiter']:
                raise StopIteration("maxIters")
            if len(shortHistory) >= L:
                pass
            else:
                L = len(shortHistory)
            lowestLTerms = sorted(shortHistory)[:L]
            avRecents = sum(lowestLTerms) / len(lowestLTerms)
            last = shortHistory[-1]
            avg_change = abs(avRecents - last)
            if (avg_change <= tol) and (len(shortHistory) > 1):
                raise StopIteration("costTol")

    #print("Starting PCE WFLO solver")
    # Optimization loop
    tok = time.perf_counter()
    try:
        mini = minimize(cost_function, theta, method='COBYLA', callback=callback)
    except StopIteration as e:
        print(e)
        mini = type('obj', (object,), {'x': theta})
        endReason = e
    tik = time.perf_counter()

    finalParas = paras[np.argmin(history)]
    solution = thetaToSolutionPCE(finalParas, parameters, keys, parametric_circuit, theta_params)
    timeTaken = round(tik - tok,5)
    return solution, history, timeTaken, timeTakenPerIter, shortHistory, endReason