from functools import total_ordering
from math import copysign
from includes import *
from helpers import *
from windfarmQUBO import *

# Much the other same as the others solver, but we bond two variables at the start, and then store a parameter for their values

def step(x, t):
    '''
    if x == 0.0:
        return 0.0
    abs_x = abs(x)
    sign = math.copysign(1.0, x)
    if abs_x < 1.0:
        base = 1.0 - abs_x
        base_pow = base
        for _ in range(1, t):
            base_pow *= base
        return sign * (1.0 - base_pow)
    else:
        return sign
    '''
    return np.tanh(t*x)

def Sim(circ, shots):
    #print(circ.draw())
    backend = AerSimulator()
    transpiled_circ = transpile(circ, backend, optimization_level=1)
    job = backend.run(transpiled_circ, shots=shots)
    result = job.result()

    return result.get_counts(circ)

def multiQubitMeasure_every_qubit(thetas, totalQubits, shots):
    qubits = list(range(len(thetas)))
    lenLists = len(qubits)
    expectedValues = [0.0] * (2 * lenLists)
    # Z parts
    circ = QuantumCircuit(totalQubits, lenLists)
    for i in range(lenLists):
        circ.ry(thetas[i], qubits[i])
        circ.measure(qubits[i], i)
    counts = Sim(circ, shots)
    for key in counts:
        for k in range(lenLists):
            if key[-(k+1)] == '1':
                expectedValues[2 * k] -= counts[key] / shots
            else:
                expectedValues[2 * k] += counts[key] / shots

    # X parts
    circ = QuantumCircuit(totalQubits, lenLists)
    for i in range(lenLists):
        circ.ry(0.3*(thetas[i]-3.5), qubits[i])
        circ.h(qubits[i])
        circ.measure(qubits[i], i)
    counts = Sim(circ, shots)
    for key in counts:
        for k in range(lenLists):
            if key[-(k+1)] == '1':
                expectedValues[2 * k + 1] -= counts[key] / shots
            else:
                expectedValues[2 * k + 1] += counts[key] / shots
    return expectedValues

def localGradient(J, h, loc1, loc2, x, xMinus, xPlus, stepSize):
    if loc1 >= len(x) or loc2 >= len(x):
        return 0.0
    else:
        costPlus, costMinus, cost, n = 0, 0, 0, len(x)
        for i in range(n):
            if i < loc1:
                costPlus += J[i, loc1] * xPlus[loc1] * x[i]
                costMinus += J[i, loc1] * xMinus[loc1] * x[i]
                cost += J[i, loc2] * x[loc2] * x[i]
            else:
                costPlus += J[loc1, i] * xPlus[loc1] * x[i]
                costMinus += J[loc1, i] * xMinus[loc1] * x[i]
                cost += J[loc2, i] * x[loc2] * x[i]
        for j in range(n):
            if j < loc2:
                costPlus += J[j, loc2] * xPlus[loc2] * x[j]
                costMinus += J[j, loc2] * xMinus[loc2] * x[j]
                cost += J[loc1, j] * x[loc1] * x[j]
            else:
                costPlus += J[loc2, j] * xPlus[loc2] * x[j]
                costMinus += J[loc2, j] * xMinus[loc2] * x[j]
                cost += J[loc1, j] * x[loc1] * x[j]
        costPlus += h[loc1] * xPlus[loc1] + h[loc2] * xPlus[loc2]
        costMinus += h[loc1] * xMinus[loc1] + h[loc2] * xMinus[loc2]
        cost += h[loc1] * x[loc1] + h[loc2] * x[loc2]

        return (costPlus - costMinus) / (2 * stepSize)

def multiQubitMeasure(thetas, totalQubits, shots):
    qubits = [2 * q for q in range(len(thetas))]
    lenLists = len(qubits)
    expectedValues = [0.0] * totalQubits
    # Z parts
    circ = QuantumCircuit(totalQubits,totalQubits)
    for i in range(lenLists):
        circ.ry(thetas[i], qubits[i])
        circ.measure(qubits[i], qubits[i])
    counts = Sim(circ, shots)
    for key in counts:
        for k in range(lenLists):
            if key[-(qubits[k]+1)] == '1':
                expectedValues[2 * k] -= counts[key]/shots
            else:
                expectedValues[2 * k] += counts[key]/shots
    # X parts
    circ = QuantumCircuit(totalQubits,totalQubits)
    for i in range(lenLists):
        circ.ry(0.3*thetas[i]-3.5, qubits[i])
        circ.h(qubits[i])
        circ.measure(qubits[i], qubits[i])
    counts = Sim(circ, shots)
    for key in counts:
        for k in range(lenLists):
            if key[-(qubits[k]+1)] == '1':
                expectedValues[2 * k + 1] -= counts[key]/shots
            else:
                expectedValues[2 * k + 1] += counts[key]/shots
    return expectedValues

def combinedGradients(J, h, locs, x, stepSize, shots, t, totalQubits, thetas, all_qubits = False):
    n = len(x)
    C = len(locs)
    c = int(len(locs)/2)
    gradients = []

    thetasPlus = thetas + stepSize
    thetasMinus = thetas - stepSize
    thetasPlus = np.mod(thetasPlus, 2 * np.pi)
    thetasMinus = np.mod(thetasMinus, 2 * np.pi)
    tok = time.perf_counter()
    if all_qubits:
        expectedValuesMinus = multiQubitMeasure_every_qubit(thetasMinus, totalQubits, shots)
        expectedValuesPlus = multiQubitMeasure_every_qubit(thetasPlus, totalQubits, shots)
    else:
        expectedValuesPlus = multiQubitMeasure(thetasPlus, totalQubits, shots)
        expectedValuesMinus = multiQubitMeasure(thetasMinus, totalQubits, shots)
    xPlus = copy.copy(x)
    xMinus = copy.copy(x)
    tik = time.perf_counter()
    sim_time = tik-tok
    for s in range(c):
        loc1 = locs[2 * s]
        loc2 = locs[2 * s + 1]
        if loc1 < n:
            xPlus[loc1] = step(expectedValuesPlus[2 * s], t)
            xMinus[loc1] = step(expectedValuesMinus[2 * s], t)
        if loc2 < n:
            xPlus[loc2] = step(expectedValuesPlus[2 * s + 1], t)
            xMinus[loc2] = step(expectedValuesMinus[2 * s + 1], t)

    for s in range(c):
        loc1 = locs[2 * s]
        loc2 = locs[2 * s + 1]

        gradients.append(localGradient(J, h, loc1, loc2, x, xMinus, xPlus, stepSize))

    return np.array(gradients), sim_time

def updatex_new(thetas, locs, x, totalQubits, shots, t):
    c = len(locs) // 2
    qubits = locs
    expectedValues = multiQubitMeasure_every_qubit(thetas, totalQubits, shots)
    n = len(x)
    for k in range(c):
        loc1 = locs[2 * k]
        loc2 = locs[2 * k + 1]
        if loc1 < n:
            x[loc1] = step(expectedValues[2 * k], t)
        if loc2 < n:
            x[loc2] = step(expectedValues[2 * k + 1], t)
    return x

def makeBonds(J):
    n = len(J)
    indices = list(range(n))
    bonds = {}
    for i in range(int(n/2)):
        bonds[i] = (indices[2 * i], indices[2 * i + 1])
    return bonds

def makeThetas(J):
    n = len(J)
    thetas = {}
    for i in range(n):
        thetas[i] = 1.0 #np.random.uniform(0, 2 * np.pi)
    return thetas

def generateRandomIndices2(n, samples):
    pairs = [(i, i+1) for i in range(0, n, 2)]
    np.random.shuffle(pairs)
    selected_pairs = pairs[:samples // 2]
    # Flatten the list of pairs
    indices = [idx for pair in selected_pairs for idx in pair]
    return indices

def WFLOmakeThetas(n, m, p):
    thetas = {}
    for i in range(int(n/2) + (n % 2)):
        thetas[2 * i] = 2.0
        thetas[2 * i + 1] = 2.0

    poss = {(-1,1) : 0.5,
            (-1,-1): 2.0,
            (1,-1): 4.0,
            (1,1): 5.5}
    mtemp = m
    indices = list(range(int(np.ceil(n/2))))
    shuffled_indices = np.random.permutation(indices)
    for k in shuffled_indices:
        if 2*k not in p and (2*k+1) not in p:
            if mtemp > 1:
                thetas[2 * k] = poss[(1, 1)]
                thetas[2 * k + 1] = poss[(1, 1)]
                mtemp -= 2
            elif mtemp == 1:
                thetas[2 * k] = poss[(1, -1)]
                thetas[2 * k + 1] = poss[(1, -1)]
                mtemp -= 1
    return thetas

def SQOE_WFLOsolver(parameters):
    Q = WindfarmQ(parameters)
    m = parameters['m']
    p = parameters['P']
    stepSize = parameters['stepSize']
    learningRate = parameters['learningRate']
    t = parameters['t']
    totalQubits = parameters['totalQubits']
    shots = parameters['shots']
    maxIters = parameters['maxiter']
    all_qubits = parameters['allQubits']

    h, J = QuboToIsing(Q)

    x = np.array([-1.0] * len(h))

    n = len(h)
    indices = list(range(n))
    stop = False
    iterations = 0
    bonds = makeBonds(J)
    if parameters['warmStart']:
        thetas = WFLOmakeThetas(n, m, p)
    else:
        thetas = makeThetas(J)

    for i in range(0, len(indices), totalQubits):
        chunk_indices = indices[i:i+totalQubits]
        chunk_paras = [thetas[chunk_indices[j]] for j in range(0, len(chunk_indices), 2)]
        x = updatex_new(chunk_paras, chunk_indices, x, totalQubits, shots, t)
    if all_qubits : samples = 2 * totalQubits
    else : samples = totalQubits

    xs, costs = [x], [x.T @ J @ x + h.T @ x]

    minCost = 100.0
    bestx = copy.copy(x)
    tok = time.perf_counter()
    grads_store = [[] for _ in range(len(thetas))]
    paras_store = [[] for _ in range(len(thetas))]
    timeTakenPerIter = []
    c = samples // 2
    sim_times = []
    while not stop:
        step_size = np.random.uniform(0.05, 1.5)
        tokIter = time.perf_counter()
        locs = generateRandomIndices2(n, samples)
        paras = np.array([thetas[locs[2*k]] for k in range(c)])
        gradients, sim_time = combinedGradients(J, h, locs, x, step_size, shots, t, totalQubits, paras, all_qubits)
        paras -= learningRate * gradients
        paras = np.mod(paras, 2 * np.pi)
        for k in range(c):
            thetas[locs[2*k]] = paras[k]
            thetas[locs[2*k+1]] = paras[k]
        x_new = updatex_new(paras, locs, x, totalQubits, shots, t)
        x = copy.copy(x_new)
        iterations += 1
        sim_times.append(sim_time)
        if iterations > maxIters:
            endReason = "maxIters"
            stop = True
        '''
        if (iterations % 10*n) == 0:
            #if gradients.T @ gradients < 1e-1:
            #    endReason = "gradientsTol"
            #    stop = True

            cost = x.T @ J @ x + h.T @ x
            #if abs(cost - min(costs)) < 1:
            #    endReason = "costTol"
            #    stop = True
            costs.append(cost)
            xs.append(x)
        '''
        
        # INSIDE LOOP — after updating thetas from paras
        for k in range(c):
            loc1 = locs[2*k]
            loc2 = locs[2*k + 1]
            grad = gradients[k]
            
            if loc1 < len(thetas):
                grads_store[loc1].append(grad)
                paras_store[loc1].append(thetas[loc1])
            if loc2 < len(thetas):
                grads_store[loc2].append(grad)
                paras_store[loc2].append(thetas[loc2])
        # ===================================
        # This is just to keep track
        cost = x.T @ J @ x + h.T @ x
        #print(f"Iteration {iterations}, Cost: {cost}")
        costs.append(cost)
        # ===================================
        tikIter = time.perf_counter()
        timeTakenPerIter.append(round(tikIter - tokIter, 5))
    tik = time.perf_counter()
    timeTaken = round(tik - tok, 5)
    if len(xs) > 1:
        x = xs[np.argmin(costs)]
        X = [copysign(1,l) for l in x]
    else:
        X = [copysign(1,l) for l in x]
    return X, grads_store, timeTaken, timeTakenPerIter, xs, paras_store, sim_times, costs
    
def SQOE_WFLOsolver_changingQubits(parameters):
    Q = WindfarmQ(parameters)
    m = parameters['m']
    p = parameters['P']
    stepSize = parameters['stepSize']
    learningRate = parameters['learningRate']
    t = parameters['t']
    totalQubits = parameters['totalQubits']
    shots = parameters['shots']
    maxIters = parameters['maxiter']
    all_qubits = parameters['allQubits']

    h, J = QuboToIsing(Q)

    x = np.array([-1.0] * len(h))

    n = len(h)
    indices = list(range(n))
    stop = False
    iterations = 0
    bonds = makeBonds(J)
    if parameters['warmStart']:
        thetas = WFLOmakeThetas(n, m, p)
    else:
        thetas = makeThetas(J)

    for i in range(0, len(indices), totalQubits):
        chunk_indices = indices[i:i+totalQubits]
        chunk_paras = [thetas[chunk_indices[j]] for j in range(0, len(chunk_indices), 2)]
        x = updatex_new(chunk_paras, chunk_indices, x, totalQubits, shots, t)
    

    xs, costs = [x], [x.T @ J @ x + h.T @ x]

    minCost = 100.0
    bestx = copy.copy(x)
    tok = time.perf_counter()
    grads = []
    timeTakenPerIter = []
    while not stop:
        totalQubits = np.random.randint(2,int(n/2))
        if all_qubits : samples = 2 * totalQubits
        else : samples = totalQubits
        c = samples // 2
        
        tokIter = time.perf_counter()
        locs = generateRandomIndices2(n, samples)
        paras = np.array([thetas[locs[2*k]] for k in range(c)])
        gradients, _ = combinedGradients(J, h, locs, x, stepSize, shots, t, totalQubits, paras, all_qubits)
        paras -= learningRate * gradients
        paras = np.mod(paras, 2 * np.pi)
        for k in range(c):
            thetas[locs[2*k]] = paras[k]
            thetas[locs[2*k+1]] = paras[k]
        x = updatex_new(paras, locs, x, totalQubits, shots, t)
        iterations += 1
        if iterations > maxIters:
            endReason = "maxIters"
            stop = True

        if (iterations % 10*n) == 0:
            '''
            if gradients.T @ gradients < 1e-1:
                endReason = "gradientsTol"
                stop = True

            cost = x.T @ J @ x + h.T @ x
            if abs(cost - min(costs)) < 5:
                endReason = "costTol"
                stop = True
            '''
            costs.append(cost)
            xs.append(x)
        grads.append(gradients)
        # ===================================
        # This is just to keep track
        #cost = x.T @ J @ x + h.T @ x
        #print(f"Iteration {iterations}, Cost: {cost}")

        # ===================================
        tikIter = time.perf_counter()
        timeTakenPerIter.append(round(tikIter - tokIter, 5))
    tik = time.perf_counter()
    timeTaken = round(tik - tok, 5)
    if len(xs) > 1:
        x = xs[np.argmin(costs)]
        X = [copysign(1,l) for l in x]
    else:
        X = [copysign(1,l) for l in x]
    return X, grads, timeTaken, timeTakenPerIter, xs, endReason