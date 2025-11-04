from includes import *

class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()

def generateBinaryStrings(n):
    """
    Generate all binary strings of length n.
    """
    return [''.join(seq) for seq in itertools.product('01', repeat=n)]

def QuboToIsing(Q):
    Q = np.array(Q)
    n = Q.shape[0]

    h = np.zeros(n)
    J = np.zeros((n, n))

    constant = np.sum(Q) / 4

    for i in range(n):
        h[i] = 1/2*(np.sum(Q[i, :]))
        for j in range(i+1, n):
            J[i, j] = Q[i, j] /2
    return h, J

def ConvertToUpperTriangular(J):
    J = np.array(J)

    n = J.shape[0]

    J_prime = np.zeros_like(J)

    for i in range(n):
        for j in range(i + 1, n):
            J_prime[i, j] = J[i, j] + J[j, i]

    return J_prime

def exhaustiveQUBOCheck(Q):
    keys = generateBinaryStrings(len(Q))
    optimal = None
    minEnergy = np.inf
    secondMinEnergy = np.inf
    for key in keys:
        vector = np.array([int(i) for i in key])
        energy = vector.T @ Q @ vector
        if energy < minEnergy:
            minEnergy = energy
            optimal = vector
    return optimal, minEnergy

def exhaustiveQUBOCheck_withSecond(Q):
    keys = generateBinaryStrings(len(Q))
    optimal = None
    minEnergy = np.inf
    secondMinEnergy = np.inf
    for key in keys:
        vector = np.array([int(i) for i in key])
        energy = vector.T @ Q @ vector
        if energy < minEnergy:
            minEnergy = energy
            optimal = vector
        if energy < secondMinEnergy and energy > minEnergy:
            secondMinEnergy = energy
    return optimal, minEnergy, secondMinEnergy

def exhaustiveIsingCheck(J, h):
    keys = generateBinaryStrings(len(h))
    optimal = None
    minEnergy = np.inf
    for key in keys:
        vector = np.array([(2*int(i) - 1) for i in key])
        energy = vector.T @ J @ vector + h.T @ vector
        if energy < minEnergy:
            minEnergy = energy
            optimal = vector
    return optimal, minEnergy

def solutionToGrid(solution, parameters):
    counter = 0
    solutionGrid = np.zeros((parameters['len_grid'], parameters['len_grid']))
    for i in range(parameters['len_grid']):
        for j in range(parameters['len_grid']):
            solutionGrid[j, i] = solution[counter]
            counter += 1
    return solutionGrid

def generateRandomQubo(n):
    Q = np.random.uniform(-10, 10, (n, n))
    Q = (Q + Q.T) / 2  # Make it symmetric
    return Q

def generateRandomIndices(n, k):
    """
    Generate k unique random indices from the range [0, n).
    """
    if k > n:
        raise ValueError("k must be less than or equal to n")
    return np.random.choice(n, size=k, replace=False).tolist()

def generateERGraph(n, p, seed=-1):
    """
    Generate an Erdős–Rényi random undirected graph adjacency matrix.
    n: number of nodes
    p: probability of edge creation
    seed: random seed (optional)
    Returns: n x n numpy array (adjacency matrix)
    """
    if seed < 0:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed)
    E = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.uniform(0.0, 1.0) < p:
                E[i, j] = 1
                E[j, i] = 1  # Ensure undirected
    return E

def maxcutQUBO(A):
    """
    Construct the QUBO matrix for the MaxCut problem from adjacency matrix A.
    A: n x n numpy array (adjacency matrix)
    Returns: n x n numpy array (QUBO matrix)
    """
    n = A.shape[0]
    Q = np.zeros((n, n))
    for u in range(n):
        for v in range(u + 1, n):
            if A[u, v] > 0:
                Q[u, u] += -1
                Q[v, v] += -1
                Q[u, v] += 2
                Q[v, u] += 2  # QUBO is symmetric
    return Q

def generateBinaryStringsWithCopies(n, k):
    '''
    Generates all binary strings of length n with exactly k 1s, 2s, and 3s
    
    Parameters:
    n (int): The length of the binary strings
    k (int): The number of 1s, 2s, and 3s in the binary strings
    
    Returns:
    list: A list of binary strings with 1s, 2s, and 3s
    '''
    result = []
    for positions in itertools.combinations(range(n), k):
        for digit in ['3', '2', '1']:
            binaryString = ['0'] * n
            for pos in positions:
                binaryString[pos] = digit
            result.append(''.join(binaryString))
    return result

def chooseNfixedK(len_grid, k):
    '''
    Finds the smallest n such that n choose k is greater than or equal to len_grid**2
    
    Parameters:
    len_grid (int): The number of grid points in one dimension
    k (int): The number of 1s, 2s, and 3s in the binary strings
    
    Returns:
    n (int): The smallest n such that n choose k is greater than or equal to len_grid**2
    '''
    N = len_grid ** 2
    n = k  # Start from k since n choose k is not defined for n < k

    while True:
        if 3*math.comb(n, k) >= N:
            return n
        n += 1

def chooseNandK(len_grid):
    '''
    Finds the smallest n and k such that n choose k is greater than or equal to len_grid**2
    
    Parameters:
    len_grid (int): The number of grid points in one dimension
    
    Returns:
    n (int): The smallest n such that n choose k is greater than or equal to len_grid**2
    k (int): The smallest k such that n choose k is greater than or equal to len_grid**2
    '''
    N = len_grid ** 2
    n = 1

    while True:
        for k in range(n + 1):
            if 3 * math.comb(n, k) >= N:
                return n, k
        n += 1

def Alltwalis_P(len_grid):
    P = []
    if len_grid == 8:
        P = [7, 8, 15, 16, 28, 36, 37, 43, 44, 45, 49, 50, 51, 52, 53, 57, 58, 59, 60, 61, 63, 64]
    elif len_grid == 9:
        P = [8, 9, 16, 17, 18, 27, 31, 40, 41, 49, 50, 55, 57, 58, 59, 64, 65, 66, 67, 68, 71, 72, 73, 74, 75, 76, 77, 79, 80, 81]
    elif len_grid == 10:
        P = [9, 10, 18, 19, 20, 29, 30, 34, 40, 44, 45, 46, 54, 55, 63, 64, 65, 71, 73, 74, 75, 81, 82, 83, 84, 85, 86, 89, 90, 91, 92, 93, 94, 95, 96, 98, 99, 100]
    return P

def wind_regime_details(windfarmParameters):
    if windfarmParameters['case'] == 'NorthSea':
        windfarmParameters['D'] = [[0, 9.77, 0.063],[30, 8.34, 0.059],[60, 7.93, 0.055],[90, 10.18, 0.078],[120, 8.14, 0.083],[150, 8.24, 0.065],[180, 9.05, 0.114],
        [210, 11.59, 0.146],[240, 12.11, 0.121],[270, 11.90, 0.085],[300, 10.38, 0.064],[330, 8.14, 0.067]]
        #windfarmParameters['D'] = [[0,12,1]]
        windfarmParameters['turbine_r'] = 82
        windfarmParameters['a'] = 0.094
        if windfarmParameters['subcase'] == 'A':
            windfarmParameters['field_length'] = 3940
            windfarmParameters['m'] = 16
            windfarmParameters['E'] = 0.0
            windfarmParameters['P'] = []
        elif windfarmParameters['subcase'] == 'B':
            windfarmParameters['field_length'] = 7872
            windfarmParameters['m'] = 49
            windfarmParameters['E'] = 0.0
            windfarmParameters['P'] = []
    elif windfarmParameters['case'] == 'Alltwalis':
        windfarmParameters['D'] = [[0, 4.65, 0.08], [30, 1.55, 0.03],[60, 1.55, 0.04],[90, 4.65, 0.07],[120, 3.10, 0.05],[150, 6.20, 0.08],[180, 7.97, 0.12],[210, 9.3, 0.14],
        [240, 7.97, 0.12],[270, 4.65, 0.08],[300, 6.20, 0.09],[330, 6.20, 0.09]]
        #windfarmParameters['D'] = [[0,12,1]]
        windfarmParameters['turbine_r'] = 46.5
        windfarmParameters['field_length'] = 1581.13
        windfarmParameters['m'] = 10
        windfarmParameters['a'] = 0.154
        windfarmParameters['E'] = 465
        windfarmParameters['P'] = Alltwalis_P(windfarmParameters['len_grid'])

    windfarmParameters['box_width'] = windfarmParameters['field_length'] / windfarmParameters['len_grid']
    return windfarmParameters