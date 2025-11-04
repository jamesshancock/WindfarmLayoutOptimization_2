from includes import *
from modules import *

windfarmParameters = {
    'len_grid': 7,
    'case': "NorthSea",  # NorthSea or Alltwalis
    'subcase': "A",  # Subcase for testingg (NorthSea only)
    'lam': 200,
    'exhaustCheck': False,  # Set to True to run exhaustive check'
    'solver': "sqoe",  # Options: 'sqoe', 'pce', 'gurobi'
    'warmStart' : True,
}

PCEParameters = {
    'fixedK': [True, 1],
    'L': 10,
    'tol': 1e-1,
    'talpha': 10,
    'miniter': 10,
    'maxiter': 1000,
    'shots': 3000,
}

SQOEParameters = {
    'stepSize': 0.1,
    'learningRate': 0.0001,
    't': 10,
    'allQubits': True,
    'totalQubits': 20,
    'maxiter': 1000,
    'shots': 1000,
}

windfarmParameters['lam1'] = windfarmParameters['lam']
windfarmParameters['lam2'] = windfarmParameters['lam']
windfarmParameters['lam3'] = windfarmParameters['lam']

windfarmParameters = wind_regime_details(windfarmParameters)

parameters = {**windfarmParameters, **PCEParameters, **SQOEParameters}

# --- QUBO/Ising setup and optimal solution ---
Q = WindfarmQ(parameters)
h, J = QuboToIsing(Q)

if __name__ == "__main__":
    solution, grad_history, timeTaken, timeTakenPerIter, xs, endReason, sim_times, costs = SQOE_WFLOsolver(parameters)
    print("solutions=",solution)
    print("grad_historys=",grad_history)
    print("timeTakens=",timeTaken)
    print("timeTakenPerIters=",timeTakenPerIter)
    print("xss=",xs)
    print("endReasons=",endReason)
    print("sim_timess=",sim_times)
    print("costss=",costs)