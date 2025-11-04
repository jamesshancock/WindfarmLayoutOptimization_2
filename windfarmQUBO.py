from includes import *
from helpers import *

def labelling(len_grid):
    offset = (len_grid - 1) / 2  # Calculate the offset to center the grid around 0
    pos_labels = {c: [k0 - offset, k1 - offset] for c, (k0, k1) in enumerate(((k0, k1) for k0 in range(len_grid) for k1 in range(len_grid)), 1)}
    return pos_labels

def cone_maker(turbine_i, len_grid, box_width, alpha_deg, turbine_radius, wake_half_angle_deg=15):
    """
    Returns a dict marking which grid cells are in the wake cone of the turbine at index turbine_i.
    - turbine_i: 1-based index of the turbine in the grid (row-major order)
    - len_grid: number of rows/cols in the grid (assumed square)
    - box_width: physical width of each grid cell (meters)
    - alpha_deg: wind direction in degrees (0 = east, 90 = north)
    - turbine_radius: radius of the turbine (meters)
    - wake_half_angle_deg: half-angle of the wake cone (degrees)
    """
    # Convert to radians
    alpha = np.deg2rad(alpha_deg)
    wake_half_angle = np.deg2rad(wake_half_angle_deg)

    # Convert 1-based index to (i, j)
    idx = turbine_i - 1
    i0, j0 = divmod(idx, len_grid)

    cone = {}
    for i in range(len_grid):
        for j in range(len_grid):
            cell_idx = i * len_grid + j + 1  # 1-based
            dx = (j - j0) * box_width
            dy = (i - i0) * box_width
            x = np.sqrt(dx**2 + dy**2)
            if x == 0:
                cone[cell_idx] = 1  # Always include the origin
                continue
            theta = np.arctan2(dy, dx)
            dtheta = np.arctan2(np.sin(theta - alpha), np.cos(theta - alpha))
            # Only include points downwind (in the direction of alpha)
            if abs(dtheta) < wake_half_angle and (dx * np.cos(alpha) + dy * np.sin(alpha)) > 0:
                cone[cell_idx] = 1
            else:
                cone[cell_idx] = 0
    return cone

def cone_maker_with_radius(
    turbine_i, len_grid, box_width, alpha_deg, turbine_radius, a, wake_half_angle_deg=15
):
    """
    Returns a dict mapping each grid cell to its wake radius (meters) if in the wake cone, else 0.
    """
    alpha = np.deg2rad(alpha_deg)
    wake_half_angle = np.deg2rad(wake_half_angle_deg)
    idx = turbine_i - 1
    i0, j0 = divmod(idx, len_grid)

    cone = {}
    for i in range(len_grid):
        for j in range(len_grid):
            cell_idx = i * len_grid + j + 1  # 1-based
            dx = (j - j0) * box_width
            dy = (i - i0) * box_width
            x = np.sqrt(dx**2 + dy**2)
            if x == 0:
                cone[cell_idx] = turbine_radius  # At the origin, wake radius is just the turbine radius
                continue
            theta = np.arctan2(dy, dx)
            dtheta = np.arctan2(np.sin(theta - alpha), np.cos(theta - alpha))
            # Only include points downwind (in the direction of alpha)
            if abs(dtheta) < wake_half_angle and (dx * np.cos(alpha) + dy * np.sin(alpha)) > 0:
                wake_radius = turbine_radius + a * x
                cone[cell_idx] = wake_radius
            else:
                cone[cell_idx] = 0
    return cone


def cone_matrix(cone, len_grid):
    """
    Converts the cone dict to a 2D matrix.
    """
    M = np.zeros((len_grid, len_grid), dtype=int)
    for idx, val in cone.items():
        i, j = divmod(idx - 1, len_grid)
        M[i, j] = val
    return M

def C_TCurve(windspeed, case):
    if case == 'NorthSea':
        windspeed_data = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])

        C_T = np.array([
            0.7000000000,
            0.722386304,
            0.773588333,
            0.773285946,
            0.767899317,
            0.732727569,
            0.688896343,
            0.623028669,
            0.500046699,
            0.373661747,
            0.293230676,
            0.238407400,
            0.196441644,
            0.163774674,
            0.137967245,
            0.117309371,
            0.100578122,
            0.086883163,
            0.075565832,
            0.066131748,
            0.058204932,
            0.051495998
        ])
    elif case == 'Alltwalis':
        windspeed_data = np.array([2.5,3.75,5.0,6.25,7.5,8.75,10.0,11.25,12.5,13.75,15.0,16.25,17.5,18.75,20.0])
        C_T = np.array([0.85,0.85,0.82,0.82,0.82,0.82,0.8,0.62,0.4,0.3,0.2,0.15,0.1,0.08,0.05])
    return float(np.interp(windspeed, windspeed_data, C_T))

def reduced_windspeed(turbine_i, turbine_j, d, parameters):
    box_width = parameters['box_width']
    turbine_radius = parameters['turbine_r']
    a = parameters['a']
    cone = cone_maker_with_radius(turbine_i, parameters['len_grid'], box_width,
                                  d[0], turbine_radius, a)
    labels = labelling(parameters['len_grid'])
    dist = np.sqrt(
        (labels[turbine_i][0] - labels[turbine_j][0])**2 +
        (labels[turbine_i][1] - labels[turbine_j][1])**2) * box_width
    wake_radius = turbine_radius + a * dist
    C_T = C_TCurve(d[1], parameters['case'])  # Thrust coefficient based on wind speed
    if cone[turbine_j] == 0:
        reduced_speed = d[1]  # No wake effect
    else:
        reduced_speed = (1-np.sqrt(1-C_T) * (turbine_radius / (wake_radius))**2) * d[1]
    return reduced_speed

def number_constraint(m,len_grid):
    C = np.zeros((len_grid**2,len_grid**2))
    for i in range(len_grid**2):
        for j in range(len_grid**2):
            if i == j:
                C[i,i] = 1 - 4*m
            else:
                C[i,j] = 2
    return C

def proximity_constraint(E, lenGrid, box_width):
    C = np.zeros((lenGrid ** 2, lenGrid ** 2))
    labels = labelling(lenGrid)
    for i in range(lenGrid ** 2):
        for j in range(lenGrid ** 2):
            if i != j:
                x1, x2 = labels[i + 1], labels[j + 1]
                # Calculate Euclidean distance in grid units, then convert to meters
                distance = math.sqrt((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) * box_width
                if distance <= E:
                    C[i, j] = 1
    return C

def LocationConstraint(noGos, lenGrid):
    C = np.zeros((lenGrid ** 2, lenGrid ** 2))
    for i in range(lenGrid ** 2):
        if i + 1 in noGos:
            C[i, i] = 1
    return C

def WindfarmQ(parameters):
    D = parameters['D']
    m = parameters['m']
    E = parameters['E']
    lam1 = parameters['lam1']
    lam2 = parameters['lam2']
    lam3 = parameters['lam3']
    len_grid = parameters['len_grid']
    Q = np.zeros((len_grid**2, len_grid**2))
    for d in D:
        prob = d[2]
        free_speed = d[1]
        # Diagonal: free wind speed cubed
        for i in range(len_grid**2):
            Q[i, i] += (1/3) * prob * (free_speed ** 3)
        # Off-diagonal: wake deficit
        for i in range(len_grid**2):
            for j in range(len_grid**2):
                if i != j:
                    u_ij = reduced_windspeed(i+1, j+1, d, parameters)
                    deficit = free_speed ** 3 - u_ij ** 3
                    Q[i, j] += -(1/3) * prob * deficit
    Q = 0.5 * (Q + Q.T)
    Q = Q - lam1 * number_constraint(m, len_grid) \
          - lam2 * proximity_constraint(E, len_grid, parameters['box_width']) \
          - lam3 * LocationConstraint(parameters['P'], len_grid)
    return -Q

def Energy(parameters,solution):
    D = parameters['D']
    m = parameters['m']
    E = parameters['E']
    m = np.count_nonzero(solution)
    energy = 0
    global len_grid
    len_grid = parameters['len_grid']
    for d in D:
        angle = d[0]
        free_speed = d[1]
        prob = d[2]
        for k in range(1,len(solution)+1):
            pos = labelling(len_grid)
            if solution[k-1] == 1:
                cone = cone_maker_with_radius(k, len_grid,
                                              parameters['box_width'], angle,
                                              parameters['turbine_r'],
                                              parameters['a'])
                term = 0
                for c in cone:
                    if cone[c] == 1 and solution[c-1] == 1:
                        if c != k:
                            reduced_speed = reduced_windspeed(k,c,d,parameters)
                            term += free_speed**3 - reduced_speed**3
                energy += 1/3*prob*(free_speed**3 - term)
    return energy

def Energy_corrected(parameters, solution):
    D = parameters['D']
    len_grid = parameters['len_grid']
    energy = 0
    for d in D:
        free_speed = d[1]
        prob = d[2]
        for j in range(1, len(solution) + 1):
            if solution[j - 1] == 1:
                # Linear superposition: sum all wake deficits at j from all other turbines i
                wake_deficit = 0
                for i in range(1, len(solution) + 1):
                    if i != j and solution[i - 1] == 1:
                        u_ij = reduced_windspeed(i, j, d, parameters)
                        wake_deficit += free_speed**3 - u_ij**3
                energy += (1 / 3) * prob * (free_speed**3 - wake_deficit)
    return energy

def solutionToGrid(solution, parameters):
    counter = 0
    solutionGrid = np.zeros((parameters['len_grid'], parameters['len_grid']))
    for i in range(parameters['len_grid']):
        for j in range(parameters['len_grid']):
            solutionGrid[j, i] = solution[counter]
            counter += 1
    return solutionGrid

def plot_wake_superposition_heatmap(solution_grid, parameters):
    len_grid = parameters['len_grid']
    D = parameters['D']
    # Flatten the solution grid to match your 1-based indexing
    solution = solution_grid.flatten(order='F')  # Fortran order to match your column-major logic

    # Initialize the heatmap with the free windspeed for each direction
    heatmap = np.zeros((len_grid, len_grid))
    total_prob = 0

    for d in D:
        free_speed = d[1]
        prob = d[2]
        wind_speeds = np.full((len_grid, len_grid), free_speed, dtype=float)

        # For each cell, compute the wind speed after all wakes
        for i in range(len_grid):
            for j in range(len_grid):
                cell_idx = i * len_grid + j + 1  # 1-based
                # Superpose all wakes affecting this cell
                for t_idx, is_turbine in enumerate(solution, 1):  # 1-based
                    if is_turbine and cell_idx != t_idx:
                        ws = reduced_windspeed(t_idx, cell_idx, d, parameters)
                        # Linear superposition: subtract the wake deficit
                        wind_speeds[i, j] -= (free_speed - ws)
                # Ensure wind speed doesn't go below zero
                wind_speeds[i, j] = max(wind_speeds[i, j], 0)

        # Weighted sum by probability
        heatmap += prob * wind_speeds
        total_prob += prob

    # Normalize by total probability (should sum to 1, but just in case)
    if total_prob > 0:
        heatmap /= total_prob

    # Plot the heatmap
    plt.figure(figsize=(8, 7), dpi=100)
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.imshow(heatmap, cmap='Blues', origin='lower', extent=[0, len_grid, 0, len_grid])
    plt.colorbar(label='Wind Speed (m/s)')
    plt.xlabel('Grid X')
    plt.ylabel('Grid Y')

    # Overlay red dots for turbine locations
    turbine_positions = np.argwhere(solution_grid == 1)
    if turbine_positions.size > 0:
        # Note: imshow origin='lower', so y is row, x is col
        plt.scatter(turbine_positions[:, 1] + 0.5, turbine_positions[:, 0] + 0.5,
                    c='red', s=100, marker='o', edgecolors='black', linewidths=1.5, label='Turbine')
        plt.legend(loc='upper right')

    plt.show()