"""
Facility Location Optimization Solver

This module contains integer programming formulations for five classic facility location problems:
1. LSCP - Location Set Covering Problem
2. MCLP - Maximum Covering Location Problem
3. P-Median Problem
4. P-Center Problem
5. SPLP - Simple Plant Location Problem

All problems are solved using integer programming with PuLP for optimal solutions.
"""

import numpy as np
import pulp
from scipy.spatial.distance import cdist


def solve_lscp_ip(distances, coverage_radius, return_status=False):
    """
    Solve Location Set Covering Problem using integer programming.
    
    Objective: Minimize the number of facilities needed to cover all demand points.
    Constraint: Each demand point must be within coverage radius of at least one facility.
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Maximum distance for coverage
    return_status : bool, optional
        If True, return solver status along with solution
        
    Returns:
    --------
    list or tuple
        Indices of selected facilities. If return_status=True, returns (facilities, status, objective_value)
    """
    n_demand, n_facilities = distances.shape
    coverage_matrix = (distances <= coverage_radius).astype(int)
    
    # Create the problem
    prob = pulp.LpProblem("LSCP", pulp.LpMinimize)
    
    # Decision variables: x[j] = 1 if facility j is selected, 0 otherwise
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_facilities)]
    
    # Objective: minimize number of facilities
    prob += pulp.lpSum(x)
    
    # Constraints: each demand point must be covered by at least one facility
    for i in range(n_demand):
        prob += pulp.lpSum([coverage_matrix[i, j] * x[j] for j in range(n_facilities)]) >= 1
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_facilities = [j for j in range(n_facilities) if x[j].varValue == 1]
    
    if return_status:
        return selected_facilities, pulp.LpStatus[prob.status], prob.objective.value()
    return selected_facilities


def solve_mclp_ip(distances, coverage_radius, p_facilities, demand_weights, return_status=False):
    """
    Solve Maximum Covering Location Problem using integer programming.
    
    Objective: Maximize the total demand covered with a fixed number of facilities.
    Constraint: Limited number of facilities (budget constraint).
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Maximum distance for coverage
    p_facilities : int
        Number of facilities to locate
    demand_weights : array-like
        Demand weights for each demand point
    return_status : bool, optional
        If True, return solver status along with solution
        
    Returns:
    --------
    list or tuple
        Indices of selected facilities. If return_status=True, returns (facilities, status, objective_value)
    """
    n_demand, n_facilities = distances.shape
    coverage_matrix = (distances <= coverage_radius).astype(int)
    
    # Create the problem
    prob = pulp.LpProblem("MCLP", pulp.LpMaximize)
    
    # Decision variables
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_facilities)]  # facility j selected
    y = [pulp.LpVariable(f"y_{i}", cat='Binary') for i in range(n_demand)]      # demand i covered
    
    # Objective: maximize weighted coverage
    prob += pulp.lpSum([demand_weights[i] * y[i] for i in range(n_demand)])
    
    # Constraints
    # Exactly p facilities
    prob += pulp.lpSum(x) == p_facilities
    
    # Coverage constraints: demand i can only be covered if at least one facility covering it is selected
    for i in range(n_demand):
        prob += y[i] <= pulp.lpSum([coverage_matrix[i, j] * x[j] for j in range(n_facilities)])
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_facilities = [j for j in range(n_facilities) if x[j].varValue == 1]
    
    if return_status:
        return selected_facilities, pulp.LpStatus[prob.status], prob.objective.value()
    return selected_facilities


def solve_pmedian_ip(distances, p_facilities, demand_weights, return_status=False):
    """
    Solve P-Median Problem using integer programming.
    
    Objective: Minimize total weighted distance from demand points to nearest facilities.
    Constraint: Exactly p facilities must be located.
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    p_facilities : int
        Number of facilities to locate
    demand_weights : array-like
        Demand weights for each demand point
    return_status : bool, optional
        If True, return solver status along with solution
        
    Returns:
    --------
    list or tuple
        Indices of selected facilities. If return_status=True, returns (facilities, status, objective_value)
    """
    n_demand, n_facilities = distances.shape
    
    # Create the problem
    prob = pulp.LpProblem("P_Median", pulp.LpMinimize)
    
    # Decision variables
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_facilities)]  # facility j selected
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat='Binary') for j in range(n_facilities)] for i in range(n_demand)]  # demand i assigned to facility j
    
    # Objective: minimize total weighted distance
    prob += pulp.lpSum([demand_weights[i] * distances[i, j] * y[i][j] 
                       for i in range(n_demand) for j in range(n_facilities)])
    
    # Constraints
    # Exactly p facilities
    prob += pulp.lpSum(x) == p_facilities
    
    # Each demand point assigned to exactly one facility
    for i in range(n_demand):
        prob += pulp.lpSum(y[i]) == 1
    
    # Assignment constraints: can only assign to selected facilities
    for i in range(n_demand):
        for j in range(n_facilities):
            prob += y[i][j] <= x[j]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_facilities = [j for j in range(n_facilities) if x[j].varValue == 1]
    
    if return_status:
        return selected_facilities, pulp.LpStatus[prob.status], prob.objective.value()
    return selected_facilities


def solve_pcenter_ip(distances, p_facilities, return_status=False):
    """
    Solve P-Center Problem using integer programming.
    
    Objective: Minimize the maximum distance from any demand point to its nearest facility.
    Constraint: Exactly p facilities must be located.
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    p_facilities : int
        Number of facilities to locate
    return_status : bool, optional
        If True, return solver status along with solution
        
    Returns:
    --------
    list or tuple
        Indices of selected facilities. If return_status=True, returns (facilities, status, objective_value)
    """
    n_demand, n_facilities = distances.shape
    
    # Create the problem
    prob = pulp.LpProblem("P_Center", pulp.LpMinimize)
    
    # Decision variables
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_facilities)]  # facility j selected
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat='Binary') for j in range(n_facilities)] for i in range(n_demand)]  # demand i assigned to facility j
    z = pulp.LpVariable("z", lowBound=0)  # maximum distance
    
    # Objective: minimize maximum distance
    prob += z
    
    # Constraints
    # Exactly p facilities
    prob += pulp.lpSum(x) == p_facilities
    
    # Each demand point assigned to exactly one facility
    for i in range(n_demand):
        prob += pulp.lpSum(y[i]) == 1
    
    # Assignment constraints: can only assign to selected facilities
    for i in range(n_demand):
        for j in range(n_facilities):
            prob += y[i][j] <= x[j]
    
    # Maximum distance constraints
    for i in range(n_demand):
        for j in range(n_facilities):
            prob += z >= distances[i, j] * y[i][j]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_facilities = [j for j in range(n_facilities) if x[j].varValue == 1]
    
    if return_status:
        return selected_facilities, pulp.LpStatus[prob.status], prob.objective.value()
    return selected_facilities


def solve_splp_ip(distances, facility_costs, demand_weights, return_status=False):
    """
    Solve Simple Plant Location Problem using integer programming.
    
    Objective: Minimize total cost (facility opening costs + transportation costs).
    Constraint: No limit on number of facilities, but each has an opening cost.
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    facility_costs : array-like
        Opening cost for each potential facility
    demand_weights : array-like
        Demand weights for each demand point
    return_status : bool, optional
        If True, return solver status along with solution
        
    Returns:
    --------
    list or tuple
        Indices of selected facilities. If return_status=True, returns (facilities, status, objective_value)
    """
    n_demand, n_facilities = distances.shape
    
    # Create the problem
    prob = pulp.LpProblem("SPLP", pulp.LpMinimize)
    
    # Decision variables
    x = [pulp.LpVariable(f"x_{j}", cat='Binary') for j in range(n_facilities)]  # facility j selected
    y = [[pulp.LpVariable(f"y_{i}_{j}", cat='Binary') for j in range(n_facilities)] for i in range(n_demand)]  # demand i assigned to facility j
    
    # Objective: minimize total cost (facility opening + transportation)
    prob += (pulp.lpSum([facility_costs[j] * x[j] for j in range(n_facilities)]) + 
             pulp.lpSum([demand_weights[i] * distances[i, j] * y[i][j] 
                        for i in range(n_demand) for j in range(n_facilities)]))
    
    # Constraints
    # Each demand point assigned to exactly one facility
    for i in range(n_demand):
        prob += pulp.lpSum(y[i]) == 1
    
    # Assignment constraints: can only assign to selected facilities
    for i in range(n_demand):
        for j in range(n_facilities):
            prob += y[i][j] <= x[j]
    
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    
    # Extract solution
    selected_facilities = [j for j in range(n_facilities) if x[j].varValue == 1]
    
    if return_status:
        return selected_facilities, pulp.LpStatus[prob.status], prob.objective.value()
    return selected_facilities


def calculate_coverage(selected_facilities, distances, coverage_radius):
    """
    Calculate coverage percentage for covering problems.
    
    Parameters:
    -----------
    selected_facilities : list
        Indices of selected facilities
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Maximum distance for coverage
        
    Returns:
    --------
    float
        Coverage percentage (0-100)
    """
    n_demand = distances.shape[0]
    coverage_matrix = (distances <= coverage_radius).astype(int)
    covered = 0
    for i in range(n_demand):
        if any(coverage_matrix[i, j] == 1 for j in selected_facilities):
            covered += 1
    return covered / n_demand * 100


def calculate_weighted_coverage(selected_facilities, distances, coverage_radius, demand_weights):
    """
    Calculate total weighted coverage for MCLP (sum of demand weights for covered demand points).
    
    This matches the IP objective: maximize sum(demand_weights[i] * y[i]) where y[i]=1 if demand point i is covered.
    
    Parameters:
    -----------
    selected_facilities : list
        Indices of selected facilities
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Maximum distance for coverage
    demand_weights : array-like
        Demand weights for each demand point
        
    Returns:
    --------
    float
        Total weighted demand covered (sum of weights for covered demand points)
    """
    import numpy as np
    n_demand = distances.shape[0]
    coverage_matrix = (distances <= coverage_radius).astype(int)
    demand_weights = np.asarray(demand_weights)  # Ensure it's a numpy array for proper indexing
    total_weighted_coverage = 0.0
    for i in range(n_demand):
        if any(coverage_matrix[i, j] == 1 for j in selected_facilities):
            total_weighted_coverage += float(demand_weights[i])
    return total_weighted_coverage


def calculate_total_distance(selected_facilities, distances, demand_weights):
    """
    Calculate total weighted distance for distance-based problems.
    
    Parameters:
    -----------
    selected_facilities : list
        Indices of selected facilities
    distances : array-like
        Distance matrix between demand points and potential facilities
    demand_weights : array-like
        Demand weights for each demand point
        
    Returns:
    --------
    float
        Total weighted distance
    """
    if not selected_facilities:
        # If no facilities are selected, return infinity (infeasible solution)
        return float('inf')
    
    total_dist = 0
    for i in range(len(demand_weights)):
        min_dist = min(distances[i, j] for j in selected_facilities)
        total_dist += demand_weights[i] * min_dist
    return total_dist


def calculate_max_distance(selected_facilities, distances):
    """
    Calculate maximum distance for P-Center problem.
    
    Parameters:
    -----------
    selected_facilities : list
        Indices of selected facilities
    distances : array-like
        Distance matrix between demand points and potential facilities
        
    Returns:
    --------
    float
        Maximum distance from any demand point to nearest facility
    """
    if not selected_facilities:
        # If no facilities are selected, return infinity (infeasible solution)
        return float('inf')
    
    return max(min(distances[i, j] for j in selected_facilities) for i in range(distances.shape[0]))


def calculate_total_cost(selected_facilities, facility_costs, distances, demand_weights):
    """
    Calculate total cost for SPLP (facility opening + transportation).
    
    Parameters:
    -----------
    selected_facilities : list
        Indices of selected facilities
    facility_costs : array-like
        Opening cost for each potential facility
    distances : array-like
        Distance matrix between demand points and potential facilities
    demand_weights : array-like
        Demand weights for each demand point
        
    Returns:
    --------
    float
        Total cost (facility opening + transportation)
    """
    facility_cost = sum(facility_costs[j] for j in selected_facilities)
    transport_cost = calculate_total_distance(selected_facilities, distances, demand_weights)
    return facility_cost + transport_cost

# Enumeration solution functions
def solve_lscp_enumeration(distances, coverage_radius, progress_callback=None):
    """
    Solve LSCP using complete enumeration
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Coverage radius for facilities
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (best_solution, best_objective, total_combinations_tested)
    """
    n_facilities = distances.shape[1]
    n_demand = distances.shape[0]
    
    # Generate all possible combinations
    from itertools import combinations
    
    best_solution = None
    best_objective = float('inf')
    total_combinations = 0
    
    # Count total combinations for progress tracking
    for k in range(1, n_facilities + 1):
        total_combinations += len(list(combinations(range(n_facilities), k)))
    
    tested_combinations = 0
    
    # Test all combinations
    for k in range(1, n_facilities + 1):
        for solution in combinations(range(n_facilities), k):
            tested_combinations += 1
            
            # Check if this solution covers all demand
            coverage = calculate_coverage(list(solution), distances, coverage_radius)
            if coverage == 100.0:
                objective = len(solution)
                if objective < best_objective:
                    best_objective = objective
                    best_solution = list(solution)
            
            # Update progress
            if progress_callback:
                progress_callback(tested_combinations / total_combinations)
    
    return best_solution, best_objective, total_combinations

def solve_mclp_enumeration(distances, coverage_radius, p_facilities, demand_weights, progress_callback=None):
    """
    Solve MCLP using complete enumeration
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    coverage_radius : float
        Coverage radius for facilities
    p_facilities : int
        Number of facilities to locate
    demand_weights : array-like
        Demand weights for each demand point
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (best_solution, best_objective, total_combinations_tested)
    """
    n_facilities = distances.shape[1]
    
    # Generate all combinations of exactly p facilities
    from itertools import combinations
    
    combinations_list = list(combinations(range(n_facilities), p_facilities))
    total_combinations = len(combinations_list)
    
    best_solution = None
    best_objective = 0
    
    # Test all combinations
    for i, solution in enumerate(combinations_list):
        solution_list = list(solution)
        # Use the same calculation function as elsewhere to ensure consistency
        weighted_coverage = calculate_weighted_coverage(solution_list, distances, coverage_radius, demand_weights)
        
        if weighted_coverage > best_objective:
            best_objective = weighted_coverage
            best_solution = solution_list
        
        # Update progress
        if progress_callback:
            progress_callback((i + 1) / total_combinations)
    
    return best_solution, best_objective, total_combinations

def solve_pmedian_enumeration(distances, p_facilities, demand_weights, progress_callback=None):
    """
    Solve P-Median using complete enumeration
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    p_facilities : int
        Number of facilities to locate
    demand_weights : array-like
        Demand weights for each demand point
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (best_solution, best_objective, total_combinations_tested)
    """
    n_facilities = distances.shape[1]
    
    # Generate all combinations of exactly p facilities
    from itertools import combinations
    
    combinations_list = list(combinations(range(n_facilities), p_facilities))
    total_combinations = len(combinations_list)
    
    best_solution = None
    best_objective = float('inf')
    
    # Test all combinations
    for i, solution in enumerate(combinations_list):
        solution_list = list(solution)
        total_distance = calculate_total_distance(solution_list, distances, demand_weights)
        
        if total_distance < best_objective:
            best_objective = total_distance
            best_solution = solution_list
        
        # Update progress
        if progress_callback:
            progress_callback((i + 1) / total_combinations)
    
    return best_solution, best_objective, total_combinations

def solve_pcenter_enumeration(distances, p_facilities, progress_callback=None):
    """
    Solve P-Center using complete enumeration
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    p_facilities : int
        Number of facilities to locate
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (best_solution, best_objective, total_combinations_tested)
    """
    n_facilities = distances.shape[1]
    
    # Generate all combinations of exactly p facilities
    from itertools import combinations
    
    combinations_list = list(combinations(range(n_facilities), p_facilities))
    total_combinations = len(combinations_list)
    
    best_solution = None
    best_objective = float('inf')
    
    # Test all combinations
    for i, solution in enumerate(combinations_list):
        solution_list = list(solution)
        max_distance = calculate_max_distance(solution_list, distances)
        
        if max_distance < best_objective:
            best_objective = max_distance
            best_solution = solution_list
        
        # Update progress
        if progress_callback:
            progress_callback((i + 1) / total_combinations)
    
    return best_solution, best_objective, total_combinations

def solve_splp_enumeration(distances, facility_costs, demand_weights, progress_callback=None):
    """
    Solve SPLP using complete enumeration
    
    Parameters:
    -----------
    distances : array-like
        Distance matrix between demand points and potential facilities
    facility_costs : array-like
        Opening cost for each potential facility
    demand_weights : array-like
        Demand weights for each demand point
    progress_callback : callable, optional
        Function to call with progress updates (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (best_solution, best_objective, total_combinations_tested)
    """
    n_facilities = distances.shape[1]
    
    # Generate all possible combinations (including empty set)
    from itertools import combinations
    
    best_solution = None
    best_objective = float('inf')
    total_combinations = 0
    
    # Count total combinations for progress tracking
    for k in range(n_facilities + 1):
        total_combinations += len(list(combinations(range(n_facilities), k)))
    
    tested_combinations = 0
    
    # Test all combinations
    for k in range(n_facilities + 1):
        for solution in combinations(range(n_facilities), k):
            tested_combinations += 1
            
            solution_list = list(solution)
            total_cost = calculate_total_cost(solution_list, facility_costs, distances, demand_weights)
            
            if total_cost < best_objective:
                best_objective = total_cost
                best_solution = solution_list
            
            # Update progress
            if progress_callback:
                progress_callback(tested_combinations / total_combinations)
    
    return best_solution, best_objective, total_combinations
