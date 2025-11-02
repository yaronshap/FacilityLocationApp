"""
Streamlit App for Facility Location Optimization Problems

This interactive app allows users to:
1. Set problem parameters (demand points, facilities, seed)
2. Generate random spatial data
3. Solve all 5 facility location problems optimally
4. Visualize results with interactive plots

Author: Generated for OMG411 course
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from scipy.spatial.distance import cdist
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')

# Import our facility location solver
from facility_location_solver import (
    solve_lscp_ip, solve_mclp_ip, solve_pmedian_ip, 
    solve_pcenter_ip, solve_splp_ip,
    solve_lscp_enumeration, solve_mclp_enumeration, solve_pmedian_enumeration, 
    solve_pcenter_enumeration, solve_splp_enumeration,
    calculate_coverage, calculate_total_distance, 
    calculate_max_distance, calculate_total_cost
)

# Import calculate_weighted_coverage with fallback
try:
    from facility_location_solver import calculate_weighted_coverage
except ImportError:
    # Fallback implementation if import fails
    def calculate_weighted_coverage(selected_facilities, distances, coverage_radius, demand_weights):
        """Calculate total weighted coverage for MCLP."""
        import numpy as np
        n_demand = distances.shape[0]
        coverage_matrix = (distances <= coverage_radius).astype(int)
        demand_weights = np.asarray(demand_weights)
        total_weighted_coverage = 0.0
        for i in range(n_demand):
            if any(coverage_matrix[i, j] == 1 for j in selected_facilities):
                total_weighted_coverage += float(demand_weights[i])
        return total_weighted_coverage

# Set page config
st.set_page_config(
    page_title="Facility Location Optimization",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .problem-description {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def generate_spatial_data(n_demand, n_facilities, seed, x_min=0, x_max=10, y_min=0, y_max=8):
    """Generate spatial data for facility location problems"""
    np.random.seed(seed)
    
    # Generate demand points (customers)
    demand_points = np.random.uniform([x_min+1, y_min+1], [x_max-1, y_max-1], (n_demand, 2))
    demand_weights = np.random.randint(10, 100, n_demand)  # Population/demand at each point
    
    # Generate potential facility locations
    facility_points = np.random.uniform([x_min+0.5, y_min+0.5], [x_max-0.5, y_max-0.5], (n_facilities, 2))
    facility_costs = np.random.randint(50, 200, n_facilities)  # Cost to open each facility
    
    # Sort demand points by x-coordinate (then by y-coordinate for ties)
    demand_sort_idx = np.lexsort((demand_points[:, 1], demand_points[:, 0]))
    demand_points = demand_points[demand_sort_idx]
    demand_weights = demand_weights[demand_sort_idx]
    
    # Sort facility points by x-coordinate (then by y-coordinate for ties)
    facility_sort_idx = np.lexsort((facility_points[:, 1], facility_points[:, 0]))
    facility_points = facility_points[facility_sort_idx]
    facility_costs = facility_costs[facility_sort_idx]
    
    # Calculate distance matrix (after sorting)
    distances = cdist(demand_points, facility_points)
    
    return demand_points, demand_weights, facility_points, facility_costs, distances

def create_base_data_visualization(demand_points, demand_weights, facility_points, facility_costs, coverage_radius):
    """Create visualization for base data only"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))  # Larger figure for better Excel export
    
    # Plot demand points (smaller markers relative to figure size)
    ax.scatter(demand_points[:, 0], demand_points[:, 1], c='lightblue', s=demand_weights*1.5, 
               alpha=0.7, edgecolors='black', linewidth=0.5, label='Demand Points (size=weight)')
    
    # Plot potential facilities (smaller markers relative to figure size)
    ax.scatter(facility_points[:, 0], facility_points[:, 1], c='gray', s=facility_costs/3, 
               alpha=0.8, marker='s', edgecolors='black', linewidth=0.5, label='Potential Facilities (size=cost)')
    
    # Add facility numbers (smaller font)
    for i, (x, y) in enumerate(facility_points):
        ax.annotate(f'F{i}', (x, y), xytext=(3, 3), textcoords='offset points', 
                   fontsize=9, fontweight='bold', color='black')
    
    # Add demand point numbers (smaller font)
    for i, (x, y) in enumerate(demand_points):
        ax.annotate(f'D{i}', (x, y), xytext=(3, 3), textcoords='offset points', 
                   fontsize=9, fontweight='bold', color='darkblue')
    
    # Add coverage radius circles around all potential facilities
    if coverage_radius:
        for j in range(len(facility_points)):
            circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                           fill=False, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
    
    # Create abbreviated title with key parameters
    n_demand = len(demand_points)
    n_facilities = len(facility_points)
    title = f'D:{n_demand} | F:{n_facilities} | R:{coverage_radius}'
    
    ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def create_manual_solution_visualization(demand_points, demand_weights, facility_points, facility_costs, 
                                       coverage_radius, manual_solution, highlighted_location=None):
    """Create interactive visualization for manual facility selection"""
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Plot demand points
    demand_colors = ['lightblue'] * len(demand_points)
    demand_sizes = demand_weights * 2
    demand_edges = ['black'] * len(demand_points)
    demand_linewidths = [0.5] * len(demand_points)
    
    # Highlight demand point if selected
    if highlighted_location and highlighted_location.startswith('D'):
        highlight_idx = int(highlighted_location[1:])
        if 0 <= highlight_idx < len(demand_points):
            demand_colors[highlight_idx] = 'yellow'
            demand_edges[highlight_idx] = 'red'
            demand_linewidths[highlight_idx] = 3
    
    ax.scatter(demand_points[:, 0], demand_points[:, 1], c=demand_colors, s=demand_sizes, 
               alpha=0.7, edgecolors=demand_edges, linewidths=demand_linewidths, label='Demand Points (size=weight)')
    
    # Plot unselected potential facilities
    unselected_facilities = [i for i in range(len(facility_points)) if i not in manual_solution]
    if unselected_facilities:
        facility_colors = ['gray'] * len(unselected_facilities)
        facility_edges = ['black'] * len(unselected_facilities)
        facility_linewidths = [0.5] * len(unselected_facilities)
        
        # Highlight facility if selected
        if highlighted_location and highlighted_location.startswith('F'):
            highlight_idx = int(highlighted_location[1:])
            if highlight_idx in unselected_facilities:
                idx_in_unselected = unselected_facilities.index(highlight_idx)
                facility_colors[idx_in_unselected] = 'yellow'
                facility_edges[idx_in_unselected] = 'red'
                facility_linewidths[idx_in_unselected] = 3
        
        ax.scatter(facility_points[unselected_facilities, 0], facility_points[unselected_facilities, 1], 
                  c=facility_colors, s=facility_costs[unselected_facilities]/2, alpha=0.5, marker='s', 
                  edgecolors=facility_edges, linewidths=facility_linewidths, label='Unselected Facilities')
        
        # Add dashed gray coverage circles for unselected facilities
        for j in unselected_facilities:
            circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                           fill=False, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
    
    # Plot selected facilities
    if manual_solution:
        selected_facilities = list(manual_solution)
        facility_colors = ['red'] * len(selected_facilities)
        facility_edges = ['black'] * len(selected_facilities)
        facility_linewidths = [2] * len(selected_facilities)
        
        # Highlight selected facility if selected
        if highlighted_location and highlighted_location.startswith('F'):
            highlight_idx = int(highlighted_location[1:])
            if highlight_idx in selected_facilities:
                idx_in_selected = selected_facilities.index(highlight_idx)
                facility_colors[idx_in_selected] = 'orange'
                facility_edges[idx_in_selected] = 'red'
                facility_linewidths[idx_in_selected] = 4
        
        ax.scatter(facility_points[selected_facilities, 0], facility_points[selected_facilities, 1], 
                  c=facility_colors, s=facility_costs[selected_facilities]/2, alpha=0.8, marker='s', 
                  edgecolors=facility_edges, linewidths=facility_linewidths, label='Selected Facilities')
        
        # Add coverage circles for selected facilities
        for j in manual_solution:
            circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                           fill=False, color='red', linestyle='--', alpha=0.5, linewidth=2)
            ax.add_patch(circle)
    
    # Add facility numbers (smaller font)
    for i, (x, y) in enumerate(facility_points):
        ax.annotate(f'F{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=6, fontweight='bold', color='black')
    
    # Add demand point numbers
    for i, (x, y) in enumerate(demand_points):
        ax.annotate(f'D{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=6, fontweight='bold', color='darkblue')
    
    # Create title with manual solution info
    n_demand = len(demand_points)
    n_facilities = len(facility_points)
    n_selected = len(manual_solution)
    title = f'Manual Solution: D:{n_demand} | F:{n_facilities} | S:{n_selected} | R:{coverage_radius}'
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def create_problem_visualization(demand_points, demand_weights, facility_points, facility_costs, 
                                solutions, titles, colors, coverage_radius=None):
    """Create visualization for all facility location problems"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Facility Location Optimization Problems', fontsize=16, fontweight='bold')
    
    # Remove the empty subplot (bottom right)
    fig.delaxes(axes[1, 2])
    
    # Panel 1: Base Data
    ax = axes[0, 0]
    ax.scatter(demand_points[:, 0], demand_points[:, 1], c='lightblue', s=demand_weights*2, 
               alpha=0.7, edgecolors='black', linewidth=0.5, label='Demand Points (size=weight)')
    ax.scatter(facility_points[:, 0], facility_points[:, 1], c='gray', s=facility_costs/2, 
               alpha=0.8, marker='s', edgecolors='black', linewidth=0.5, label='Potential Facilities (size=cost)')
    
    # Add coverage radius circles around all potential facilities
    if coverage_radius:
        for j in range(len(facility_points)):
            circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                           fill=False, color='gray', linestyle='--', alpha=0.3, linewidth=1)
            ax.add_patch(circle)
    
    # Add facility and demand labels for base data panel
    for i, (x, y) in enumerate(facility_points):
        ax.annotate(f'F{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=6, fontweight='bold', color='black')
    for i, (x, y) in enumerate(demand_points):
        ax.annotate(f'D{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=6, fontweight='bold', color='darkblue')
    
    ax.set_title(f'Base Data: Demand & Potential Facilities\n(Coverage Radius = {coverage_radius})', fontweight='bold', fontsize=10)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    # Panels 2-6: Optimization Problems
    panel_configs = [
        (0, 1, 0, 'LSCP: Location Set Covering', True),   # LSCP with coverage circles
        (0, 2, 1, 'MCLP: Maximum Covering', True),        # MCLP with coverage circles
        (1, 0, 2, 'P-Median Problem', False),             # P-Median with assignment lines
        (1, 1, 3, 'P-Center Problem', False),             # P-Center with assignment lines
        (1, 2, 4, 'SPLP: Simple Plant Location', False)   # SPLP with assignment lines
    ]
    
    for row, col, sol_idx, title, show_coverage in panel_configs:
        ax = axes[row, col]
        solution = solutions[sol_idx]
        
        # Plot demand points
        ax.scatter(demand_points[:, 0], demand_points[:, 1], c='lightblue', s=demand_weights*2, 
                   alpha=0.7, edgecolors='black', linewidth=0.5, label='Demand Points')
        
        # Plot potential facilities
        ax.scatter(facility_points[:, 0], facility_points[:, 1], c='gray', s=50, 
                   alpha=0.5, marker='s', label='Potential Facilities')
        
        # Plot selected facilities
        if solution:
            ax.scatter(facility_points[solution, 0], facility_points[solution, 1], 
                       c='red', s=facility_costs[solution]/2, marker='s', edgecolors='black', 
                       linewidth=2, label='Selected Facilities')
            
            # Add coverage circles for covering problems
            if show_coverage and coverage_radius:
                for j in solution:
                    circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                                   fill=False, color=colors[sol_idx], linestyle='--', alpha=0.5)
                    ax.add_patch(circle)
            
            # Add assignment lines for distance-based problems
            if not show_coverage:
                distances = cdist(demand_points, facility_points)
                for i in range(len(demand_points)):
                    closest_facility = min(solution, key=lambda j: distances[i, j])
                    ax.plot([demand_points[i, 0], facility_points[closest_facility, 0]], 
                            [demand_points[i, 1], facility_points[closest_facility, 1]], 
                            colors[sol_idx], alpha=0.3, linewidth=1)
        
        # Add facility and demand labels for each problem panel
        for i, (x, y) in enumerate(facility_points):
            ax.annotate(f'F{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=6, fontweight='bold', color='black')
        for i, (x, y) in enumerate(demand_points):
            ax.annotate(f'D{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                       fontsize=6, fontweight='bold', color='darkblue')
        
        ax.set_title(title, fontweight='bold', fontsize=10)
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
    
    plt.tight_layout()
    return fig

def create_individual_problem_visualization(demand_points, demand_weights, facility_points, facility_costs, 
                                          solution, title, color, coverage_radius=None, show_coverage=False):
    """Create visualization for a single facility location problem"""
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    
    # Plot demand points
    ax.scatter(demand_points[:, 0], demand_points[:, 1], c='lightblue', s=demand_weights*2, 
               alpha=0.7, edgecolors='black', linewidth=0.5, label='Demand Points')
    
    # Plot potential facilities
    ax.scatter(facility_points[:, 0], facility_points[:, 1], c='gray', s=50, 
               alpha=0.5, marker='s', label='Potential Facilities')
    
    # Plot selected facilities
    if solution:
        ax.scatter(facility_points[solution, 0], facility_points[solution, 1], 
                  c='red', s=facility_costs[solution]/2, marker='s', edgecolors='black', 
                  linewidth=2, label='Selected Facilities')
        
        # Add coverage circles for covering problems
        if show_coverage and coverage_radius:
            for j in solution:
                circle = Circle((facility_points[j, 0], facility_points[j, 1]), coverage_radius, 
                               fill=False, color='red', linestyle='--', alpha=0.5)
                ax.add_patch(circle)
        
        # Add assignment lines for distance-based problems
        if not show_coverage:
            distances = cdist(demand_points, facility_points)
            for i in range(len(demand_points)):
                closest_facility = min(solution, key=lambda j: distances[i, j])
                ax.plot([demand_points[i, 0], facility_points[closest_facility, 0]], 
                        [demand_points[i, 1], facility_points[closest_facility, 1]], 
                        color, alpha=0.3, linewidth=1)
    
    # Add facility numbers
    for i, (x, y) in enumerate(facility_points):
        ax.annotate(f'F{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, fontweight='bold', color='black')
    
    # Add demand point numbers
    for i, (x, y) in enumerate(demand_points):
        ax.annotate(f'D{i}', (x, y), xytext=(5, 5), textcoords='offset points', 
                   fontsize=8, fontweight='bold', color='darkblue')
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig

def compute_all_problems(demand_points, demand_weights, facility_points, facility_costs, 
                        distances, coverage_radius, num_facilities, progress_callback=None):
    """
    Solve all facility location problems using both IP and enumeration methods.
    
    Parameters:
    -----------
    demand_points : array-like
        Demand point coordinates
    demand_weights : array-like
        Demand weights
    facility_points : array-like
        Facility point coordinates
    facility_costs : array-like
        Facility opening costs
    distances : array-like
        Distance matrix
    coverage_radius : float
        Coverage radius for covering problems
    num_facilities : int
        Number of facilities for fixed-facility problems
    progress_callback : callable, optional
        Function to call with progress updates. Should accept:
        - progress: float (0.0 to 1.0)
        - current_task: str (description of current task)
        - remaining_tasks: list of str (remaining task names)
        
    Returns:
    --------
    dict
        Dictionary containing results for all problems and methods
    """
    results = {
        'ip_results': {},
        'enumeration_results': {},
        'problem_names': ['LSCP', 'MCLP', 'P-Median', 'P-Center', 'SPLP'],
        'total_problems': 10  # 5 problems × 2 methods
    }
    
    # Solve IP problems
    ip_problems = [
        ('LSCP', lambda: solve_lscp_ip(distances, coverage_radius, return_status=True)),
        ('MCLP', lambda: solve_mclp_ip(distances, coverage_radius, num_facilities, demand_weights, return_status=True)),
        ('P-Median', lambda: solve_pmedian_ip(distances, num_facilities, demand_weights, return_status=True)),
        ('P-Center', lambda: solve_pcenter_ip(distances, num_facilities, return_status=True)),
        ('SPLP', lambda: solve_splp_ip(distances, facility_costs, demand_weights, return_status=True))
    ]
    
    for i, (problem_name, solve_func) in enumerate(ip_problems):
        if progress_callback:
            remaining_ip = [name for name, _ in ip_problems[i+1:]]
            remaining_enum = [name for name, _ in ip_problems]  # All enum problems remain
            all_remaining = [f"{name} (IP)" for name in remaining_ip] + [f"{name} (Enum)" for name in remaining_enum]
            progress_callback(i / results['total_problems'], f"Solving {problem_name} using Integer Programming", all_remaining)
        
        start_time = time.time()
        try:
            solution, status, objective = solve_func()
            solve_time = time.time() - start_time
            
            if status == 'Optimal':
                # Calculate metrics
                metrics = {'facilities': len(solution)}
                if problem_name in ['LSCP', 'MCLP']:
                    metrics['coverage'] = calculate_coverage(solution, distances, coverage_radius)
                elif problem_name == 'P-Median':
                    metrics['total_distance'] = calculate_total_distance(solution, distances, demand_weights)
                elif problem_name == 'P-Center':
                    metrics['max_distance'] = calculate_max_distance(solution, distances)
                elif problem_name == 'SPLP':
                    metrics['total_cost'] = calculate_total_cost(solution, facility_costs, distances, demand_weights)
                
                results['ip_results'][problem_name] = {
                    'solution': solution,
                    'status': status,
                    'objective': objective,
                    'solve_time': solve_time,
                    'metrics': metrics,
                    'feasible': True
                }
            else:
                results['ip_results'][problem_name] = {
                    'solution': None,
                    'status': status,
                    'objective': None,
                    'solve_time': solve_time,
                    'metrics': {},
                    'feasible': False
                }
        except Exception as e:
            results['ip_results'][problem_name] = {
                'solution': None,
                'status': f'Error: {str(e)}',
                'objective': None,
                'solve_time': 0,
                'metrics': {},
                'feasible': False
            }
    
    # Solve enumeration problems
    enum_problems = [
        ('LSCP', lambda: solve_lscp_enumeration(distances, coverage_radius, progress_callback=None)),
        ('MCLP', lambda: solve_mclp_enumeration(distances, coverage_radius, num_facilities, demand_weights, progress_callback=None)),
        ('P-Median', lambda: solve_pmedian_enumeration(distances, num_facilities, demand_weights, progress_callback=None)),
        ('P-Center', lambda: solve_pcenter_enumeration(distances, num_facilities, progress_callback=None)),
        ('SPLP', lambda: solve_splp_enumeration(distances, facility_costs, demand_weights, progress_callback=None))
    ]
    
    for i, (problem_name, solve_func) in enumerate(enum_problems):
        if progress_callback:
            remaining_enum = [name for name, _ in enum_problems[i+1:]]
            all_remaining = [f"{name} (Enum)" for name in remaining_enum]
            progress_callback((i + 5) / results['total_problems'], f"Solving {problem_name} using Complete Enumeration", all_remaining)
        
        start_time = time.time()
        try:
            solution, objective, total_combinations = solve_func()
            solve_time = time.time() - start_time
            
            if solution is not None:
                # Recalculate objective for MCLP to ensure correctness
                if problem_name == 'MCLP':
                    import numpy as np
                    n_demand = distances.shape[0]
                    coverage_matrix = (distances <= coverage_radius).astype(int)
                    demand_weights_arr = np.asarray(demand_weights)
                    weighted_coverage = 0.0
                    for demand_idx in range(n_demand):
                        if any(coverage_matrix[demand_idx, j] == 1 for j in solution):
                            weighted_coverage += float(demand_weights_arr[demand_idx])
                    objective = weighted_coverage
                
                # Recalculate objective for P-Center to ensure correctness
                if problem_name == 'P-Center':
                    objective = calculate_max_distance(solution, distances)
                
                # Calculate metrics
                metrics = {'facilities': len(solution)}
                if problem_name in ['LSCP', 'MCLP']:
                    metrics['coverage'] = calculate_coverage(solution, distances, coverage_radius)
                elif problem_name == 'P-Median':
                    metrics['total_distance'] = calculate_total_distance(solution, distances, demand_weights)
                elif problem_name == 'P-Center':
                    metrics['max_distance'] = calculate_max_distance(solution, distances)
                elif problem_name == 'SPLP':
                    metrics['total_cost'] = calculate_total_cost(solution, facility_costs, distances, demand_weights)
                
                results['enumeration_results'][problem_name] = {
                    'solution': solution,
                    'objective': objective,
                    'solve_time': solve_time,
                    'total_combinations': total_combinations,
                    'metrics': metrics,
                    'feasible': True
                }
            else:
                results['enumeration_results'][problem_name] = {
                    'solution': None,
                    'objective': None,
                    'solve_time': solve_time,
                    'total_combinations': 0,
                    'metrics': {},
                    'feasible': False
                }
        except Exception as e:
            results['enumeration_results'][problem_name] = {
                'solution': None,
                'objective': None,
                'solve_time': 0,
                'total_combinations': 0,
                'metrics': {},
                'feasible': False
            }
    
    if progress_callback:
        progress_callback(1.0, "All problems solved!", [])
    
    return results

def main():
    # Initialize session state
    if 'solved_problems' not in st.session_state:
        st.session_state.solved_problems = {}
    if 'spatial_data' not in st.session_state:
        st.session_state.spatial_data = None
    if 'show_about' not in st.session_state:
        st.session_state.show_about = False
    if 'show_manual' not in st.session_state:
        st.session_state.show_manual = False
    if 'manual_solution' not in st.session_state:
        st.session_state.manual_solution = set()  # For the separate manual solution page
    if 'manual_solutions' not in st.session_state:
        st.session_state.manual_solutions = {}  # Problem-specific manual solutions
    if 'current_problem' not in st.session_state:
        st.session_state.current_problem = None
    if 'solve_error' not in st.session_state:
        st.session_state.solve_error = None
    if 'enumeration_results' not in st.session_state:
        st.session_state.enumeration_results = {}
    if 'compute_all_results' not in st.session_state:
        st.session_state.compute_all_results = None
    if 'show_comparison' not in st.session_state:
        st.session_state.show_comparison = False
    
    # Sidebar for parameters
    st.sidebar.markdown('<h1 style="text-align: center; margin-bottom: 20px;">🏭 Facility Location Optimization</h1>', unsafe_allow_html=True)
    st.sidebar.header("🔧 Common Parameters")
    
    # Sliders in vertical layout (one column)
    n_demand = st.sidebar.slider(
            "Number of Demand Points", 
            min_value=2, max_value=50, value=10, step=1,
            help="Number of customer/demand locations"
        )
        
    n_facilities = st.sidebar.slider(
            "Number of Potential Facilities", 
            min_value=2, max_value=50, value=5, step=1,
            help="Number of potential facility locations"
        )
    
    coverage_radius = st.sidebar.slider(
            "Coverage Radius", 
            min_value=1.0, max_value=5.0, value=3.0, step=0.1,
            help="Maximum distance for coverage (used by LSCP and MCLP)"
        )
        
    num_facilities = st.sidebar.slider(
            "Number of Facilities", 
            min_value=2, max_value=min(8, n_facilities), value=3, step=1,
            help="Number of facilities to locate (used by MCLP, P-Median, and P-Center)"
        )
    
    # Random seed field (full width below the grid)
    seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=1, max_value=10000, value=1234, step=1,
        help="Seed for random number generation"
    )
    
    # Manual Solution button
    if st.sidebar.button("📊 Visualize data and explore solutions", type="secondary", width='stretch'):
        st.session_state.show_manual = True
        st.session_state.show_about = False
        st.session_state.current_problem = None
        st.session_state.solve_error = None
        st.rerun()
    
    # Generate spatial data - always regenerate when parameters change
    current_params = (n_demand, n_facilities, seed, coverage_radius, num_facilities)
    if (st.session_state.spatial_data is None or 
        st.session_state.spatial_data.get('params') != current_params):
        with st.spinner("Generating spatial data..."):
            demand_points, demand_weights, facility_points, facility_costs, distances = generate_spatial_data(
                n_demand, n_facilities, seed
            )
            st.session_state.spatial_data = {
                'demand_points': demand_points,
                'demand_weights': demand_weights,
                'facility_points': facility_points,
                'facility_costs': facility_costs,
                'distances': distances,
                'params': current_params
            }
            # Clear all solutions when parameters change
            st.session_state.solved_problems = {}
            st.session_state.enumeration_results = {}
            st.session_state.compute_all_results = None
            st.session_state.manual_solutions = {}
            st.session_state.manual_solution = set()
            # Return to main screen
            st.session_state.current_problem = None
            st.session_state.show_manual = False
            st.session_state.show_about = False
            st.session_state.show_comparison = False
            st.session_state.solve_error = None
    
    # Always get current data
    demand_points = st.session_state.spatial_data['demand_points']
    demand_weights = st.session_state.spatial_data['demand_weights']
    facility_points = st.session_state.spatial_data['facility_points']
    facility_costs = st.session_state.spatial_data['facility_costs']
    distances = st.session_state.spatial_data['distances']
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 Solve Problems")
    
    # Individual solve buttons for each problem (vertical layout)
    if st.sidebar.button("🔴 LSCP", type="secondary", width='stretch'):
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = 'LSCP'
        st.session_state.solve_error = None
        st.rerun()
    
    if st.sidebar.button("🔵 MCLP", type="secondary", width='stretch'):
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = 'MCLP'
        st.session_state.solve_error = None
        st.rerun()
    
    if st.sidebar.button("🟢 P-Median", type="secondary", width='stretch'):
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = 'P-Median'
        st.session_state.solve_error = None
        st.rerun()
    
    if st.sidebar.button("🟠 P-Center", type="secondary", width='stretch'):
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = 'P-Center'
        st.session_state.solve_error = None
        st.rerun()
    
    if st.sidebar.button("🟣 SPLP", type="secondary", width='stretch'):
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = 'SPLP'
        st.session_state.solve_error = None
        st.rerun()
    
    # Compute All button
    if st.sidebar.button("🚀 Compute All", type="primary", width='stretch'):
        st.session_state.show_comparison = True
        st.session_state.show_about = False
        st.session_state.show_manual = False
        st.session_state.current_problem = None
        st.session_state.solve_error = None
        st.rerun()
    
    # About This App button
    if st.sidebar.button("ℹ️ About This App", type="secondary", width='stretch'):
        st.session_state.show_about = True
        st.session_state.show_manual = False
        st.session_state.show_comparison = False
        st.session_state.current_problem = None
        st.session_state.solve_error = None
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Credits")
    st.sidebar.markdown("**Created by:** Yaron Shaposhnik")
    st.sidebar.markdown("**Email:** yaron.shaposhnik@simon.rochester.edu")
    st.sidebar.markdown("**Year:** 2025")
    
    # Under construction warning banner (visible on all screens)
    st.warning("⚠️ **Under Construction:** This application is currently under development. Some features may be incomplete or subject to change.", icon="🚧")
    
    # Main screen content based on state
    if st.session_state.current_problem:
        problem_name = st.session_state.current_problem
        
        # Problem definitions
        problem_info = {
            'LSCP': {
                'name': 'LSCP - Location Set Covering',
                'color': 'red',
                'description': 'Minimize facilities to cover all demand',
                'constraint': 'Each demand point must be within coverage radius',
                'application': 'Emergency services, wireless coverage',
                'show_coverage': True
            },
            'MCLP': {
                'name': 'MCLP - Maximum Covering',
                'color': 'blue', 
                'description': 'Maximize coverage with fixed number of facilities',
                'constraint': 'Limited number of facilities (budget)',
                'application': 'Healthcare facilities, retail locations',
                'show_coverage': True
            },
            'P-Median': {
                'name': 'P-Median Problem',
                'color': 'green',
                'description': 'Minimize total weighted distance to facilities',
                'constraint': 'Fixed number of facilities',
                'application': 'Distribution centers, schools',
                'show_coverage': False
            },
            'P-Center': {
                'name': 'P-Center Problem',
                'color': 'orange',
                'description': 'Minimize maximum distance to furthest demand point',
                'constraint': 'Fixed number of facilities',
                'application': 'Emergency services, fire stations',
                'show_coverage': False
            },
            'SPLP': {
                'name': 'SPLP - Simple Plant Location',
                'color': 'purple',
                'description': 'Minimize total cost (facility opening + transportation)',
                'constraint': 'No fixed number of facilities',
                'application': 'Manufacturing plants, warehouses',
                'show_coverage': False
            }
        }
        
        if problem_name in problem_info:
            info = problem_info[problem_name]
            
            # Page header
            st.markdown(f"# {info['name']}")
            st.markdown(f"**Objective**: {info['description']}")
            st.markdown(f"**Constraint**: {info['constraint']}")
            st.markdown(f"**Application**: {info['application']}")
            
            # Back button
            if st.button("← Back to Main", type="secondary"):
                st.session_state.current_problem = None
                st.session_state.solve_error = None
                st.rerun()
            
            st.markdown("---")
            
            # SECTION 1: Manual Solution
            st.subheader("✋ Step 1: Manual Solution")
            st.markdown("**Instructions:** Select facilities manually using the checkboxes below to see how different solutions perform.")
            
            # Create two-column layout: facility selection on left, plot on right
            col_left, col_right = st.columns([1, 2])
            
            with col_left:
                st.markdown("### Select Facilities:")
                
                # Get problem-specific manual solution
                manual_solution = st.session_state.manual_solutions.get(problem_name, set())
                
                # Display current selection
                if manual_solution:
                    selected_list = sorted(list(manual_solution))
                    solution_list = list(manual_solution)
                    num_selected = len(solution_list)
                    
                    # Calculate objective value and check constraints for each problem
                    if problem_name == 'LSCP':
                        # Objective: number of facilities (minimize)
                        objective_value = num_selected
                        # Constraint: 100% coverage
                        coverage = calculate_coverage(solution_list, distances, coverage_radius)
                        constraint_satisfied = coverage >= 100.0
                        constraint_text = f"Coverage: {coverage:.1f}% {'✅' if constraint_satisfied else '❌'}"
                        st.info(f"**Selected:** {selected_list} | **Objective:** {objective_value} facilities | **{constraint_text}**")
                    elif problem_name == 'MCLP':
                        # Objective: weighted coverage (maximize)
                        weighted_coverage = calculate_weighted_coverage(solution_list, distances, coverage_radius, demand_weights)
                        objective_value = weighted_coverage
                        # MCLP has no constraints in manual solution
                        coverage = calculate_coverage(solution_list, distances, coverage_radius)
                        st.info(f"**Selected:** {selected_list} | **Objective:** {objective_value:.1f} | **Coverage:** {coverage:.1f}%")
                    elif problem_name == 'P-Median':
                        # Objective: total distance (minimize)
                        total_dist = calculate_total_distance(solution_list, distances, demand_weights)
                        objective_value = total_dist
                        # Constraint: exactly num_facilities
                        constraint_satisfied = num_selected == num_facilities
                        constraint_text = f"Facilities: {num_selected}/{num_facilities} {'✅' if constraint_satisfied else '❌'}"
                        st.info(f"**Selected:** {selected_list} | **Objective:** {objective_value:.1f} | **{constraint_text}**")
                    elif problem_name == 'P-Center':
                        # Objective: max distance (minimize)
                        max_dist = calculate_max_distance(solution_list, distances)
                        objective_value = max_dist
                        # Constraint: exactly num_facilities
                        constraint_satisfied = num_selected == num_facilities
                        constraint_text = f"Facilities: {num_selected}/{num_facilities} {'✅' if constraint_satisfied else '❌'}"
                        st.info(f"**Selected:** {selected_list} | **Objective:** {objective_value:.2f} | **{constraint_text}**")
                    elif problem_name == 'SPLP':
                        # Objective: total cost (minimize)
                        total_cost = calculate_total_cost(solution_list, facility_costs, distances, demand_weights)
                        objective_value = total_cost
                        # SPLP has no constraint on number of facilities
                        constraint_satisfied = True  # Always satisfied (no constraint)
                        st.info(f"**Selected:** {selected_list} | **Objective:** {objective_value:.1f} ✅")
                    else:
                        st.info(f"**Selected:** {selected_list}")
                else:
                    st.info("**No facilities selected**")
                
                # Facility selection checkboxes
                selected_facilities = []
                for i in range(len(facility_points)):
                    # Only show cost for SPLP (the only problem that uses facility costs)
                    if problem_name == 'SPLP':
                        label = f"F{i} (Cost: {facility_costs[i]:.1f})"
                    else:
                        label = f"F{i}"
                    
                    if st.checkbox(
                        label, 
                        value=i in manual_solution,
                        key=f"manual_facility_{problem_name}_{i}"
                    ):
                        selected_facilities.append(i)
                
                # Update manual solution if selection changed
                new_solution = set(selected_facilities)
                if new_solution != manual_solution:
                    st.session_state.manual_solutions[problem_name] = new_solution
                    st.rerun()
                
                # Highlight location dropdown
                st.markdown("#### Highlight Location")
                highlight_options = ["None"] + [f"D{i}" for i in range(len(demand_points))] + [f"F{i}" for i in range(len(facility_points))]
                highlighted_location = st.selectbox(
                    "Choose a location to highlight:",
                    options=highlight_options,
                    key=f"highlight_location_{problem_name}"
                )
            
            with col_right:
                # Get problem-specific manual solution
                manual_solution = st.session_state.manual_solutions.get(problem_name, set())
                
                # Create manual solution visualization
                manual_fig = create_manual_solution_visualization(
                    demand_points, demand_weights, facility_points, facility_costs, 
                    coverage_radius, manual_solution, highlighted_location
                )
                st.pyplot(manual_fig)
            
            # Display objective values for manual solution
            manual_solution = st.session_state.manual_solutions.get(problem_name, set())
            if manual_solution:
                solution_list = list(manual_solution)
                
                # All objective values are now shown in the selection label above
                # This section is kept for potential future use but currently displays nothing
            
            st.markdown("---")
            
            # SECTION 2: IP Solution
            st.subheader("🔴 Step 2: Solve using Integer Programming")
            
            # Button to solve using IP
            if st.button(f"🧮 Solve {problem_name} using Integer Programming", type="primary"):
                st.session_state.solve_error = None
                start_time = time.time()
                try:
                    if problem_name == 'LSCP':
                        solution, status, obj = solve_lscp_ip(distances, coverage_radius, return_status=True)
                    elif problem_name == 'MCLP':
                        solution, status, obj = solve_mclp_ip(distances, coverage_radius, num_facilities, demand_weights, return_status=True)
                    elif problem_name == 'P-Median':
                        solution, status, obj = solve_pmedian_ip(distances, num_facilities, demand_weights, return_status=True)
                    elif problem_name == 'P-Center':
                        solution, status, obj = solve_pcenter_ip(distances, num_facilities, return_status=True)
                    elif problem_name == 'SPLP':
                        solution, status, obj = solve_splp_ip(distances, facility_costs, demand_weights, return_status=True)
                    
                    solve_time = time.time() - start_time
                    
                    if status == 'Optimal':
                        # Calculate metrics
                        metrics = {'facilities': len(solution)}
                        if problem_name in ['LSCP', 'MCLP']:
                            metrics['coverage'] = calculate_coverage(solution, distances, coverage_radius)
                        elif problem_name == 'P-Median':
                            metrics['total_distance'] = calculate_total_distance(solution, distances, demand_weights)
                        elif problem_name == 'P-Center':
                            metrics['max_distance'] = calculate_max_distance(solution, distances)
                        elif problem_name == 'SPLP':
                            metrics['total_cost'] = calculate_total_cost(solution, facility_costs, distances, demand_weights)
                        
                        st.session_state.solved_problems[problem_name] = {
                            'solution': solution,
                            'status': status,
                            'objective': obj,
                            'solve_time': solve_time,
                            'metrics': metrics
                        }
                        st.rerun()
                    else:
                        st.session_state.solve_error = f"{problem_name} failed: {status}"
                        st.rerun()
                except Exception as e:
                    st.session_state.solve_error = f"Error solving {problem_name}: {str(e)}"
                    st.rerun()
            
            # Display IP solution if available
            if problem_name in st.session_state.solved_problems:
                problem_data = st.session_state.solved_problems[problem_name]
                st.success(f"✅ IP Solution found! Solve time: {problem_data['solve_time']:.3f} seconds")
                
                col1, col2 = st.columns([1, 2])
            with col1:
                    st.markdown("#### IP Solution Details:")
                    st.markdown(f"**Facilities**: {sorted(problem_data['solution'])}")
                    st.markdown(f"**Objective Value**: {problem_data['objective']:.2f}")
                    
                    # Objective breakdown/explanation with complete list of terms
                    if problem_name == 'LSCP':
                        st.markdown("**Objective Calculation**:")
                        solution_list = problem_data['solution']
                        st.markdown(f"• Minimize: Σ x_j (count of selected facilities)")
                        st.markdown(f"• Selected facilities contribute 1 each:")
                        for j in sorted(solution_list):
                            st.markdown(f"  - Facility F{j}: +1")
                        st.markdown(f"• **Total**: {len(solution_list)} facilities")
                    elif problem_name == 'MCLP':
                        st.markdown("**Objective Calculation**:")
                        solution_list = problem_data['solution']
                        import numpy as np
                        n_demand = distances.shape[0]
                        coverage_matrix = (distances <= coverage_radius).astype(int)
                        demand_weights_arr = np.asarray(demand_weights)
                        
                        st.markdown(f"• Maximize: Σ (w[i] × y[i]) where y[i]=1 if demand i is covered")
                        covered_points = []
                        for i in range(n_demand):
                            if any(coverage_matrix[i, j] == 1 for j in solution_list):
                                covered_points.append((i, float(demand_weights_arr[i])))
                        
                        # Calculate the sum to verify
                        calculated_sum = sum(weight for _, weight in covered_points)
                        
                        if len(covered_points) <= 10:
                            for i, weight in covered_points:
                                st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                        else:
                            for i, weight in covered_points[:5]:
                                st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                            st.markdown(f"  - ... ({len(covered_points) - 5} more covered points) ...")
                            for i, weight in covered_points[-2:]:
                                st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                        st.markdown(f"• **Total**: {calculated_sum:.1f} ({len(covered_points)} covered points)")
                        if abs(calculated_sum - problem_data['objective']) > 1e-6:
                            st.error(f"⚠️ **Mismatch!** Calculated: {calculated_sum:.1f}, Objective: {problem_data['objective']:.1f}")
                    elif problem_name == 'P-Median':
                        st.markdown("**Objective Calculation**:")
                        solution_list = problem_data['solution']
                        import numpy as np
                        n_demand = distances.shape[0]
                        demand_weights_arr = np.asarray(demand_weights)
                        
                        st.markdown(f"• Minimize: Σ (w[i] × d[i,j*]) where j* is nearest facility to i")
                        terms = []
                        for i in range(n_demand):
                            min_dist = min(distances[i, j] for j in solution_list)
                            contribution = float(demand_weights_arr[i]) * min_dist
                            terms.append((i, float(demand_weights_arr[i]), min_dist, contribution))
                        
                        if len(terms) <= 10:
                            for i, weight, dist, contrib in terms:
                                st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        else:
                            for i, weight, dist, contrib in terms[:5]:
                                st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                            st.markdown(f"  - ... ({len(terms) - 5} more demand points) ...")
                            for i, weight, dist, contrib in terms[-2:]:
                                st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        st.markdown(f"• **Total**: {problem_data['objective']:.1f}")
                    elif problem_name == 'P-Center':
                        st.markdown("**Objective Calculation**:")
                        solution_list = problem_data['solution']
                        import numpy as np
                        n_demand = distances.shape[0]
                        
                        st.markdown(f"• Minimize: max(d[i,j*] for all i) where j* is nearest facility to i")
                        distances_to_nearest = []
                        for i in range(n_demand):
                            min_dist = min(distances[i, j] for j in solution_list)
                            distances_to_nearest.append((i, min_dist))
                        
                        max_dist = problem_data['objective']
                        if len(distances_to_nearest) <= 15:
                            for i, dist in distances_to_nearest:
                                marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                                st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                        else:
                            for i, dist in distances_to_nearest[:7]:
                                marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                                st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                            st.markdown(f"  - ... ({len(distances_to_nearest) - 7} more demand points) ...")
                            for i, dist in distances_to_nearest[-3:]:
                                marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                                st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                        st.markdown(f"• **Maximum**: {max_dist:.2f}")
                    elif problem_name == 'SPLP':
                        st.markdown("**Objective Calculation**:")
                        solution_list = problem_data['solution']
                        import numpy as np
                        n_demand = distances.shape[0]
                        demand_weights_arr = np.asarray(demand_weights)
                        
                        st.markdown(f"• Minimize: Σ f[j] × x[j] + Σ (w[i] × d[i,j*] × y[i,j*])")
                        
                        # Facility opening costs
                        st.markdown(f"• **Opening Costs**:")
                        opening_terms = [(j, float(facility_costs[j])) for j in sorted(solution_list)]
                        if len(opening_terms) <= 5:
                            for j, cost in opening_terms:
                                st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                        else:
                            for j, cost in opening_terms[:3]:
                                st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                            st.markdown(f"  - ... ({len(opening_terms) - 3} more facilities) ...")
                            for j, cost in opening_terms[-2:]:
                                st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                        facility_cost_total = sum(cost for _, cost in opening_terms)
                        st.markdown(f"  **Subtotal**: {facility_cost_total:.1f}")
                        
                        # Transportation costs
                        st.markdown(f"• **Transportation Costs**:")
                        transport_terms = []
                        for i in range(n_demand):
                            nearest_facility = min(solution_list, key=lambda j: distances[i, j])
                            dist_to_nearest = distances[i, nearest_facility]
                            contrib = float(demand_weights_arr[i]) * dist_to_nearest
                            transport_terms.append((i, nearest_facility, float(demand_weights_arr[i]), dist_to_nearest, contrib))
                        
                        if len(transport_terms) <= 10:
                            for i, j, weight, dist, contrib in transport_terms:
                                st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        else:
                            for i, j, weight, dist, contrib in transport_terms[:5]:
                                st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                            st.markdown(f"  - ... ({len(transport_terms) - 5} more demand points) ...")
                            for i, j, weight, dist, contrib in transport_terms[-2:]:
                                st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        transport_cost_total = sum(contrib for _, _, _, _, contrib in transport_terms)
                        st.markdown(f"  **Subtotal**: {transport_cost_total:.1f}")
                        
                        st.markdown(f"• **Total**: {problem_data['objective']:.1f} = {facility_cost_total:.1f} + {transport_cost_total:.1f}")
                    
                    if 'coverage' in problem_data.get('metrics', {}):
                        st.markdown(f"**Coverage**: {problem_data['metrics']['coverage']:.1f}%")
                    if 'total_distance' in problem_data.get('metrics', {}):
                        st.markdown(f"**Total Distance**: {problem_data['metrics']['total_distance']:.1f}")
                    if 'max_distance' in problem_data.get('metrics', {}):
                        st.markdown(f"**Max Distance**: {problem_data['metrics']['max_distance']:.2f}")
                    if 'total_cost' in problem_data.get('metrics', {}):
                        st.markdown(f"**Total Cost**: {problem_data['metrics']['total_cost']:.1f}")
            
            with col2:
                    ip_fig = create_individual_problem_visualization(
                        demand_points, demand_weights, facility_points, facility_costs,
                        problem_data['solution'], f"{problem_name} IP: {info['name']}", 
                        info['color'], coverage_radius, info['show_coverage']
                    )
                    st.pyplot(ip_fig)
        
        if st.session_state.solve_error:
            st.error(f"❌ {st.session_state.solve_error}")
        
        st.markdown("---")
        
        # SECTION 3: Complete Enumeration Solution
        st.subheader("🔵 Step 3: Solve using Complete Enumeration")
        
        # Button to solve using Enumeration
        if st.button(f"🧮 Solve {problem_name} using Complete Enumeration", type="primary"):
            # Clear any previous enumeration results for this problem
            if problem_name in st.session_state.enumeration_results:
                del st.session_state.enumeration_results[problem_name]
            
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(progress):
                progress_bar.progress(progress)
                status_text.text(f"Testing combinations... {progress:.1%}")
            
            # Solve using enumeration based on problem type
            start_time = time.time()
            
            try:
                if problem_name == 'LSCP':
                    enum_solution, enum_objective, total_combinations = solve_lscp_enumeration(
                        distances, coverage_radius, progress_callback=update_progress
                    )
                elif problem_name == 'MCLP':
                    enum_solution, enum_objective, total_combinations = solve_mclp_enumeration(
                        distances, coverage_radius, num_facilities, demand_weights, progress_callback=update_progress
                    )
                elif problem_name == 'P-Median':
                    enum_solution, enum_objective, total_combinations = solve_pmedian_enumeration(
                        distances, num_facilities, demand_weights, progress_callback=update_progress
                    )
                elif problem_name == 'P-Center':
                    enum_solution, enum_objective, total_combinations = solve_pcenter_enumeration(
                        distances, num_facilities, progress_callback=update_progress
                    )
                elif problem_name == 'SPLP':
                    enum_solution, enum_objective, total_combinations = solve_splp_enumeration(
                        distances, facility_costs, demand_weights, progress_callback=update_progress
                    )
                
                enum_time = time.time() - start_time
                
                # Store enumeration results
                st.session_state.enumeration_results[problem_name] = {
                    'solution': enum_solution,
                    'objective': enum_objective,
                    'solve_time': enum_time,
                    'total_combinations': total_combinations
                }
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.rerun()
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Enumeration failed: {str(e)}")
        
        # Display Enumeration solution if available
        if problem_name in st.session_state.enumeration_results:
            enum_data = st.session_state.enumeration_results[problem_name]
            
            # Recalculate objective for MCLP to ensure correctness
            if problem_name == 'MCLP':
                import numpy as np
                solution_list = enum_data['solution']
                n_demand = distances.shape[0]
                coverage_matrix = (distances <= coverage_radius).astype(int)
                demand_weights_arr = np.asarray(demand_weights)
                weighted_coverage = 0.0
                for i in range(n_demand):
                    if any(coverage_matrix[i, j] == 1 for j in solution_list):
                        weighted_coverage += float(demand_weights_arr[i])
                enum_data['objective'] = weighted_coverage
                st.session_state.enumeration_results[problem_name] = enum_data
            
            st.success(f"✅ Enumeration Solution found! Tested {enum_data['total_combinations']:,} combinations in {enum_data['solve_time']:.3f} seconds")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("#### Enumeration Solution Details:")
                st.markdown(f"**Facilities**: {sorted(enum_data['solution'])}")
                st.markdown(f"**Objective Value**: {enum_data['objective']:.2f}")
                
                # Objective breakdown/explanation with complete list of terms
                if problem_name == 'LSCP':
                    st.markdown("**Objective Calculation**:")
                    solution_list = enum_data['solution']
                    st.markdown(f"• Minimize: Σ x_j (count of selected facilities)")
                    st.markdown(f"• Selected facilities contribute 1 each:")
                    for j in sorted(solution_list):
                        st.markdown(f"  - Facility F{j}: +1")
                    st.markdown(f"• **Total**: {len(solution_list)} facilities")
                elif problem_name == 'MCLP':
                    st.markdown("**Objective Calculation**:")
                    solution_list = enum_data['solution']
                    import numpy as np
                    n_demand = distances.shape[0]
                    coverage_matrix = (distances <= coverage_radius).astype(int)
                    demand_weights_arr = np.asarray(demand_weights)
                    
                    st.markdown(f"• Maximize: Σ (w[i] × y[i]) where y[i]=1 if demand i is covered")
                    covered_points = []
                    for i in range(n_demand):
                        if any(coverage_matrix[i, j] == 1 for j in solution_list):
                            covered_points.append((i, float(demand_weights_arr[i])))
                    
                    # Calculate the sum to verify
                    calculated_sum = sum(weight for _, weight in covered_points)
                    
                    if len(covered_points) <= 10:
                        for i, weight in covered_points:
                            st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                    else:
                        for i, weight in covered_points[:5]:
                            st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                        st.markdown(f"  - ... ({len(covered_points) - 5} more covered points) ...")
                        for i, weight in covered_points[-2:]:
                            st.markdown(f"  - Demand D{i}: +{weight:.1f}")
                    st.markdown(f"• **Total**: {calculated_sum:.1f} ({len(covered_points)} covered points)")
                    if abs(calculated_sum - enum_data['objective']) > 1e-6:
                        st.error(f"⚠️ **Mismatch!** Calculated: {calculated_sum:.1f}, Objective: {enum_data['objective']:.1f}")
                elif problem_name == 'P-Median':
                    st.markdown("**Objective Calculation**:")
                    solution_list = enum_data['solution']
                    import numpy as np
                    n_demand = distances.shape[0]
                    demand_weights_arr = np.asarray(demand_weights)
                    
                    st.markdown(f"• Minimize: Σ (w[i] × d[i,j*]) where j* is nearest facility to i")
                    terms = []
                    for i in range(n_demand):
                        min_dist = min(distances[i, j] for j in solution_list)
                        contribution = float(demand_weights_arr[i]) * min_dist
                        terms.append((i, float(demand_weights_arr[i]), min_dist, contribution))
                    
                    if len(terms) <= 10:
                        for i, weight, dist, contrib in terms:
                            st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                    else:
                        for i, weight, dist, contrib in terms[:5]:
                            st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        st.markdown(f"  - ... ({len(terms) - 5} more demand points) ...")
                        for i, weight, dist, contrib in terms[-2:]:
                            st.markdown(f"  - Demand D{i}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                    st.markdown(f"• **Total**: {enum_data['objective']:.1f}")
                elif problem_name == 'P-Center':
                    st.markdown("**Objective Calculation**:")
                    solution_list = enum_data['solution']
                    import numpy as np
                    n_demand = distances.shape[0]
                    
                    st.markdown(f"• Minimize: max(d[i,j*] for all i) where j* is nearest facility to i")
                    distances_to_nearest = []
                    for i in range(n_demand):
                        min_dist = min(distances[i, j] for j in solution_list)
                        distances_to_nearest.append((i, min_dist))
                    
                    max_dist = enum_data['objective']
                    if len(distances_to_nearest) <= 15:
                        for i, dist in distances_to_nearest:
                            marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                            st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                    else:
                        for i, dist in distances_to_nearest[:7]:
                            marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                            st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                        st.markdown(f"  - ... ({len(distances_to_nearest) - 7} more demand points) ...")
                        for i, dist in distances_to_nearest[-3:]:
                            marker = " ← **MAX**" if abs(dist - max_dist) < 1e-6 else ""
                            st.markdown(f"  - Demand D{i}: {dist:.2f}{marker}")
                    st.markdown(f"• **Maximum**: {max_dist:.2f}")
                elif problem_name == 'SPLP':
                    st.markdown("**Objective Calculation**:")
                    solution_list = enum_data['solution']
                    import numpy as np
                    n_demand = distances.shape[0]
                    demand_weights_arr = np.asarray(demand_weights)
                    
                    st.markdown(f"• Minimize: Σ f[j] × x[j] + Σ (w[i] × d[i,j*] × y[i,j*])")
                    
                    # Facility opening costs
                    st.markdown(f"• **Opening Costs**:")
                    opening_terms = [(j, float(facility_costs[j])) for j in sorted(solution_list)]
                    if len(opening_terms) <= 5:
                        for j, cost in opening_terms:
                            st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                    else:
                        for j, cost in opening_terms[:3]:
                            st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                        st.markdown(f"  - ... ({len(opening_terms) - 3} more facilities) ...")
                        for j, cost in opening_terms[-2:]:
                            st.markdown(f"  - Facility F{j}: +{cost:.1f}")
                    facility_cost_total = sum(cost for _, cost in opening_terms)
                    st.markdown(f"  **Subtotal**: {facility_cost_total:.1f}")
                    
                    # Transportation costs
                    st.markdown(f"• **Transportation Costs**:")
                    transport_terms = []
                    for i in range(n_demand):
                        nearest_facility = min(solution_list, key=lambda j: distances[i, j])
                        dist_to_nearest = distances[i, nearest_facility]
                        contrib = float(demand_weights_arr[i]) * dist_to_nearest
                        transport_terms.append((i, nearest_facility, float(demand_weights_arr[i]), dist_to_nearest, contrib))
                    
                    if len(transport_terms) <= 10:
                        for i, j, weight, dist, contrib in transport_terms:
                            st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                    else:
                        for i, j, weight, dist, contrib in transport_terms[:5]:
                            st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                        st.markdown(f"  - ... ({len(transport_terms) - 5} more demand points) ...")
                        for i, j, weight, dist, contrib in transport_terms[-2:]:
                            st.markdown(f"  - Demand D{i} → F{j}: {weight:.1f} × {dist:.2f} = {contrib:.1f}")
                    transport_cost_total = sum(contrib for _, _, _, _, contrib in transport_terms)
                    st.markdown(f"  **Subtotal**: {transport_cost_total:.1f}")
                    
                    st.markdown(f"• **Total**: {enum_data['objective']:.1f} = {facility_cost_total:.1f} + {transport_cost_total:.1f}")
                
                st.markdown(f"**Combinations Tested**: {enum_data['total_combinations']:,}")
            
            with col2:
                enum_fig = create_individual_problem_visualization(
                    demand_points, demand_weights, facility_points, facility_costs,
                    enum_data['solution'], f"{problem_name} Enum: {info['name']}", 
                    info['color'], coverage_radius, info['show_coverage']
                )
                st.pyplot(enum_fig)
            
        # SECTION 4: Comparison
        # Show comparison if both IP and Enumeration solutions exist
        if (problem_name in st.session_state.solved_problems and 
            problem_name in st.session_state.enumeration_results):
            st.markdown("---")
            st.subheader("🔄 Step 4: Comparison - IP vs Enumeration")
            
            problem_data = st.session_state.solved_problems[problem_name]
            enum_data = st.session_state.enumeration_results[problem_name]
            
            comp_col1, comp_col2 = st.columns(2)
            with comp_col1:
                st.markdown("#### 🔴 IP Solution")
                st.markdown(f"**Objective**: {problem_data['objective']:.2f}")
                st.markdown(f"**Facilities**: {sorted(problem_data['solution'])}")
                st.markdown(f"**Solve Time**: {problem_data['solve_time']:.3f}s")
                
                # Objective breakdown for IP
                if problem_name == 'LSCP':
                    st.markdown("*Calculation*: Count of facilities = {:.0f}".format(problem_data['objective']))
                elif problem_name == 'MCLP':
                    st.markdown("*Calculation*: Sum of weights for covered points = {:.1f}".format(problem_data['objective']))
                elif problem_name == 'P-Median':
                    st.markdown("*Calculation*: Σ(weight × distance) = {:.1f}".format(problem_data['objective']))
                elif problem_name == 'P-Center':
                    st.markdown("*Calculation*: Max distance = {:.2f}".format(problem_data['objective']))
                elif problem_name == 'SPLP':
                    ip_facility_cost = sum(facility_costs[j] for j in problem_data['solution'])
                    ip_transport_cost = calculate_total_distance(problem_data['solution'], distances, demand_weights)
                    st.markdown("*Calculation*: {:.1f} = {:.1f} (opening) + {:.1f} (transport)".format(
                        problem_data['objective'], ip_facility_cost, ip_transport_cost))
            
            with comp_col2:
                st.markdown("#### 🔵 Enumeration Solution")
                st.markdown(f"**Objective**: {enum_data['objective']:.2f}")
                st.markdown(f"**Facilities**: {sorted(enum_data['solution'])}")
                st.markdown(f"**Solve Time**: {enum_data['solve_time']:.3f}s")
                st.markdown(f"**Combinations**: {enum_data['total_combinations']:,}")
                
                # Objective breakdown for Enumeration
                if problem_name == 'LSCP':
                    st.markdown("*Calculation*: Count of facilities = {:.0f}".format(enum_data['objective']))
                elif problem_name == 'MCLP':
                    st.markdown("*Calculation*: Sum of weights for covered points = {:.1f}".format(enum_data['objective']))
                elif problem_name == 'P-Median':
                    st.markdown("*Calculation*: Σ(weight × distance) = {:.1f}".format(enum_data['objective']))
                elif problem_name == 'P-Center':
                    st.markdown("*Calculation*: Max distance = {:.2f}".format(enum_data['objective']))
                elif problem_name == 'SPLP':
                    enum_facility_cost = sum(facility_costs[j] for j in enum_data['solution'])
                    enum_transport_cost = calculate_total_distance(enum_data['solution'], distances, demand_weights)
                    st.markdown("*Calculation*: {:.1f} = {:.1f} (opening) + {:.1f} (transport)".format(
                        enum_data['objective'], enum_facility_cost, enum_transport_cost))
            
            # Check if solutions match
            if abs(problem_data['objective'] - enum_data['objective']) < 1e-6:
                st.success("✅ **Perfect Match!** Both methods found identical solutions.")
            else:
                st.warning(f"⚠️ **Different Results** IP: {problem_data['objective']:.2f}, Enum: {enum_data['objective']:.2f}")
    
    elif st.session_state.current_problem and st.session_state.solve_error:
        # Show error message
        st.error(f"❌ {st.session_state.solve_error}")
        
        # Back button
        if st.button("← Back to Main", type="secondary"):
            st.session_state.current_problem = None
            st.session_state.solve_error = None
            st.rerun()
    
    elif st.session_state.show_manual:
        # Manual solution interface
        st.subheader("✋ Manual Solution Creation")
        
        # Instructions
        st.markdown("**Instructions:** Click on facilities in the plot below to select/deselect them. Use the buttons to clear selection or go back.")
        
        # Control buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🗑️ Clear Selection", type="secondary"):
                st.session_state.manual_solution = set()
                st.rerun()
        with col2:
            if st.button("← Back to Main", type="secondary"):
                st.session_state.show_manual = False
                st.rerun()
        with col3:
            if st.button("🔄 Refresh Plot", type="secondary"):
                st.rerun()
        
        # Create two-column layout: facility selection on left, plot on right
        col_left, col_right = st.columns([1, 2])
        
        with col_left:
            st.markdown("### Select Facilities:")
            
            # Display current selection
            if st.session_state.manual_solution:
                selected_list = sorted(list(st.session_state.manual_solution))
                st.info(f"**Selected:** {selected_list}")
            else:
                st.info("**No facilities selected**")
            
            # Facility selection checkboxes
            # Note: In the separate manual solution page, we show costs since users might want to compare all problems
            selected_facilities = []
            for i in range(len(facility_points)):
                if st.checkbox(
                    f"F{i} (Cost: {facility_costs[i]:.1f})", 
                    value=i in st.session_state.manual_solution,
                    key=f"facility_{i}"
                ):
                    selected_facilities.append(i)
            
            # Update manual solution if selection changed
            new_solution = set(selected_facilities)
            if new_solution != st.session_state.manual_solution:
                st.session_state.manual_solution = new_solution
                st.rerun()
            
            # Highlight location dropdown
            st.markdown("#### Highlight Location")
            highlight_options = ["None"] + [f"D{i}" for i in range(len(demand_points))] + [f"F{i}" for i in range(len(facility_points))]
            highlighted_location = st.selectbox(
                "Choose a location to highlight:",
                options=highlight_options,
                key="highlight_location"
            )
        
        with col_right:
            # Create manual solution visualization
            manual_fig = create_manual_solution_visualization(
                demand_points, demand_weights, facility_points, facility_costs, 
                coverage_radius, st.session_state.manual_solution, highlighted_location
            )
            st.pyplot(manual_fig)
        
        # Compute and display objective values
        if st.session_state.manual_solution:
            st.markdown("### Objective Values:")
            
            # Convert to list for calculations
            solution_list = list(st.session_state.manual_solution)
            
            # Create columns for different problems
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### Coverage Problems")
                
                # LSCP - Check if all demand is covered
                coverage = calculate_coverage(solution_list, distances, coverage_radius)
                if coverage == 100.0:
                    st.success(f"**LSCP:** ✅ Feasible\nCoverage: {coverage:.1f}%\nFacilities: {len(solution_list)}")
                else:
                    st.error(f"**LSCP:** ❌ Not Feasible\nCoverage: {coverage:.1f}% (Need 100%)")
                
                # MCLP - Check if within facility limit
                if len(solution_list) <= num_facilities:
                    st.success(f"**MCLP:** ✅ Feasible\nCoverage: {coverage:.1f}%\nFacilities: {len(solution_list)}/{num_facilities}")
                else:
                    st.error(f"**MCLP:** ❌ Not Feasible\nToo many facilities: {len(solution_list)}/{num_facilities}")
            
            with col2:
                st.markdown("#### Distance Problems")
                
                # P-Median - Check if within facility limit
                if len(solution_list) == num_facilities:
                    total_dist = calculate_total_distance(solution_list, distances, demand_weights)
                    st.success(f"**P-Median:** ✅ Feasible\nTotal Distance: {total_dist:.1f}\nFacilities: {len(solution_list)}")
                else:
                    st.error(f"**P-Median:** ❌ Not Feasible\nNeed exactly {num_facilities} facilities, have {len(solution_list)}")
                
                # P-Center - Check if within facility limit
                if len(solution_list) == num_facilities:
                    max_dist = calculate_max_distance(solution_list, distances)
                    st.success(f"**P-Center:** ✅ Feasible\nMax Distance: {max_dist:.2f}\nFacilities: {len(solution_list)}")
                else:
                    st.error(f"**P-Center:** ❌ Not Feasible\nNeed exactly {num_facilities} facilities, have {len(solution_list)}")
            
            with col3:
                st.markdown("#### Cost Problem")
                
                # SPLP - Always feasible, just compute cost
                total_cost = calculate_total_cost(solution_list, facility_costs, distances, demand_weights)
                st.success(f"**SPLP:** ✅ Feasible\nTotal Cost: {total_cost:.1f}\nFacilities: {len(solution_list)}")
        
        else:
            st.info("Select some facilities to see objective values.")
        
        # Distance and data table
        st.markdown("---")
        st.markdown("### 📊 Data Details")
        
        # Create distance matrix table
        st.markdown("#### Distance Matrix")
        distance_df = pd.DataFrame(
            distances,
            index=[f"D{i}" for i in range(len(demand_points))],
            columns=[f"F{i}" for i in range(len(facility_points))]
        )
        st.dataframe(distance_df.round(2), use_container_width=True)
        
        # Create demand points details table
        st.markdown("#### Demand Points Details")
        demand_df = pd.DataFrame({
            'Demand Point': [f"D{i}" for i in range(len(demand_points))],
            'X Coordinate': demand_points[:, 0].round(2),
            'Y Coordinate': demand_points[:, 1].round(2),
            'Weight': demand_weights.astype(int)
        })
        st.dataframe(demand_df, use_container_width=True)
        
        # Create facility points details table
        st.markdown("#### Facility Points Details")
        facility_df = pd.DataFrame({
            'Facility': [f"F{i}" for i in range(len(facility_points))],
            'X Coordinate': facility_points[:, 0].round(2),
            'Y Coordinate': facility_points[:, 1].round(2),
            'Opening Cost': facility_costs.astype(int)
        })
        st.dataframe(facility_df, use_container_width=True)
        
        # Download button for Excel export
        st.markdown("---")
        st.markdown("### 📥 Download Data")
        
        # Create Excel file in memory
        from io import BytesIO
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # Sheet 1: Problem Parameters
            params_df = pd.DataFrame({
                'Parameter': [
                    'Number of Demand Points',
                    'Number of Facilities',
                    'Coverage Radius',
                    'Number of Facilities (p)',
                    'Random Seed'
                ],
                'Value': [
                    len(demand_points),
                    len(facility_points),
                    coverage_radius,
                    num_facilities,
                    seed
                ]
            })
            params_df.to_excel(writer, sheet_name='Parameters', index=False)
            
            # Sheet 2: Distance Matrix
            distance_df.round(2).to_excel(writer, sheet_name='Distance Matrix')
            
            # Sheet 3: Demand Points
            demand_df.to_excel(writer, sheet_name='Demand Points', index=False)
            
            # Sheet 4: Facility Points
            facility_df.to_excel(writer, sheet_name='Facility Points', index=False)
            
            # Sheet 5: Coordinates
            coordinates_df = pd.DataFrame({
                'Type': ['Demand'] * len(demand_points) + ['Facility'] * len(facility_points),
                'ID': [f"D{i}" for i in range(len(demand_points))] + [f"F{i}" for i in range(len(facility_points))],
                'X': list(demand_points[:, 0].round(2)) + list(facility_points[:, 0].round(2)),
                'Y': list(demand_points[:, 1].round(2)) + list(facility_points[:, 1].round(2))
            })
            coordinates_df.to_excel(writer, sheet_name='All Coordinates', index=False)
            
            # Sheet 6: Visualization - Create and save plot
            fig = create_base_data_visualization(
                demand_points, demand_weights, facility_points, facility_costs, coverage_radius
            )
            
            # Save plot to BytesIO buffer
            img_buffer = BytesIO()
            fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
            img_buffer.seek(0)
            plt.close(fig)
            
            # Create a worksheet for the visualization
            workbook = writer.book
            worksheet = workbook.create_sheet('Visualization')
            
            # Insert image into worksheet
            from openpyxl.drawing.image import Image
            img = Image(img_buffer)
            # Scale image to fit better in Excel (now that figure is larger, scale it down more)
            img.width = img.width * 0.6
            img.height = img.height * 0.6
            worksheet.add_image(img, 'A1')
        
        excel_data = output.getvalue()
        
        # Create abbreviated filename with parameters
        filename = f"FacLoc_D{len(demand_points)}_F{len(facility_points)}_R{coverage_radius:.1f}_P{num_facilities}_S{seed}.xlsx"
        
        st.download_button(
            label="📥 Download Excel File",
            data=excel_data,
            file_name=filename,
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
        
        st.info("💡 **Excel file contains 6 sheets:**\n"
                "1. **Parameters** - Problem configuration\n"
                "2. **Distance Matrix** - Distances between all demand-facility pairs\n"
                "3. **Demand Points** - Demand locations, coordinates, and weights\n"
                "4. **Facility Points** - Facility locations, coordinates, and costs\n"
                "5. **All Coordinates** - Combined coordinate data for easy plotting\n"
                "6. **Visualization** - Plot showing demand points, facilities, and coverage\n\n"
                f"📄 **Filename format:** `FacLoc_D[demands]_F[facilities]_R[radius]_P[p-facilities]_S[seed].xlsx`")
    
    elif st.session_state.show_about:
        # About this app content
        st.subheader("ℹ️ About This App")
        
        st.markdown("### How to use this app:")
        st.markdown("1. **Adjust parameters** in the sidebar (demand points, facilities, coverage radius, etc.)")
        st.markdown("2. **Click 'Visualize Data'** to see the current data visualization")
        st.markdown("3. **Click solve buttons** for the problems you want to analyze")
        st.markdown("4. **Compare results** across different optimization objectives")
        st.markdown("5. **Data updates automatically** when you change parameters")
        
        st.markdown("---")
        st.markdown("### Data Generation Process")
        st.markdown("""
        The app generates random spatial data for each problem instance:
        
        **Demand Points:**
        - Randomly distributed in a 10×10 coordinate space
        - Each point has a weight (demand) randomly sampled from 1-10
        - Point size in visualization represents demand weight
        
        **Potential Facilities:**
        - Randomly distributed in the same 10×10 coordinate space
        - Each facility has an opening cost randomly sampled from 50-200
        - Facility size in visualization represents opening cost
        
        **Distance Matrix:**
        - Euclidean distances calculated between all demand points and facilities
        - Used for coverage, distance, and cost calculations
        
        **Coverage Radius:**
        - User-defined parameter (default: 2.0)
        - Determines which demand points can be served by each facility
        - Visualized as circles around facilities
        """)
        
        st.markdown("---")
        st.markdown("### How It Works")
        st.markdown("""
        This app solves 5 classic facility location problems optimally using integer programming:
        
        - **LSCP**: Minimize facilities to cover all demand
        - **MCLP**: Maximize coverage with fixed facilities  
        - **P-Median**: Minimize total weighted distance
        - **P-Center**: Minimize maximum distance
        - **SPLP**: Minimize total cost (facility + transport)
        
        All solutions are guaranteed optimal using PuLP/CBC solver.
        """)
        
        # Button to go back
        if st.button("← Back to Main", type="secondary"):
            st.session_state.show_about = False
            st.session_state.show_manual = False
            st.rerun()
    
    elif st.session_state.show_comparison:
        # Compute All Results screen
        st.subheader("🚀 Complete Problem Analysis")
        
        # Back button
        if st.button("← Back to Main", type="secondary"):
            st.session_state.show_comparison = False
            st.session_state.compute_all_results = None
            st.rerun()
        
        # Check if we need to compute results
        if st.session_state.compute_all_results is None:
            st.info("Computing all problems using both Integer Programming and Complete Enumeration...")
            
            # Create progress bar and status displays
            progress_bar = st.progress(0)
            current_task_text = st.empty()
            remaining_tasks_text = st.empty()
            
            def update_progress(progress, current_task="", remaining_tasks=None):
                progress_bar.progress(progress)
                
                # Display current task
                if current_task:
                    current_task_text.markdown(f"**🔄 Current:** {current_task}")
                
                # Display remaining tasks
                if remaining_tasks:
                    if len(remaining_tasks) > 0:
                        remaining_str = ", ".join(remaining_tasks[:5])  # Show first 5
                        if len(remaining_tasks) > 5:
                            remaining_str += f" + {len(remaining_tasks) - 5} more"
                        remaining_tasks_text.markdown(f"**⏳ Remaining:** {remaining_str}")
                    else:
                        remaining_tasks_text.markdown("**✅ All tasks completed!**")
                else:
                    remaining_tasks_text.empty()
            
            # Compute all problems
            with st.spinner("Solving all problems..."):
                st.session_state.compute_all_results = compute_all_problems(
                    demand_points, demand_weights, facility_points, facility_costs,
                    distances, coverage_radius, num_facilities, update_progress
                )
            
            # Clear progress indicators
            progress_bar.empty()
            current_task_text.empty()
            remaining_tasks_text.empty()
            st.rerun()
        
        # Display results
        results = st.session_state.compute_all_results
        
        # Summary statistics
        st.markdown("### 📊 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            ip_feasible = sum(1 for r in results['ip_results'].values() if r['feasible'])
            st.metric("IP Feasible", f"{ip_feasible}/5", f"{ip_feasible/5*100:.0f}%")
        
        with col2:
            enum_feasible = sum(1 for r in results['enumeration_results'].values() if r['feasible'])
            st.metric("Enum Feasible", f"{enum_feasible}/5", f"{enum_feasible/5*100:.0f}%")
        
        with col3:
            total_ip_time = sum(r['solve_time'] for r in results['ip_results'].values())
            st.metric("Total IP Time", f"{total_ip_time:.3f}s")
        
        with col4:
            total_enum_time = sum(r['solve_time'] for r in results['enumeration_results'].values())
            st.metric("Total Enum Time", f"{total_enum_time:.3f}s")
        
        # Detailed comparison table
        st.markdown("### 📋 Detailed Comparison")
        
        # Create comparison data
        comparison_data = []
        for problem_name in results['problem_names']:
            ip_result = results['ip_results'][problem_name]
            enum_result = results['enumeration_results'][problem_name]
            
            # Determine if solutions match
            solutions_match = "N/A"
            if ip_result['feasible'] and enum_result['feasible']:
                if ip_result['solution'] is not None and enum_result['solution'] is not None:
                    solutions_match = "✅ Yes" if set(ip_result['solution']) == set(enum_result['solution']) else "❌ No"
                else:
                    solutions_match = "❌ No"
            elif not ip_result['feasible'] and not enum_result['feasible']:
                solutions_match = "✅ Both Infeasible"
            else:
                solutions_match = "❌ Different Feasibility"
            
            # Get objective values (convert to string to avoid mixed types in DataFrame)
            ip_obj = f"{ip_result['objective']:.2f}" if ip_result['feasible'] else "Infeasible"
            enum_obj = f"{enum_result['objective']:.2f}" if enum_result['feasible'] else "Infeasible"
            
            # Get solve times
            ip_time = ip_result['solve_time']
            enum_time = enum_result['solve_time']
            
            # Get facility counts (convert to string to avoid mixed types in DataFrame)
            ip_facilities = str(len(ip_result['solution'])) if ip_result['feasible'] and ip_result['solution'] else "N/A"
            enum_facilities = str(len(enum_result['solution'])) if enum_result['feasible'] and enum_result['solution'] else "N/A"
            
            # Determine if objectives match (use original objective values, not formatted strings)
            objectives_match = "N/A"
            if ip_result['feasible'] and enum_result['feasible']:
                if ip_result['objective'] is not None and enum_result['objective'] is not None:
                    objectives_match = "✅ Yes" if abs(ip_result['objective'] - enum_result['objective']) < 1e-6 else "❌ No"
            
            # Get solution lists
            ip_solution_str = str(sorted(ip_result['solution'])) if ip_result['feasible'] and ip_result['solution'] else "N/A"
            enum_solution_str = str(sorted(enum_result['solution'])) if enum_result['feasible'] and enum_result['solution'] else "N/A"
            
            comparison_data.append({
                'Problem': problem_name,
                'IP Feasible': "✅" if ip_result['feasible'] else "❌",
                'Enum Feasible': "✅" if enum_result['feasible'] else "❌",
                'IP Objective': ip_obj,
                'Enum Objective': enum_obj,
                'Objective Match': objectives_match,
                'Solutions Match': solutions_match,
                'IP Solution': ip_solution_str,
                'Enum Solution': enum_solution_str,
                'IP Facilities': ip_facilities,
                'Enum Facilities': enum_facilities,
                'IP Time (s)': f"{ip_time:.3f}",
                'Enum Time (s)': f"{enum_time:.3f}",
                'Relative Speed': f"IP {enum_time/ip_time:.1f}x faster" if ip_time > 0 and enum_time > 0 and enum_time > ip_time else (f"Enum {ip_time/enum_time:.1f}x faster" if ip_time > 0 and enum_time > 0 and ip_time > enum_time else "Similar")
            })
        
        # Display comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Problem-specific details
        st.markdown("### 🔍 Problem-Specific Details")
        
        for problem_name in results['problem_names']:
            with st.expander(f"📋 {problem_name} Details", expanded=False):
                ip_result = results['ip_results'][problem_name]
                enum_result = results['enumeration_results'][problem_name]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔴 Integer Programming")
                    if ip_result['feasible']:
                        st.success(f"✅ **Feasible**")
                        st.write(f"**Solution:** {sorted(ip_result['solution'])}")
                        st.write(f"**Objective:** {ip_result['objective']:.2f}")
                        st.write(f"**Solve Time:** {ip_result['solve_time']:.3f}s")
                        st.write(f"**Status:** {ip_result['status']}")
                        
                        # Problem-specific metrics
                        if 'coverage' in ip_result['metrics']:
                            st.write(f"**Coverage:** {ip_result['metrics']['coverage']:.1f}%")
                        if 'total_distance' in ip_result['metrics']:
                            st.write(f"**Total Distance:** {ip_result['metrics']['total_distance']:.1f}")
                        if 'max_distance' in ip_result['metrics']:
                            st.write(f"**Max Distance:** {ip_result['metrics']['max_distance']:.2f}")
                        if 'total_cost' in ip_result['metrics']:
                            st.write(f"**Total Cost:** {ip_result['metrics']['total_cost']:.1f}")
                    else:
                        st.error(f"❌ **Infeasible**")
                        st.write(f"**Status:** {ip_result['status']}")
                        st.write(f"**Solve Time:** {ip_result['solve_time']:.3f}s")
                
                with col2:
                    st.markdown("#### 🔵 Complete Enumeration")
                    if enum_result['feasible']:
                        st.success(f"✅ **Feasible**")
                        st.write(f"**Solution:** {sorted(enum_result['solution'])}")
                        st.write(f"**Objective:** {enum_result['objective']:.2f}")
                        st.write(f"**Solve Time:** {enum_result['solve_time']:.3f}s")
                        st.write(f"**Combinations Tested:** {enum_result['total_combinations']:,}")
                        
                        # Problem-specific metrics
                        if 'coverage' in enum_result['metrics']:
                            st.write(f"**Coverage:** {enum_result['metrics']['coverage']:.1f}%")
                        if 'total_distance' in enum_result['metrics']:
                            st.write(f"**Total Distance:** {enum_result['metrics']['total_distance']:.1f}")
                        if 'max_distance' in enum_result['metrics']:
                            st.write(f"**Max Distance:** {enum_result['metrics']['max_distance']:.2f}")
                        if 'total_cost' in enum_result['metrics']:
                            st.write(f"**Total Cost:** {enum_result['metrics']['total_cost']:.1f}")
                    else:
                        st.error(f"❌ **Infeasible**")
                        st.write(f"**Solve Time:** {enum_result['solve_time']:.3f}s")
                        st.write(f"**Combinations Tested:** {enum_result['total_combinations']:,}")
                
                # Comparison summary
                if ip_result['feasible'] and enum_result['feasible']:
                    if set(ip_result['solution']) == set(enum_result['solution']):
                        st.success("🎯 **Perfect Match!** Both methods found identical solutions.")
                    else:
                        st.warning("⚠️ **Different Solutions** The methods found different optimal solutions.")
                elif not ip_result['feasible'] and not enum_result['feasible']:
                    st.info("ℹ️ **Both Infeasible** No feasible solution exists for this problem instance.")
                else:
                    st.warning("⚠️ **Different Feasibility** One method found a solution while the other did not.")
            
    else:
        # Blank main screen
        st.markdown("### Welcome!")
        st.markdown("Use the controls in the sidebar to get started")

if __name__ == "__main__":
    main()
