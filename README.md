# Facility Location Optimization App

An interactive Streamlit application for exploring five classic facility location optimization problems using integer programming.

## Features

- **Interactive Parameter Control**: Adjust number of demand points, facilities, coverage radius, and random seed
- **Five Optimization Problems**: LSCP, MCLP, P-Median, P-Center, and SPLP
- **Optimal Solutions**: Uses integer programming with PuLP/CBC solver for guaranteed optimal solutions
- **Real-time Visualization**: Interactive plots showing optimal facility locations and assignments
- **Solution Metrics**: Detailed performance metrics for each problem type
- **Data Export**: Download results as CSV files

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:
```bash
streamlit run facility_location_app.py
```

3. Open your browser to `http://localhost:8501`

## Usage

### Parameter Settings

- **Number of Demand Points**: 10-50 (default: 25)
- **Number of Potential Facilities**: 8-25 (default: 15)  
- **Random Seed**: 1-10000 (default: 1234)
- **Coverage Radius**: 1.0-4.0 (default: 2.0)
- **P-Median/P-Center Facilities**: 2-8 (default: 3)
- **MCLP Facilities**: 2-8 (default: 4)

### Problem Types

1. **LSCP (Location Set Covering Problem)**
   - Minimize facilities to cover all demand points
   - Each demand point must be within coverage radius

2. **MCLP (Maximum Covering Location Problem)**
   - Maximize coverage with fixed number of facilities
   - Budget constraint on number of facilities

3. **P-Median Problem**
   - Minimize total weighted distance
   - Exactly p facilities must be located

4. **P-Center Problem**
   - Minimize maximum distance from any demand point
   - Exactly p facilities must be located

5. **SPLP (Simple Plant Location Problem)**
   - Minimize total cost (facility opening + transportation)
   - No limit on facilities, but each has opening cost

### Visualization

The app displays:
- **Base Data Panel**: Shows demand points (size = weight) and potential facilities (size = cost)
- **Coverage Problems**: LSCP and MCLP show coverage circles around selected facilities
- **Distance Problems**: P-Median, P-Center, and SPLP show assignment lines from demand to facilities
- **Color Coding**: Each problem type uses a distinct color scheme

### Solution Metrics

- **LSCP**: Number of facilities, coverage percentage
- **MCLP**: Number of facilities, coverage percentage  
- **P-Median**: Number of facilities, total weighted distance
- **P-Center**: Number of facilities, maximum distance
- **SPLP**: Number of facilities, total cost

## Technical Details

- **Solver**: PuLP with CBC (Coin-or Branch and Cut) for optimal solutions
- **Problem Size**: Designed for 10-50 demand points and 8-25 facilities
- **Random Generation**: Uses numpy with configurable seed for reproducible results
- **Spatial Data**: Demand and facility locations generated in 10×8 coordinate space

## Files

- `facility_location_app.py`: Main Streamlit application
- `facility_location_solver.py`: Optimization functions (imported by app)
- `requirements.txt`: Python package dependencies
- `multi_panel_figure.ipynb`: Jupyter notebook version

## Educational Use

This app is designed for educational purposes in operations research and facility location courses. It demonstrates:

- Different optimization objectives and their trade-offs
- Impact of problem parameters on solutions
- Visual interpretation of optimization results
- Comparison between different facility location models

## Troubleshooting

- **Import Error**: Ensure `facility_location_solver.py` is in the same directory
- **Solver Issues**: PuLP requires CBC solver (usually installed automatically)
- **Memory Issues**: Reduce problem size for very large instances
- **Slow Performance**: Integer programming can be slow for large problems

## License

Educational use - OMG411 Course Material
