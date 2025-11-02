"""
Unit tests for facility location solver.

Tests all five facility location problems (LSCP, MCLP, P-Median, P-Center, SPLP)
using both Integer Programming and Complete Enumeration methods.

Run with: pytest test_facility_location_solver.py -v
Or: python -m pytest test_facility_location_solver.py -v
Or: python test_facility_location_solver.py (runs basic tests)
"""

import numpy as np
import unittest
from facility_location_solver import (
    solve_lscp_ip, solve_mclp_ip, solve_pmedian_ip, solve_pcenter_ip, solve_splp_ip,
    solve_lscp_enumeration, solve_mclp_enumeration, solve_pmedian_enumeration,
    solve_pcenter_enumeration, solve_splp_enumeration,
    calculate_coverage, calculate_weighted_coverage, calculate_total_distance,
    calculate_max_distance, calculate_total_cost
)


class TestFacilityLocationSolver(unittest.TestCase):
    """Test suite for facility location optimization problems"""
    
    def setUp(self):
        """Set up test data - simple 3x3 problem"""
        # 3 demand points at corners of a triangle
        self.demand_points = np.array([
            [1.0, 1.0],
            [9.0, 1.0],
            [5.0, 8.0]
        ])
        
        # 4 potential facility locations
        self.facility_points = np.array([
            [2.0, 2.0],
            [8.0, 2.0],
            [5.0, 7.0],
            [5.0, 5.0]
        ])
        
        # Demand weights (population, demand, etc.)
        self.demand_weights = np.array([10.0, 15.0, 20.0])
        
        # Facility opening costs
        self.facility_costs = np.array([100.0, 120.0, 90.0, 110.0])
        
        # Distance matrix (Euclidean distances)
        self.distances = np.zeros((len(self.demand_points), len(self.facility_points)))
        for i in range(len(self.demand_points)):
            for j in range(len(self.facility_points)):
                dx = self.demand_points[i, 0] - self.facility_points[j, 0]
                dy = self.demand_points[i, 1] - self.facility_points[j, 1]
                self.distances[i, j] = np.sqrt(dx**2 + dy**2)
        
        # Coverage radius for covering problems
        self.coverage_radius = 5.0
        
        # Number of facilities for p-facility problems
        self.p_facilities = 2
    
    # =========================
    # LSCP Tests
    # =========================
    
    def test_lscp_ip_feasible(self):
        """Test LSCP IP returns a feasible solution"""
        solution, status, objective = solve_lscp_ip(
            self.distances, self.coverage_radius, return_status=True
        )
        
        self.assertEqual(status, 'Optimal')
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        self.assertIsInstance(objective, (int, float))
        
        # Verify 100% coverage
        coverage = calculate_coverage(solution, self.distances, self.coverage_radius)
        self.assertAlmostEqual(coverage, 100.0, places=1)
    
    def test_lscp_enumeration_feasible(self):
        """Test LSCP enumeration returns a feasible solution"""
        solution, objective, total_combinations = solve_lscp_enumeration(
            self.distances, self.coverage_radius
        )
        
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        self.assertGreater(total_combinations, 0)
        
        # Verify 100% coverage
        coverage = calculate_coverage(solution, self.distances, self.coverage_radius)
        self.assertAlmostEqual(coverage, 100.0, places=1)
    
    def test_lscp_ip_vs_enumeration(self):
        """Test LSCP IP and enumeration give same objective"""
        ip_solution, ip_status, ip_objective = solve_lscp_ip(
            self.distances, self.coverage_radius, return_status=True
        )
        enum_solution, enum_objective, _ = solve_lscp_enumeration(
            self.distances, self.coverage_radius
        )
        
        # Both should minimize number of facilities
        self.assertEqual(len(ip_solution), ip_objective)
        self.assertEqual(len(enum_solution), enum_objective)
        self.assertEqual(ip_objective, enum_objective)
    
    # =========================
    # MCLP Tests
    # =========================
    
    def test_mclp_ip_feasible(self):
        """Test MCLP IP returns a feasible solution"""
        solution, status, objective = solve_mclp_ip(
            self.distances, self.coverage_radius, self.p_facilities,
            self.demand_weights, return_status=True
        )
        
        self.assertEqual(status, 'Optimal')
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        
        # Verify objective matches weighted coverage calculation
        weighted_coverage = calculate_weighted_coverage(
            solution, self.distances, self.coverage_radius, self.demand_weights
        )
        self.assertAlmostEqual(objective, weighted_coverage, places=2)
    
    def test_mclp_enumeration_feasible(self):
        """Test MCLP enumeration returns a feasible solution"""
        solution, objective, total_combinations = solve_mclp_enumeration(
            self.distances, self.coverage_radius, self.p_facilities, self.demand_weights
        )
        
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        self.assertGreater(total_combinations, 0)
        
        # Verify objective matches weighted coverage calculation
        weighted_coverage = calculate_weighted_coverage(
            solution, self.distances, self.coverage_radius, self.demand_weights
        )
        self.assertAlmostEqual(objective, weighted_coverage, places=2)
    
    def test_mclp_ip_vs_enumeration(self):
        """Test MCLP IP and enumeration give same objective"""
        ip_solution, ip_status, ip_objective = solve_mclp_ip(
            self.distances, self.coverage_radius, self.p_facilities,
            self.demand_weights, return_status=True
        )
        enum_solution, enum_objective, _ = solve_mclp_enumeration(
            self.distances, self.coverage_radius, self.p_facilities, self.demand_weights
        )
        
        # Both should maximize weighted coverage
        self.assertAlmostEqual(ip_objective, enum_objective, places=2)
    
    # =========================
    # P-Median Tests
    # =========================
    
    def test_pmedian_ip_feasible(self):
        """Test P-Median IP returns a feasible solution"""
        solution, status, objective = solve_pmedian_ip(
            self.distances, self.p_facilities, self.demand_weights, return_status=True
        )
        
        self.assertEqual(status, 'Optimal')
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        
        # Verify objective matches total distance calculation
        total_distance = calculate_total_distance(
            solution, self.distances, self.demand_weights
        )
        self.assertAlmostEqual(objective, total_distance, places=2)
    
    def test_pmedian_enumeration_feasible(self):
        """Test P-Median enumeration returns a feasible solution"""
        solution, objective, total_combinations = solve_pmedian_enumeration(
            self.distances, self.p_facilities, self.demand_weights
        )
        
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        self.assertGreater(total_combinations, 0)
        
        # Verify objective matches total distance calculation
        total_distance = calculate_total_distance(
            solution, self.distances, self.demand_weights
        )
        self.assertAlmostEqual(objective, total_distance, places=2)
    
    def test_pmedian_ip_vs_enumeration(self):
        """Test P-Median IP and enumeration give same objective"""
        ip_solution, ip_status, ip_objective = solve_pmedian_ip(
            self.distances, self.p_facilities, self.demand_weights, return_status=True
        )
        enum_solution, enum_objective, _ = solve_pmedian_enumeration(
            self.distances, self.p_facilities, self.demand_weights
        )
        
        # Both should minimize weighted distance
        self.assertAlmostEqual(ip_objective, enum_objective, places=2)
    
    # =========================
    # P-Center Tests
    # =========================
    
    def test_pcenter_ip_feasible(self):
        """Test P-Center IP returns a feasible solution"""
        solution, status, objective = solve_pcenter_ip(
            self.distances, self.p_facilities, return_status=True
        )
        
        self.assertEqual(status, 'Optimal')
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        
        # Verify objective matches max distance calculation
        max_distance = calculate_max_distance(solution, self.distances)
        self.assertAlmostEqual(objective, max_distance, places=2)
    
    def test_pcenter_enumeration_feasible(self):
        """Test P-Center enumeration returns a feasible solution"""
        solution, objective, total_combinations = solve_pcenter_enumeration(
            self.distances, self.p_facilities
        )
        
        self.assertEqual(len(solution), self.p_facilities)
        self.assertGreater(objective, 0)
        self.assertGreater(total_combinations, 0)
        
        # Verify objective matches max distance calculation
        max_distance = calculate_max_distance(solution, self.distances)
        self.assertAlmostEqual(objective, max_distance, places=2)
    
    def test_pcenter_ip_vs_enumeration(self):
        """Test P-Center IP and enumeration give same objective"""
        ip_solution, ip_status, ip_objective = solve_pcenter_ip(
            self.distances, self.p_facilities, return_status=True
        )
        enum_solution, enum_objective, _ = solve_pcenter_enumeration(
            self.distances, self.p_facilities
        )
        
        # Both should minimize maximum distance
        self.assertAlmostEqual(ip_objective, enum_objective, places=2)
    
    # =========================
    # SPLP Tests
    # =========================
    
    def test_splp_ip_feasible(self):
        """Test SPLP IP returns a feasible solution"""
        solution, status, objective = solve_splp_ip(
            self.distances, self.facility_costs, self.demand_weights, return_status=True
        )
        
        self.assertEqual(status, 'Optimal')
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        self.assertGreater(objective, 0)
        
        # Verify objective includes both opening and transportation costs
        opening_cost = sum(self.facility_costs[j] for j in solution)
        transport_cost = calculate_total_distance(
            solution, self.distances, self.demand_weights
        )
        total_cost = opening_cost + transport_cost
        self.assertAlmostEqual(objective, total_cost, places=2)
    
    def test_splp_enumeration_feasible(self):
        """Test SPLP enumeration returns a feasible solution"""
        solution, objective, total_combinations = solve_splp_enumeration(
            self.distances, self.facility_costs, self.demand_weights
        )
        
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)
        self.assertGreater(objective, 0)
        self.assertGreater(total_combinations, 0)
        
        # Verify objective includes both opening and transportation costs
        opening_cost = sum(self.facility_costs[j] for j in solution)
        transport_cost = calculate_total_distance(
            solution, self.distances, self.demand_weights
        )
        total_cost = opening_cost + transport_cost
        self.assertAlmostEqual(objective, total_cost, places=2)
    
    def test_splp_ip_vs_enumeration(self):
        """Test SPLP IP and enumeration give same objective"""
        ip_solution, ip_status, ip_objective = solve_splp_ip(
            self.distances, self.facility_costs, self.demand_weights, return_status=True
        )
        enum_solution, enum_objective, _ = solve_splp_enumeration(
            self.distances, self.facility_costs, self.demand_weights
        )
        
        # Both should minimize total cost
        self.assertAlmostEqual(ip_objective, enum_objective, places=2)
    
    # =========================
    # Helper Function Tests
    # =========================
    
    def test_calculate_coverage(self):
        """Test coverage calculation"""
        # Test with all facilities - should have 100% coverage with right radius
        all_facilities = list(range(len(self.facility_points)))
        coverage = calculate_coverage(all_facilities, self.distances, 10.0)
        self.assertAlmostEqual(coverage, 100.0, places=1)
        
        # Test with no facilities
        coverage = calculate_coverage([], self.distances, self.coverage_radius)
        self.assertEqual(coverage, 0.0)
    
    def test_calculate_weighted_coverage(self):
        """Test weighted coverage calculation"""
        # Test with all facilities
        all_facilities = list(range(len(self.facility_points)))
        weighted = calculate_weighted_coverage(
            all_facilities, self.distances, 10.0, self.demand_weights
        )
        # Should cover all demand
        self.assertAlmostEqual(weighted, sum(self.demand_weights), places=1)
        
        # Test with no facilities
        weighted = calculate_weighted_coverage(
            [], self.distances, self.coverage_radius, self.demand_weights
        )
        self.assertEqual(weighted, 0.0)
    
    def test_calculate_total_distance(self):
        """Test total distance calculation"""
        # Test with single facility (middle one)
        solution = [3]  # Facility at center
        total_dist = calculate_total_distance(
            solution, self.distances, self.demand_weights
        )
        self.assertGreater(total_dist, 0)
        
        # Verify manual calculation
        expected = sum(
            self.demand_weights[i] * self.distances[i, 3]
            for i in range(len(self.demand_points))
        )
        self.assertAlmostEqual(total_dist, expected, places=2)
    
    def test_calculate_max_distance(self):
        """Test max distance calculation"""
        # Test with single facility
        solution = [3]  # Facility at center
        max_dist = calculate_max_distance(solution, self.distances)
        self.assertGreater(max_dist, 0)
        
        # Verify it's actually the maximum
        expected = max(self.distances[i, 3] for i in range(len(self.demand_points)))
        self.assertAlmostEqual(max_dist, expected, places=2)
    
    def test_calculate_total_cost(self):
        """Test total cost calculation for SPLP"""
        solution = [0, 2]  # Two facilities
        total_cost = calculate_total_cost(
            solution, self.facility_costs, self.distances, self.demand_weights
        )
        
        # Should be opening cost + transportation cost
        opening = sum(self.facility_costs[j] for j in solution)
        transport = calculate_total_distance(solution, self.distances, self.demand_weights)
        expected = opening + transport
        
        self.assertAlmostEqual(total_cost, expected, places=2)
    
    # =========================
    # Edge Case Tests
    # =========================
    
    def test_single_facility_available(self):
        """Test with only one facility available"""
        distances = self.distances[:, [0]]  # Only first facility
        facility_costs = np.array([100.0])
        
        # LSCP - may be infeasible if coverage is not 100%
        solution, status, obj = solve_lscp_ip(distances, self.coverage_radius, return_status=True)
        self.assertIn(status, ['Optimal', 'Infeasible'])
        
        # SPLP - should always be feasible
        solution, status, obj = solve_splp_ip(
            distances, facility_costs, self.demand_weights, return_status=True
        )
        self.assertEqual(status, 'Optimal')
        self.assertEqual(solution, [0])
    
    def test_zero_coverage_radius(self):
        """Test LSCP with zero coverage radius - should need many facilities"""
        # With zero radius, each demand needs its own facility nearby
        solution, status, obj = solve_lscp_ip(
            self.distances, 0.01, return_status=True
        )
        # Will likely be infeasible unless facilities are exactly at demand points
        self.assertIsNotNone(status)
    
    def test_large_coverage_radius(self):
        """Test LSCP with large coverage radius - should need only one facility"""
        solution, status, obj = solve_lscp_ip(
            self.distances, 100.0, return_status=True
        )
        self.assertEqual(status, 'Optimal')
        self.assertEqual(len(solution), 1)  # One facility should cover everything


class TestConsistencyAcrossMethods(unittest.TestCase):
    """Test that IP and Enumeration methods are consistent"""
    
    def setUp(self):
        """Set up small test problem for enumeration"""
        # Very small problem for quick enumeration
        self.demand_points = np.array([[2.0, 2.0], [8.0, 8.0]])
        self.facility_points = np.array([[1.0, 1.0], [5.0, 5.0], [9.0, 9.0]])
        self.demand_weights = np.array([5.0, 10.0])
        self.facility_costs = np.array([50.0, 60.0, 70.0])
        
        # Calculate distances
        self.distances = np.zeros((2, 3))
        for i in range(2):
            for j in range(3):
                dx = self.demand_points[i, 0] - self.facility_points[j, 0]
                dy = self.demand_points[i, 1] - self.facility_points[j, 1]
                self.distances[i, j] = np.sqrt(dx**2 + dy**2)
        
        self.coverage_radius = 6.0
        self.p_facilities = 2
    
    def test_all_problems_consistency(self):
        """Test that all five problems give consistent results between IP and Enum"""
        
        # LSCP
        lscp_ip = solve_lscp_ip(self.distances, self.coverage_radius, return_status=True)[2]
        lscp_enum = solve_lscp_enumeration(self.distances, self.coverage_radius)[1]
        self.assertEqual(lscp_ip, lscp_enum, "LSCP IP and Enum objectives differ")
        
        # MCLP
        mclp_ip = solve_mclp_ip(self.distances, self.coverage_radius, self.p_facilities,
                                 self.demand_weights, return_status=True)[2]
        mclp_enum = solve_mclp_enumeration(self.distances, self.coverage_radius,
                                           self.p_facilities, self.demand_weights)[1]
        self.assertAlmostEqual(mclp_ip, mclp_enum, places=2, 
                              msg="MCLP IP and Enum objectives differ")
        
        # P-Median
        pmedian_ip = solve_pmedian_ip(self.distances, self.p_facilities,
                                       self.demand_weights, return_status=True)[2]
        pmedian_enum = solve_pmedian_enumeration(self.distances, self.p_facilities,
                                                  self.demand_weights)[1]
        self.assertAlmostEqual(pmedian_ip, pmedian_enum, places=2,
                              msg="P-Median IP and Enum objectives differ")
        
        # P-Center
        pcenter_ip = solve_pcenter_ip(self.distances, self.p_facilities, return_status=True)[2]
        pcenter_enum = solve_pcenter_enumeration(self.distances, self.p_facilities)[1]
        self.assertAlmostEqual(pcenter_ip, pcenter_enum, places=2,
                              msg="P-Center IP and Enum objectives differ")
        
        # SPLP
        splp_ip = solve_splp_ip(self.distances, self.facility_costs,
                                 self.demand_weights, return_status=True)[2]
        splp_enum = solve_splp_enumeration(self.distances, self.facility_costs,
                                           self.demand_weights)[1]
        self.assertAlmostEqual(splp_ip, splp_enum, places=2,
                              msg="SPLP IP and Enum objectives differ")


def run_basic_tests():
    """Run basic smoke tests without pytest"""
    print("="*70)
    print("Running Basic Facility Location Solver Tests")
    print("="*70)
    
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFacilityLocationSolver)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "="*70)
    print("Running Consistency Tests")
    print("="*70)
    
    suite2 = unittest.TestLoader().loadTestsFromTestCase(TestConsistencyAcrossMethods)
    result2 = runner.run(suite2)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    total_tests = result.testsRun + result2.testsRun
    total_failures = len(result.failures) + len(result2.failures)
    total_errors = len(result.errors) + len(result2.errors)
    
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {total_tests - total_failures - total_errors}")
    print(f"Failed: {total_failures}")
    print(f"Errors: {total_errors}")
    
    if total_failures == 0 and total_errors == 0:
        print("\n[PASS] ALL TESTS PASSED!")
        return 0
    else:
        print("\n[FAIL] SOME TESTS FAILED!")
        return 1


if __name__ == '__main__':
    # Run basic tests if executed directly
    import sys
    sys.exit(run_basic_tests())

