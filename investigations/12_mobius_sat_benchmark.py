#!/usr/bin/env python3
"""
Investigation 12: Möbius SAT vs WalkSAT vs MiniSAT Benchmark
=============================================================

Comprehensive benchmark comparing:
1. 18-strip Möbius SAT (Investigation 11 winner)
2. WalkSAT (local search - industry standard)
3. DPLL solver (complete search - MiniSAT-style)

Tests on multiple problem sizes at phase transition (m ~ 4.2n).

Author: tHHmL Investigation Suite
Date: 2025-12-19
"""

import sys
from pathlib import Path
import time
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import warnings

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hhml.core.optimization.mobius_sat_solver import (
    SATInstance as MobiusSATInstance,
    solve_mobius_sat
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class BenchmarkResult:
    """Results from a single solver run."""
    solver_name: str
    satisfaction_ratio: float
    solve_time: float
    num_satisfied: int
    total_clauses: int
    n_vars: int
    success: bool


class WalkSAT:
    """
    WalkSAT local search solver.

    Industry-standard stochastic local search algorithm.
    Randomly flips variables to satisfy clauses.
    """

    def __init__(self, instance: MobiusSATInstance):
        self.instance = instance
        self.n_vars = instance.n_vars
        self.clauses = instance.clauses

    def solve(self, max_flips: int = 10000, p: float = 0.5, seed: Optional[int] = None) -> Dict:
        """
        Run WalkSAT algorithm.

        Args:
            max_flips: Maximum number of variable flips
            p: Probability of random walk (vs greedy)
            seed: Random seed

        Returns:
            Result dictionary with satisfaction ratio and assignment
        """
        if seed is not None:
            np.random.seed(seed)

        # Random initial assignment
        assignment = np.random.choice([-1, 1], size=self.n_vars)

        best_assignment = assignment.copy()
        best_satisfaction = self._evaluate(assignment)

        for flip in range(max_flips):
            # Find unsatisfied clauses
            unsatisfied = []
            for i, clause in enumerate(self.clauses):
                if not self._is_satisfied(clause, assignment):
                    unsatisfied.append((i, clause))

            if len(unsatisfied) == 0:
                # Found satisfying assignment
                return {
                    'satisfaction_ratio': 1.0,
                    'num_satisfied': len(self.clauses),
                    'total_clauses': len(self.clauses),
                    'assignment': assignment,
                    'flips': flip
                }

            # Pick random unsatisfied clause
            clause_idx, clause = unsatisfied[np.random.randint(len(unsatisfied))]

            # With probability p, do random walk
            if np.random.rand() < p:
                # Flip random variable in clause
                literal = clause[np.random.randint(len(clause))]
                var_idx = abs(literal) - 1
                assignment[var_idx] *= -1
            else:
                # Greedy: flip variable that maximizes satisfied clauses
                best_var = None
                best_delta = -float('inf')

                for literal in clause:
                    var_idx = abs(literal) - 1

                    # Try flipping this variable
                    assignment[var_idx] *= -1
                    delta = self._evaluate(assignment) - best_satisfaction
                    assignment[var_idx] *= -1  # Flip back

                    if delta > best_delta:
                        best_delta = delta
                        best_var = var_idx

                if best_var is not None:
                    assignment[best_var] *= -1

            # Update best
            satisfaction = self._evaluate(assignment)
            if satisfaction > best_satisfaction:
                best_satisfaction = satisfaction
                best_assignment = assignment.copy()

        # Return best found
        num_satisfied = int(best_satisfaction * len(self.clauses))
        return {
            'satisfaction_ratio': best_satisfaction,
            'num_satisfied': num_satisfied,
            'total_clauses': len(self.clauses),
            'assignment': best_assignment,
            'flips': max_flips
        }

    def _is_satisfied(self, clause: List[int], assignment: np.ndarray) -> bool:
        """Check if clause is satisfied."""
        for literal in clause:
            var_idx = abs(literal) - 1
            if literal > 0 and assignment[var_idx] == 1:
                return True
            if literal < 0 and assignment[var_idx] == -1:
                return True
        return False

    def _evaluate(self, assignment: np.ndarray) -> float:
        """Evaluate satisfaction ratio."""
        satisfied = sum(1 for clause in self.clauses if self._is_satisfied(clause, assignment))
        return satisfied / len(self.clauses)


class HybridMobiusWalkSAT:
    """
    Hybrid Möbius+WalkSAT solver.

    Uses Möbius SAT to quickly find approximate solution (~92% satisfaction),
    then uses WalkSAT to refine to near-perfect solution.

    Hypothesis: Starting from 92% should be much faster than starting from
    random 50% assignment.
    """

    def __init__(self, instance: MobiusSATInstance):
        self.instance = instance
        self.n_vars = instance.n_vars
        self.clauses = instance.clauses

    def solve(self,
              mobius_strips: int = 20,
              mobius_omega: float = 0.1,
              walksat_max_flips: int = 5000,  # Reduced from 10000
              walksat_p: float = 0.5,
              seed: Optional[int] = None) -> Dict:
        """
        Run hybrid solver: Möbius SAT -> WalkSAT refinement.

        Args:
            mobius_strips: Number of Möbius strips (20 optimal - prime benchmark testing)
            mobius_omega: Omega parameter for Möbius SAT
            walksat_max_flips: Max flips for WalkSAT refinement
            walksat_p: WalkSAT random walk probability
            seed: Random seed

        Returns:
            Result dictionary with combined metrics
        """
        # Phase 1: Möbius SAT for fast approximate solution
        mobius_start = time.time()
        mobius_result = solve_mobius_sat(
            self.instance,
            num_strips=mobius_strips,
            omega=mobius_omega,
            num_iterations=3,
            seed=seed
        )
        mobius_time = time.time() - mobius_start

        # Phase 2: WalkSAT refinement starting from Möbius solution
        initial_assignment = mobius_result['assignment']

        if seed is not None:
            np.random.seed(seed + 1)  # Different seed for WalkSAT phase

        assignment = initial_assignment.copy()
        best_assignment = assignment.copy()
        best_satisfaction = self._evaluate(assignment)

        walksat_start = time.time()
        for flip in range(walksat_max_flips):
            # Find unsatisfied clauses
            unsatisfied = []
            for i, clause in enumerate(self.clauses):
                if not self._is_satisfied(clause, assignment):
                    unsatisfied.append((i, clause))

            if len(unsatisfied) == 0:
                # Found perfect solution
                walksat_time = time.time() - walksat_start
                total_time = mobius_time + walksat_time

                return {
                    'satisfaction_ratio': 1.0,
                    'num_satisfied': len(self.clauses),
                    'total_clauses': len(self.clauses),
                    'assignment': assignment,
                    'mobius_time': mobius_time,
                    'walksat_time': walksat_time,
                    'total_time': total_time,
                    'mobius_satisfaction': mobius_result['satisfaction_ratio'],
                    'walksat_flips': flip,
                    'improvement': 1.0 - mobius_result['satisfaction_ratio']
                }

            # Pick random unsatisfied clause
            clause_idx, clause = unsatisfied[np.random.randint(len(unsatisfied))]

            # With probability p, do random walk
            if np.random.rand() < walksat_p:
                # Flip random variable in clause
                literal = clause[np.random.randint(len(clause))]
                var_idx = abs(literal) - 1
                assignment[var_idx] *= -1
            else:
                # Greedy: flip variable that maximizes satisfied clauses
                best_var = None
                best_delta = -float('inf')

                for literal in clause:
                    var_idx = abs(literal) - 1

                    # Try flipping this variable
                    assignment[var_idx] *= -1
                    delta = self._evaluate(assignment) - best_satisfaction
                    assignment[var_idx] *= -1  # Flip back

                    if delta > best_delta:
                        best_delta = delta
                        best_var = var_idx

                if best_var is not None:
                    assignment[best_var] *= -1

            # Update best
            satisfaction = self._evaluate(assignment)
            if satisfaction > best_satisfaction:
                best_satisfaction = satisfaction
                best_assignment = assignment.copy()

        # Return best found after max flips
        walksat_time = time.time() - walksat_start
        total_time = mobius_time + walksat_time

        num_satisfied = int(best_satisfaction * len(self.clauses))
        return {
            'satisfaction_ratio': best_satisfaction,
            'num_satisfied': num_satisfied,
            'total_clauses': len(self.clauses),
            'assignment': best_assignment,
            'mobius_time': mobius_time,
            'walksat_time': walksat_time,
            'total_time': total_time,
            'mobius_satisfaction': mobius_result['satisfaction_ratio'],
            'walksat_flips': walksat_max_flips,
            'improvement': best_satisfaction - mobius_result['satisfaction_ratio']
        }

    def _is_satisfied(self, clause: List[int], assignment: np.ndarray) -> bool:
        """Check if clause is satisfied."""
        for literal in clause:
            var_idx = abs(literal) - 1
            if literal > 0 and assignment[var_idx] == 1:
                return True
            if literal < 0 and assignment[var_idx] == -1:
                return True
        return False

    def _evaluate(self, assignment: np.ndarray) -> float:
        """Evaluate satisfaction ratio."""
        satisfied = sum(1 for clause in self.clauses if self._is_satisfied(clause, assignment))
        return satisfied / len(self.clauses)


class DPLLSolver:
    """
    DPLL-based complete SAT solver (MiniSAT-style).

    Uses backtracking with unit propagation and pure literal elimination.
    """

    def __init__(self, instance: MobiusSATInstance):
        self.instance = instance
        self.n_vars = instance.n_vars
        self.clauses = instance.clauses
        self.assignment = {}
        self.decisions = 0

    def solve(self, timeout: float = 10.0, seed: Optional[int] = None) -> Dict:
        """
        Run DPLL solver with timeout.

        Args:
            timeout: Maximum solve time in seconds
            seed: Random seed for variable ordering

        Returns:
            Result dictionary
        """
        if seed is not None:
            np.random.seed(seed)

        start_time = time.time()
        self.assignment = {}
        self.decisions = 0

        # Try to solve with DPLL
        try:
            result = self._dpll(timeout=timeout, start_time=start_time)

            if result:
                # Found satisfying assignment
                assignment = self._to_array()
                return {
                    'satisfaction_ratio': 1.0,
                    'num_satisfied': len(self.clauses),
                    'total_clauses': len(self.clauses),
                    'assignment': assignment,
                    'decisions': self.decisions
                }
            else:
                # UNSAT (shouldn't happen for random instances at phase transition)
                # Return random assignment
                assignment = np.random.choice([-1, 1], size=self.n_vars)
                satisfaction, num_sat = self.instance.evaluate(assignment)
                return {
                    'satisfaction_ratio': satisfaction,
                    'num_satisfied': num_sat,
                    'total_clauses': len(self.clauses),
                    'assignment': assignment,
                    'decisions': self.decisions
                }
        except TimeoutError:
            # Timeout - return best partial assignment
            if len(self.assignment) == 0:
                assignment = np.random.choice([-1, 1], size=self.n_vars)
            else:
                assignment = self._to_array()

            satisfaction, num_sat = self.instance.evaluate(assignment)
            return {
                'satisfaction_ratio': satisfaction,
                'num_satisfied': num_sat,
                'total_clauses': len(self.clauses),
                'assignment': assignment,
                'decisions': self.decisions,
                'timeout': True
            }

    def _dpll(self, timeout: float, start_time: float) -> bool:
        """Recursive DPLL algorithm."""
        # Check timeout
        if time.time() - start_time > timeout:
            raise TimeoutError()

        # Unit propagation
        while True:
            unit_clause = self._find_unit_clause()
            if unit_clause is None:
                break

            literal = unit_clause[0]
            var_idx = abs(literal)
            value = 1 if literal > 0 else -1
            self.assignment[var_idx] = value

        # Check if satisfied
        if self._all_satisfied():
            return True

        # Check if conflict
        if self._has_conflict():
            return False

        # Choose unassigned variable
        var = self._choose_variable()
        if var is None:
            return self._all_satisfied()

        # Try positive assignment
        self.decisions += 1
        self.assignment[var] = 1
        if self._dpll(timeout, start_time):
            return True

        # Backtrack and try negative
        self.assignment[var] = -1
        if self._dpll(timeout, start_time):
            return True

        # Backtrack
        del self.assignment[var]
        return False

    def _find_unit_clause(self) -> Optional[List[int]]:
        """Find a unit clause (only one unassigned literal)."""
        for clause in self.clauses:
            unassigned = []
            satisfied = False

            for literal in clause:
                var_idx = abs(literal)
                if var_idx in self.assignment:
                    value = self.assignment[var_idx]
                    if (literal > 0 and value == 1) or (literal < 0 and value == -1):
                        satisfied = True
                        break
                else:
                    unassigned.append(literal)

            if not satisfied and len(unassigned) == 1:
                return unassigned

        return None

    def _all_satisfied(self) -> bool:
        """Check if all clauses are satisfied."""
        for clause in self.clauses:
            satisfied = False
            for literal in clause:
                var_idx = abs(literal)
                if var_idx not in self.assignment:
                    satisfied = None  # Unknown
                    break
                value = self.assignment[var_idx]
                if (literal > 0 and value == 1) or (literal < 0 and value == -1):
                    satisfied = True
                    break

            if satisfied is False:
                return False

        return True

    def _has_conflict(self) -> bool:
        """Check if there's an unsatisfiable clause."""
        for clause in self.clauses:
            all_assigned = True
            satisfied = False

            for literal in clause:
                var_idx = abs(literal)
                if var_idx not in self.assignment:
                    all_assigned = False
                    break
                value = self.assignment[var_idx]
                if (literal > 0 and value == 1) or (literal < 0 and value == -1):
                    satisfied = True
                    break

            if all_assigned and not satisfied:
                return True

        return False

    def _choose_variable(self) -> Optional[int]:
        """Choose next unassigned variable."""
        for i in range(1, self.n_vars + 1):
            if i not in self.assignment:
                return i
        return None

    def _to_array(self) -> np.ndarray:
        """Convert assignment dict to array."""
        assignment = np.zeros(self.n_vars)
        for i in range(1, self.n_vars + 1):
            if i in self.assignment:
                assignment[i - 1] = self.assignment[i]
            else:
                assignment[i - 1] = np.random.choice([-1, 1])
        return assignment


def run_benchmark(n_vars: int, num_trials: int = 5, seed_base: int = 42) -> Dict:
    """
    Run benchmark for given problem size.

    Args:
        n_vars: Number of variables
        num_trials: Number of trials per solver
        seed_base: Base random seed

    Returns:
        Dictionary with results for all solvers
    """
    m_clauses = int(4.2 * n_vars)  # Phase transition

    print(f"Problem: {n_vars} variables, {m_clauses} clauses")
    print()

    results = {
        'mobius': [],
        'walksat': [],
        'hybrid': [],
        'dpll': []
    }

    for trial in range(num_trials):
        seed = seed_base + trial
        instance = MobiusSATInstance.random(n_vars, m_clauses, seed=seed)

        print(f"  Trial {trial + 1}/{num_trials}...")

        # Möbius SAT
        try:
            start = time.time()
            result = solve_mobius_sat(instance, seed=seed)
            duration = time.time() - start

            results['mobius'].append(BenchmarkResult(
                solver_name='Möbius SAT',
                satisfaction_ratio=result['satisfaction_ratio'],
                solve_time=duration,
                num_satisfied=result['num_satisfied'],
                total_clauses=result['total_clauses'],
                n_vars=n_vars,
                success=True
            ))
            print(f"    Möbius: {result['satisfaction_ratio']:.4f} ({duration:.3f}s)")
        except Exception as e:
            print(f"    Möbius: FAILED ({e})")

        # WalkSAT
        try:
            solver = WalkSAT(instance)
            start = time.time()
            result = solver.solve(max_flips=10000, seed=seed)
            duration = time.time() - start

            results['walksat'].append(BenchmarkResult(
                solver_name='WalkSAT',
                satisfaction_ratio=result['satisfaction_ratio'],
                solve_time=duration,
                num_satisfied=result['num_satisfied'],
                total_clauses=result['total_clauses'],
                n_vars=n_vars,
                success=True
            ))
            print(f"    WalkSAT: {result['satisfaction_ratio']:.4f} ({duration:.3f}s)")
        except Exception as e:
            print(f"    WalkSAT: FAILED ({e})")

        # Hybrid Möbius+WalkSAT
        try:
            solver = HybridMobiusWalkSAT(instance)
            start = time.time()
            result = solver.solve(
                mobius_strips=18,
                mobius_omega=0.1,
                walksat_max_flips=5000,
                seed=seed
            )
            duration = time.time() - start

            results['hybrid'].append(BenchmarkResult(
                solver_name='Hybrid',
                satisfaction_ratio=result['satisfaction_ratio'],
                solve_time=duration,
                num_satisfied=result['num_satisfied'],
                total_clauses=result['total_clauses'],
                n_vars=n_vars,
                success=True
            ))
            mobius_sat = result.get('mobius_satisfaction', 0.0)
            improvement = result.get('improvement', 0.0)
            print(f"    Hybrid: {result['satisfaction_ratio']:.4f} ({duration:.3f}s) "
                  f"[Mobius: {mobius_sat:.4f}, +{improvement:.4f}]")
        except Exception as e:
            print(f"    Hybrid: FAILED ({e})")

        # DPLL (only for small instances)
        if n_vars <= 50:
            try:
                solver = DPLLSolver(instance)
                start = time.time()
                result = solver.solve(timeout=5.0, seed=seed)
                duration = time.time() - start

                results['dpll'].append(BenchmarkResult(
                    solver_name='DPLL',
                    satisfaction_ratio=result['satisfaction_ratio'],
                    solve_time=duration,
                    num_satisfied=result['num_satisfied'],
                    total_clauses=result['total_clauses'],
                    n_vars=n_vars,
                    success=not result.get('timeout', False)
                ))
                timeout_str = " (timeout)" if result.get('timeout', False) else ""
                print(f"    DPLL: {result['satisfaction_ratio']:.4f} ({duration:.3f}s){timeout_str}")
            except Exception as e:
                print(f"    DPLL: FAILED ({e})")
        else:
            print(f"    DPLL: SKIPPED (too large)")

    print()
    return results


def print_summary(all_results: Dict):
    """Print comprehensive summary statistics."""
    print("="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    print()

    for n_vars in sorted(all_results.keys()):
        results = all_results[n_vars]

        print(f"Problem Size: {n_vars} variables")
        print("-" * 60)

        for solver_name in ['mobius', 'walksat', 'hybrid', 'dpll']:
            if solver_name not in results or len(results[solver_name]) == 0:
                continue

            solver_results = results[solver_name]

            satisfactions = [r.satisfaction_ratio for r in solver_results]
            times = [r.solve_time for r in solver_results]
            successes = [r.success for r in solver_results]

            mean_sat = np.mean(satisfactions)
            std_sat = np.std(satisfactions)
            mean_time = np.mean(times)
            std_time = np.std(times)
            success_rate = np.mean(successes)

            display_name = solver_results[0].solver_name
            print(f"  {display_name:15s}: sat={mean_sat:.4f} +/- {std_sat:.4f}, "
                  f"time={mean_time:.3f}s +/- {std_time:.3f}s, "
                  f"success={success_rate:.1%}")

        print()

    print("="*80)
    print("WINNER ANALYSIS")
    print("="*80)
    print()

    for n_vars in sorted(all_results.keys()):
        results = all_results[n_vars]

        # Compare Möbius vs others
        if 'mobius' in results and len(results['mobius']) > 0:
            mobius_sat = np.mean([r.satisfaction_ratio for r in results['mobius']])

            winners = []

            if 'walksat' in results and len(results['walksat']) > 0:
                walksat_sat = np.mean([r.satisfaction_ratio for r in results['walksat']])
                improvement = ((mobius_sat - walksat_sat) / walksat_sat) * 100
                if mobius_sat > walksat_sat:
                    winners.append(f"WalkSAT by {improvement:+.2f}%")
                elif mobius_sat < walksat_sat:
                    print(f"  n={n_vars}: WalkSAT WINS by {-improvement:+.2f}%")
                    continue

            if 'dpll' in results and len(results['dpll']) > 0:
                dpll_sat = np.mean([r.satisfaction_ratio for r in results['dpll']])
                improvement = ((mobius_sat - dpll_sat) / dpll_sat) * 100
                if mobius_sat > dpll_sat:
                    winners.append(f"DPLL by {improvement:+.2f}%")

            if winners:
                print(f"  n={n_vars}: Möbius WINS over " + ", ".join(winners))
            else:
                print(f"  n={n_vars}: TIE")

    print()


def main():
    """Run comprehensive SAT solver benchmark."""
    print("="*80)
    print("Investigation 12: Möbius SAT Benchmark (+ Hybrid)")
    print("="*80)
    print()
    print("Comparing:")
    print("  1. 18-strip Möbius SAT (Investigation 11)")
    print("  2. WalkSAT (local search)")
    print("  3. Hybrid Möbius+WalkSAT (Möbius -> WalkSAT refinement)")
    print("  4. DPLL (complete search, small instances only)")
    print()
    print("Test: Random 3-SAT at phase transition (m ~ 4.2n)")
    print("="*80)
    print()

    # Test problem sizes
    problem_sizes = [20, 50, 100, 200]
    num_trials = 5

    all_results = {}

    for n_vars in problem_sizes:
        print(f"\n{'='*80}")
        print(f"TESTING: {n_vars} VARIABLES")
        print(f"{'='*80}\n")

        results = run_benchmark(n_vars, num_trials=num_trials)
        all_results[n_vars] = results

    # Print summary
    print_summary(all_results)

    print("="*80)
    print("Investigation 12 Complete")
    print("="*80)

    return 0


if __name__ == '__main__':
    sys.exit(main())
