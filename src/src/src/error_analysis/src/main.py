import numpy as np
from problems.test_problem import f, exact_solution
from solvers.rk2 import rk2_step
from solvers.rk3 import rk3_step
from solvers.rk4 import rk4_step
from solver import solve_ode
from error_analysis.global_error import compute_global_error
from error_analysis.convergence_rate import compute_convergence_rate
from error_analysis.local_error import local_error_analysis
from visualization.convergence_plot import plot_convergence
from visualization.solution_plot import plot_solutions

x_range = (0, 1)
y0 = 1.0
h_values = [0.5, 0.2, 0.1, 0.05, 0.025, 0.01]

methods = {
    "RK2": rk2_step,
    "RK3": rk3_step,
    "RK4": rk4_step
}

errors = {}

for name, method in methods.items():
    errors[name] = compute_global_error(
        method, f, exact_solution, x_range, y0, h_values
    )

rates = compute_convergence_rate(errors, h_values)

print("\nEmpirical Convergence Rates:")
for k, v in rates.items():
    print(f"{k}: {v:.3f}")

plot_convergence(errors, h_values)
plot_solutions(methods, f, exact_solution, x_range, y0, h=0.2)

# Local error (appendix)
local_errors = local_error_analysis(f, exact_solution)
print("\nLocal Truncation Errors (h=0.1):")
for k, v in local_errors.items():
    print(f"{k}: {v:.2e}")
