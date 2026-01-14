import numpy as np

def compute_convergence_rate(errors, h_values):
    rates = {}

    for method, error_list in errors.items():
        local_rates = []
        for i in range(len(error_list) - 1):
            rate = np.log(error_list[i+1] / error_list[i]) / \
                   np.log(h_values[i+1] / h_values[i])
            local_rates.append(rate)

        rates[method] = np.mean(local_rates)

    return rates
