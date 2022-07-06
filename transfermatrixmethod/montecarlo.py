import time

import numpy as np

from transfermatrixmethod import simulate_structure

def run_montecarlo(structures, freq, theta, pol, N_samples, perturbation):
    start = time.time()

    n_freqs = len(freq)
    transmissions = np.zeros((N_samples, n_freqs), dtype=np.complex64)
    reflections = np.zeros((N_samples, n_freqs), dtype=np.complex64)

    for i in range(N_samples):
        reflection, transmission = simulate_structure(
            structures=structures,
            freq=freq,
            theta=theta,
            pol=pol,
            alter=perturbation,
        )
        transmissions[i, :] = transmission
        reflections[i, :] = reflection

    stop = time.time()
    print("seconds elapsed: ", stop - start)

    return transmissions, reflections
