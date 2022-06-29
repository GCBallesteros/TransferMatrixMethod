import numpy as np


def _T_interface(ni, nj, cos_i, cos_j, pol):
    T_interface = np.zeros((2, 2, len(ni)), dtype=np.complex64)
    T_interface[0, 0, :] = 1
    T_interface[1, 1, :] = 1
    if pol == "s":
        rij = (ni * cos_i - nj * cos_j) / (ni * cos_i + nj * cos_j)
        tij = 1 + rij
    elif pol == "p":
        rij = (nj * cos_i - ni * cos_j) / (nj * cos_i + ni * cos_j)
        tij = (1 + rij) * (ni / nj)

    T_interface[0, 1] = rij
    T_interface[1, 0] = rij
    T_interface = T_interface / tij
    return T_interface


def _T_propagation(phase):
    prop_matrix = np.zeros((2, 2, len(phase)), dtype=np.complex64)
    prop_matrix[0, 0, :] = np.exp(-1j * phase)
    prop_matrix[1, 1, :] = np.exp(1j * phase)

    return prop_matrix


def multilayer_response(pol, t_layers, index_layers, freq, theta_in):
    """Response of multilayer structure to normal incident plane wave.

    Wave impinges from the left.

    Arguments:
      pol (str): 's' or 'p'
      t_layers: Thickness of layers including exterior infinite media.
                The exterior layer thickness is irrelevant.
      index_layers: Index of refraction of all layers. Includes exterior
                    semiinfinite media. Besides a number it can also be
                    supplied as a function that computes the index as
                    a function of 1/lambda.
      freq: Frequency in Meep units, i.e. f = 1/lambda
      theta_in: Angle of incidence

    Returns:
      Tuple (R, T) with the power Transmitted and Reflected.
    """
    n_layers = len(t_layers)
    if n_layers != len(index_layers):
        print("Number of t_layers and index_layers is different!")
        raise

    # Exterior layers correspond to infinite media. So we want
    # the propagation matrix to be the identity
    t_layers[0] = 0
    t_layers[-1] = 0

    # We also admit functions that compute the index as a function of frequency
    # so we must check if we have a function or just a number and then generate
    # the indices of refraction for each wavelength accordingly
    for layer_idx, index in enumerate(index_layers):
        if callable(index):
            index_layers[layer_idx] = index(freq).astype(np.complex64)
        else:
            index_layers[layer_idx] = index * np.ones(len(freq), dtype=np.complex64)

    index_layers = np.asarray(index_layers)
    t_layers = np.asarray(t_layers)

    # Compute cos(theta_i) which need for Fresnel Coeff
    # and interface T matrix.
    sin_thetas = (index_layers[0] / index_layers) * np.sin(theta_in)
    cos_thetas = np.sqrt(1 - sin_thetas ** 2)

    # Power factor to convert from t coefficient to transmittance
    power_factor = (index_layers[-1] / index_layers[0]) * (
        cos_thetas[-1] / cos_thetas[0]
    )

    # Create total transfer matrix for all frequencies
    T = np.zeros((2, 2, len(freq)), dtype=np.complex64)
    T[0, 0, :] = 1
    T[1, 1, :] = 1

    for ii in np.arange(0, n_layers - 1):
        # Compute propagation matrix
        phase_prop = 2 * np.pi * freq * t_layers[ii] * index_layers[ii] * cos_thetas[ii]
        T_prop = _T_propagation(phase_prop)

        # Compute interface matrix
        n1 = index_layers[ii]
        n2 = index_layers[ii + 1]
        cos_1 = cos_thetas[ii]
        cos_2 = cos_thetas[ii + 1]
        T_ij = _T_interface(n1, n2, cos_1, cos_2, pol)

        # Update system transfer matrix
        # We want matrix products of matrices at the same frequencies but not cross
        # information across frequencies hence the complicated einsums
        T = np.einsum("ijk,jhk->ihk", T, np.einsum("ijk,jhk->ihk", T_prop, T_ij))

    R = np.abs(T[0, 1] / T[0, 0]) ** 2
    T = power_factor * np.abs(1.0 / T[0, 0]) ** 2

    return (R, T)


if __name__ == "__main__":
    # Silicon Dioxide on Silicon Wafer Example
    import matplotlib

    matplotlib.rcParams["backend"] = "TkAgg"
    import matplotlib.pyplot as plt

    # Completely made up functions for wavelength dependence of index of
    # refraction
    def n_Si(wavelength):
        return 3.5 + wavelength * 0.01

    def n_SiO2(wavelength):
        return 1.5-0.1j + wavelength * 0.01

    lambda_min = 400
    lambda_max = 1000
    wavelengths = np.linspace(lambda_min, lambda_max, 400)
    freq = 1.0 / wavelengths[::-1]

    index_layers = [1, n_SiO2, n_Si]
    t_layers = [0, 220, 0]

    (theory_reflection, theory_transmission) = multilayer_response(
        "s", t_layers, index_layers, freq, 0 * np.pi / 180
    )

    plt.figure()
    plt.plot(1 / freq, 10 * np.log10(theory_reflection), label="Reflection")
    plt.plot(1 / freq, 10 * np.log10(theory_transmission), label="Tranmission")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.show()
