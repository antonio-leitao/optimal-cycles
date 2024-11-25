from scipy.fft import fftshift, fftfreq, fft, ifftshift, ifft
from scipy.signal import find_peaks
import numpy as np
import matplotlib.pyplot as plt


##########
#  MAIN
##########


def find_frequency_peaks(fx, x=None, min_height=0.005, distance=10, width=1, wlen=10):
    # Fourier decomposition
    rate = 1 if x is None else np.max(x) / len(x)
    freqs = fftshift(fftfreq(len(fx), d=rate / (2 * np.pi)))
    coefs = fftshift(fft(fx, norm="forward"))
    # Get relevant coefficients
    peaks, _ = find_peaks(
        np.absolute(coefs),
        distance=distance,
        width=width,
        prominence=min_height,
        wlen=wlen,
    )
    mask = np.zeros_like(coefs).astype(bool)
    mask[peaks] = True
    # peaks = peaks[freqs[peaks] != 0] #what is this though?
    return freqs, coefs, mask


def omega_matrix(delay, freqs):
    omega = []
    for d in range(len(freqs)):
        omega.append([np.exp(freq * 1j * delay * d) for freq in freqs])
    return np.array(omega)


def time_delay_embedding(time_series, delay, dimension):
    n = len(time_series)
    num_rows = n - (dimension - 1) * delay
    embedded_array = np.zeros((num_rows, dimension))

    for i in range(dimension):
        start_index = i * delay
        end_index = start_index + num_rows
        embedded_array[:, i] = time_series[start_index:end_index]

    return embedded_array


#########
#  AUX
#########


def reconstruct_signal(coefs, mask):
    # zero out all the unselected coefficients
    new_coefs = np.zeros_like(coefs)
    new_coefs[mask] = coefs[mask]
    # reconstruct
    return ifft(ifftshift(new_coefs), norm="forward")


def gamma_value(delay, freqs):
    value = 0
    for i, fi in enumerate(freqs):
        for j, fj in enumerate(freqs[i:]):
            col_sum = np.sum(
                [np.exp((fi - fj) * 1j * delay * d) for d in range(len(freqs))]
            )
            value += np.square(np.abs(col_sum))
    return value


###########
#  PLOTS
###########
def plot_frequency_peaks(freqs, coefs, mask, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(freqs, np.abs(coefs))
    # ax.axhline(min_height, linestyle="--", color="k", alpha=0.3)
    ax.set_xticks(np.round(freqs[mask], 2))
    for peak in freqs[mask]:
        ax.axvline(peak, color="r", alpha=0.3)
    ax.spines[["right", "top"]].set_visible(False)
