import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import welch


def calculate_ber(tx_bits, rx_bits):
    """Calculate bit error rate by comparing transmitted and received bits."""
    tx_bits = np.asarray(tx_bits).astype(int)
    rx_bits = np.asarray(rx_bits).astype(int)

    if len(tx_bits) != len(rx_bits):
        raise ValueError("tx_bits and rx_bits must have the same length.")

    errors = np.sum(tx_bits != rx_bits)
    return errors / len(tx_bits)


def theoretical_bpsk_ber(eb_n0_db):
    """Theoretical BER of coherent BPSK over AWGN."""
    eb_n0_linear = 10 ** (np.asarray(eb_n0_db) / 10)
    return 0.5 * erfc(np.sqrt(eb_n0_linear))


def calculate_snr_db(signal, noise):
    """Estimate signal-to-noise ratio from signal and noise arrays."""
    signal_power = np.mean(np.abs(signal) ** 2)
    noise_power = np.mean(np.abs(noise) ** 2)

    if noise_power == 0:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)


def plot_ber_vs_ebn0(eb_n0_db, ideal_ber, synchronized_ber,
                     unsynchronized_ber, show_theory=True):
    """Plot the three receiver BER curves and the theoretical BPSK curve."""
    eb_n0_db = np.asarray(eb_n0_db)

    plt.figure(figsize=(9, 6))

    if show_theory:
        theory = theoretical_bpsk_ber(eb_n0_db)
        plt.semilogy(
            eb_n0_db, theory,
            "k--", linewidth=2,
            label="Theoretical BPSK"
        )

    plt.semilogy(
        eb_n0_db, ideal_ber,
        "o-", label="Ideal receiver"
    )
    plt.semilogy(
        eb_n0_db, synchronized_ber,
        "s-", label="Synchronized receiver"
    )
    plt.semilogy(
        eb_n0_db, unsynchronized_ber,
        "^-", label="Unsynchronized receiver"
    )

    plt.xlabel(r"$E_b/N_0$ (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("BPSK BER vs $E_b/N_0$")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_constellation(symbols, title="BPSK Constellation"):
    """Plot received complex symbol samples in the I-Q plane."""
    symbols = np.asarray(symbols)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        np.real(symbols),
        np.imag(symbols),
        s=10,
        alpha=0.5
    )

    plt.axhline(0, linewidth=0.8)
    plt.axvline(0, linewidth=0.8)

    plt.xlabel("In-phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()


def plot_eye_diagram(samples, samples_per_symbol, num_traces=100,
                     title="Eye Diagram"):
    """Overlay consecutive symbol periods to form an eye diagram."""
    samples = np.asarray(samples)

    if samples_per_symbol < 2:
        raise ValueError("samples_per_symbol must be at least 2.")

    trace_length = 2 * samples_per_symbol
    max_traces = (len(samples) - trace_length) // samples_per_symbol

    num_traces = min(num_traces, max_traces)

    if num_traces <= 0:
        raise ValueError("Not enough samples to make an eye diagram.")

    plt.figure(figsize=(9, 6))

    time = np.arange(trace_length) / samples_per_symbol

    for i in range(num_traces):
        start = i * samples_per_symbol
        trace = samples[start:start + trace_length]
        plt.plot(time, np.real(trace), alpha=0.25)

    plt.axvline(1, linestyle="--", linewidth=1,
                label="Nominal sampling point")

    plt.xlabel("Time (symbols)")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_psd(signal, sample_rate, title="Power Spectral Density"):
    """Plot the estimated power spectral density using Welch's method."""
    signal = np.asarray(signal)

    frequencies, power = welch(
        signal,
        fs=sample_rate,
        return_onesided=False
    )

    frequencies = np.fft.fftshift(frequencies)
    power = np.fft.fftshift(power)

    plt.figure(figsize=(9, 6))

    plt.plot(frequencies, 10 * np.log10(power + 1e-12))

    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power/Frequency (dB)")
    plt.title(title)
    plt.grid(True, linestyle=":")
    plt.tight_layout()
    plt.show()
