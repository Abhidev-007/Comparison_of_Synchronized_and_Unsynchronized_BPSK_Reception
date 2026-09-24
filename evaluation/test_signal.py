import numpy as np
from scipy.signal import upfirdn


def rrc_filter(beta, span, samples_per_symbol):
    """Create a Root Raised Cosine filter."""
    n = span * samples_per_symbol
    t = np.arange(-n / 2, n / 2 + 1) / samples_per_symbol

    h = np.zeros_like(t, dtype=float)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1 + beta * (4 / np.pi - 1)

        elif beta > 0 and abs(abs(4 * beta * ti) - 1) < 1e-12:
            h[i] = (
                beta / np.sqrt(2)
                * (
                    (1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                    + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))
                )
            )

        else:
            numerator = (
                np.sin(np.pi * ti * (1 - beta))
                + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))
            )
            denominator = np.pi * ti * (1 - (4 * beta * ti) ** 2)

            h[i] = numerator / denominator

    h /= np.sqrt(np.sum(h ** 2))
    return h


def generate_bpsk_signal(
    num_bits=10000,
    eb_n0_db=6,
    samples_per_symbol=8,
    rolloff=0.35,
    span=8,
    phase_offset=0.0,
    frequency_offset=0.0,
    timing_offset=0.0,
    seed=42
):
    """
    Generate dummy BPSK data for testing the evaluation code.

    The generated signal is NOT the project's final receiver.
    It is only a convenient stand-in until the real receiver is connected.
    """
    rng = np.random.default_rng(seed)

    # Generate random 0/1 data.
    bits = rng.integers(0, 2, num_bits)

    # BPSK mapping: 0 -> -1, 1 -> +1.
    symbols = 2 * bits - 1

    # Pulse-shape the symbols using an RRC filter.
    rrc = rrc_filter(rolloff, span, samples_per_symbol)
    shaped = upfirdn(
        rrc,
        symbols.astype(float),
        up=samples_per_symbol
    )

    # Normalize the signal so its average symbol energy is close to 1.
    shaped /= np.sqrt(np.mean(shaped ** 2))

    # Add a simple fractional timing shift.
    if timing_offset != 0:
        sample_positions = np.arange(len(shaped))
        shifted_positions = sample_positions - timing_offset
        shaped = np.interp(
            shifted_positions,
            sample_positions,
            shaped,
            left=0,
            right=0
        )

    # Apply carrier phase and frequency offsets.
    sample_rate = samples_per_symbol
    time = np.arange(len(shaped)) / sample_rate

    carrier_error = np.exp(
        1j * (2 * np.pi * frequency_offset * time + phase_offset)
    )

    impaired_signal = shaped * carrier_error

    # Add complex AWGN.
    eb_n0_linear = 10 ** (eb_n0_db / 10)
    noise_variance = 1 / (2 * eb_n0_linear)

    noise = np.sqrt(noise_variance) * (
        rng.standard_normal(len(impaired_signal))
        + 1j * rng.standard_normal(len(impaired_signal))
    )

    received = impaired_signal + noise

    return {
        "bits": bits,
        "symbols": symbols,
        "tx_signal": shaped,
        "rx_signal": received,
        "noise": noise,
        "sample_rate": sample_rate,
        "samples_per_symbol": samples_per_symbol,
    }


def make_received_symbols(bits, eb_n0_db, phase_offset=0.0,
                          seed=42):
    """
    Create noisy symbol-rate BPSK samples.

    This is useful for quickly testing constellation plots.
    """
    rng = np.random.default_rng(seed)

    symbols = 2 * bits - 1
    phase = np.exp(1j * phase_offset)

    eb_n0_linear = 10 ** (eb_n0_db / 10)
    noise_std = np.sqrt(1 / (2 * eb_n0_linear))

    noise = noise_std * (
        rng.standard_normal(len(symbols))
        + 1j * rng.standard_normal(len(symbols))
    )

    return symbols * phase + noise


def make_received_bits(bits, ber, seed=42):
    """
    Make dummy receiver decisions with a chosen BER.

    This is ONLY for testing the BER plot.
    """
    rng = np.random.default_rng(seed)

    received = bits.copy()
    num_errors = int(round(len(bits) * ber))

    if num_errors > 0:
        error_indices = rng.choice(
            len(bits),
            size=num_errors,
            replace=False
        )
        received[error_indices] ^= 1

    return received
