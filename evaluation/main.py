import numpy as np

from evaluation import (
    calculate_ber,
    calculate_snr_db,
    plot_ber_vs_ebn0,
    plot_constellation,
    plot_eye_diagram,
    plot_psd,
    theoretical_bpsk_ber,
)

from test_signal import (
    generate_bpsk_signal,
    make_received_symbols,
    make_received_bits,
)


def main():
    # ---------------------------------------------------------
    # 1. BER vs Eb/N0
    # ---------------------------------------------------------
    #
    # These are temporary BER values.
    # They are deliberately generated for testing the plotting
    # system. Later, replace them with the actual receiver results.
    #
    eb_n0_db = np.arange(0, 11, 1)

    ideal_ber = []
    synchronized_ber = []
    unsynchronized_ber = []

    for i, eb_n0 in enumerate(eb_n0_db):
        data = generate_bpsk_signal(
            num_bits=10000,
            eb_n0_db=eb_n0,
            seed=100 + i
        )

        tx_bits = data["bits"]

        # Temporary values representing three receiver cases.
        #
        # In the final project, these will come directly from
        # the actual receiver.
        theory = theoretical_bpsk_ber(eb_n0)

        # We keep a small floor here so the log plot stays visible.
        ideal_target = max(theory, 1e-5)

        # Dummy synchronization penalty.
        sync_target = max(theory * 1.5, 1e-5)

        # Dummy unsynchronized error floor.
        unsync_target = max(theory * 4 + 0.015, 1e-4)

        ideal_bits = make_received_bits(
            tx_bits, ideal_target, seed=200 + i
        )
        sync_bits = make_received_bits(
            tx_bits, sync_target, seed=300 + i
        )
        unsync_bits = make_received_bits(
            tx_bits, unsync_target, seed=400 + i
        )

        ideal_ber.append(calculate_ber(tx_bits, ideal_bits))
        synchronized_ber.append(calculate_ber(tx_bits, sync_bits))
        unsynchronized_ber.append(calculate_ber(tx_bits, unsync_bits))

    print("BER results:")
    print("Eb/N0(dB)   Ideal       Synchronized   Unsynchronized")

    for i, eb_n0 in enumerate(eb_n0_db):
        print(
            f"{eb_n0:>3}        "
            f"{ideal_ber[i]:.6f}     "
            f"{synchronized_ber[i]:.6f}        "
            f"{unsynchronized_ber[i]:.6f}"
        )

    plot_ber_vs_ebn0(
        eb_n0_db,
        ideal_ber,
        synchronized_ber,
        unsynchronized_ber
    )

    # ---------------------------------------------------------
    # 2. Constellation diagram
    # ---------------------------------------------------------

    bits = np.random.default_rng(123).integers(0, 2, 2000)

    received_symbols = make_received_symbols(
        bits,
        eb_n0_db=6,
        phase_offset=np.deg2rad(20)
    )

    plot_constellation(
        received_symbols,
        title="Dummy BPSK Constellation - Before Synchronization"
    )

    # ---------------------------------------------------------
    # 3. Eye diagram
    # ---------------------------------------------------------

    eye_data = generate_bpsk_signal(
        num_bits=2000,
        eb_n0_db=8,
        samples_per_symbol=8,
        seed=123
    )

    plot_eye_diagram(
        eye_data["rx_signal"],
        eye_data["samples_per_symbol"],
        num_traces=100,
        title="Dummy BPSK Eye Diagram"
    )

    # ---------------------------------------------------------
    # 4. PSD
    # ---------------------------------------------------------

    plot_psd(
        eye_data["rx_signal"],
        eye_data["sample_rate"],
        title="Dummy BPSK Received Signal PSD"
    )

    # ---------------------------------------------------------
    # 5. Example measured SNR
    # ---------------------------------------------------------

    snr = calculate_snr_db(
        eye_data["tx_signal"],
        eye_data["noise"]
    )

    print(f"\nExample measured signal-to-noise ratio: {snr:.2f} dB")


if __name__ == "__main__":
    main()
