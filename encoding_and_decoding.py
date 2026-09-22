import numpy as np

"""
    Rate 1/2 Convolutional Encoder (Constraint Length K=3)
    Generators:
        G1 = 1 + D + D^2  => Bit ^ State[0] ^ State[1]
        G2 = 1 + D^2      => Bit ^ State[1]
    Input:
        data_with_crc: array(payload + CRC)
    Output:array containing the encoded output bits
    """


def convolutional_encoder_1_2_payload_crc(data_with_crc):
    data_arr = np.asarray(data_with_crc, dtype=int)
    flush_zeros = np.array([0, 0], dtype=int)# Append flush zeros to input array inside the function to bring thje final state of encoder to 00
    full_input = np.concatenate([data_arr, flush_zeros])
    state = [0, 0]  
    encoded_bits = []
    for bit in full_input:
        g1 = bit ^ state[0] ^ state[1]
        g2 = bit ^ state[1]
        encoded_bits.extend([g1, g2])
        state = [bit, state[0]]
    return np.array(encoded_bits)
