import numpy as np

#Starting the project with 100 bits
#seed 122807528840384100672342137672332424406
rng = np.random.default_rng(122807528840384100672342137672332424406)
bitstream = rng.integers(low=0, high=1,size=100,endpoint=True)
#BPSK mapping
for i in range(len(bitstream)):
    bitstream[i] = 1 if bitstream[i] == 0 else -1
print(bitstream)