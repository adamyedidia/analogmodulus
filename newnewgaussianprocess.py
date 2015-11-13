import numpy as np
import matplotlib.pyplot as p
from math import pi

n = 21

undoctoredWaveform = [np.random.normal(0, 1) for _ in range(n)]

p.plot(range(n), undoctoredWaveform, "b-")

p.savefig("original_wf.png")
p.show()

transformedWaveform = np.fft.fft(undoctoredWaveform)

p.plot(range(n), transformedWaveform, "b-")

p.show()

truncatedTransformedWaveform = \
     [val * (i > pi) for i, val in enumerate(transformedWaveform)]
#    [val * (i > n/2. - pi) * (i < n/2. + pi) for i, val in enumerate(transformedWaveform)]
    
p.plot(range(n), truncatedTransformedWaveform)

p.show()

oversampledWaveform = np.fft.ifft(truncatedTransformedWaveform)

p.plot(range(n), oversampledWaveform)

p.savefig("oversampled_wf.png")
p.show()