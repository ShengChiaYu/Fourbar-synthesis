import numpy as np
def Batchsampler(samples, batch_size=1):
    remainder = samples.shape[0] % batch_size
    size_wo_rem = samples.shape[0] - remainder
    if remainder == 0:
        batches = np.split(samples, size_wo_rem/batch_size)
    else:
        batches = samples[0:size_wo_rem]
        batches = np.split(batches, size_wo_rem/batch_size)
    return batches

samples = np.arange(14)
print(samples)
batches = Batchsampler(samples, batch_size=3)
print(batches)
