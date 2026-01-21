from alphafold3.model import params
import numpy as np
import zstandard
import processor

parameters = processor.process("parameters.txt")

with zstandard.open('random_weights.bin.zst', 'wb') as compressed:
  for item in parameters:
    scope_name = item["name"]
    shape = item["shape"]
    dtype = item["dtype"]
    if scope_name == '__meta__:__identifier__':
      # The identifier can be all zeros.
      arr = np.zeros(shape=shape, dtype=dtype)
    else:
      # Do not use all-zero params, instead sample uniformly between -1 and 1.
      arr = np.random.uniform(low=-1, high=1, size=shape).astype(dtype)
    scope_name = scope_name.split(':')
    compressed.write(params.encode_record(*scope_name, arr))
