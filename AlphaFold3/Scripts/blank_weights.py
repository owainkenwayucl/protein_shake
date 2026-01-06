# This script creates an alphafold3 weights shaped object from the real weights, but with all
# the values replaced by 0.0 in the appropriate format. This is useful as a proxy set of weights
# to get the code working without generating real results.

from pathlib import Path
from alphafold3.model import params

import jax.numpy as jnp

path_str = "/root/models"
p = Path(path_str)
k = params.get_model_haiku_params(model_dir=p) # read in weights

for a in k.keys():
	for b in k[a].keys():
		s = k[a][b].shape
		t = k[a][b].dtype
		n = jnp.zeros(s,t)
		k[a][b] = n		
		
with open('zeros.bin', 'wb') as file:
	for a in k.keys():
		for b in k[a].keys():
			arr = k[a][b]
			rec_b = params.encode_record(a, b, arr)
			file.write(rec_b)
