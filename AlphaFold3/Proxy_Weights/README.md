# Proxy Weights generation

The code in this directory allows you to generate proxy weights which DO NOT PRODUCE SCIENTIFIC RESULTS for the purposes of benchmarking/porting the AlphaFold3 code.

Everything in this directory is based on [model_parameters.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/model_parameters.md) and [issue #594](https://github.com/google-deepmind/alphafold3/issues/594) and is therefore covered under the license of the AlphaFold3 code aka Attribution-NonCommercial-ShareAlike 4.0 International (see [LICENSE](LICENSE))

To generate proxy weights, run `generate.py` *inside* a valid AlphaFold3 container. It should produce a file called `random_weights.bin.zst` which will successfully exercise the AlphaFold3 code without doing useful science.

The contents of `generate.py` and `parameters.txt` are adapted from [model_parameters.md](https://github.com/google-deepmind/alphafold3/blob/main/docs/model_parameters.md) and `processor.py` is written based on the structure of `parameters.txt`.

Thanks to the AlphaFold3 team for making the model structure and the code in `generate.py` available.
