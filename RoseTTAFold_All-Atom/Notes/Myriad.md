# Notes on installing on Myriad

## Attempt one - personal, home dir.

Repo at: https://github.com/baker-laboratory/RoseTTAFold-All-Atom

1. Downloading various data files to `ACFS/Datasets/RoseTTaFold_All-Atom/`

```
wget https://files.ipd.uw.edu/pub/RF-All-Atom/weights/RFAA_paper_weights.pt
wget https://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
```

2. I have registered for signalp-6.0 fast and downloaded it to Myriad.

3. Trying to decode the very confusing "install mamba" instructions.

Downloaded `Miniforge3-24.11.3-0-Linux-x86_64.sh`
Checksum matches that in AUR: https://aur.archlinux.org/cgit/aur.git/tree/PKGBUILD?h=miniforge

```
2e1ad2188fe69fcdd522c2b20c08c800a5c7411b775eca768318b1540ed32e53  Miniforge3-24.11.3-0-Linux-x86_64.sh
```

Installing with:

```
bash Miniforge3-24.11.3-0-Linux-x86_64.sh -b -p /home/uccaoke/Scratch/miniforge3
```

To activate:

```
source /home/uccaoke/Scratch/miniforge3/etc/profile.d/conda.sh
source /home/uccaoke/Scratch/miniforge3/etc/profile.d/mamba.sh
```

4. Cloned Repo

5. `CONDA_OVERRIDE_CUDA="11.8" mamba env create -f environment.yaml`

The `CONDA_OVERRIDE_CUDA="11.8"` is required because the login node does not have a GPU so mamba fails to dependency checks for Tensorflow.

6. Activate env and build package

```
mamba activate RFAA

cd rf2aa/SE3Transformer/
pip3 install --no-cache-dir -r requirements.txt
python3 setup.py install
cd ../../
```

7. signalp6 - I've registered for the software and downloaded the repo. Now I need to understand the install instructions which make no sense.

Right - so the tarball is a copy of a python package. *Its* instructions say to install it to our menv with `pip`

```
pip install signalp-6-package/  
```

Note that this appears to install a different version of the Cuda run-time into the menv which is super interesting and will cause no problems later, as well as downgrading torch to 1.13.1

This seems very bad so I think I may have to start again and have a different (venv?) for signalp?

https://github.com/baker-laboratory/RoseTTAFold-All-Atom/issues/164

I think this may be contradictory instructions, as signalp is included in the `environment.yaml`??

OK, I think what they want is the *weights* from the downloaded package added to the version of the package in the menv.

No, despite being in the `environment.yaml` it's not installed!?!?

Well it's not according to `pip` and it's not in site-packages but `conda list` thinks its installed.

Uh...

```
find . | grep signalp6
./conda-meta/signalp6-6.0g-1.json
./share/signalp6-6.0g-1
./share/signalp6-6.0g-1/test_set.fasta
./share/signalp6-6.0g-1/unregister.sh
./share/signalp6-6.0g-1/placeholder.sh
./share/signalp6-6.0g-1/register.sh
./share/signalp6-6.0g-1/signalp6
./share/signalp6-6.0g-1/signalp
./bin/signalp6-register
./bin/.signalp6-pre-unlink.sh
./bin/signalp6
./bin/.signalp6-post-link.sh
```

Re-reading the instructions, they mean *literally* run

```
signalp6-register signalp-6.0h.fast.tar.gz
```

I do not like this output.

Presumably I have to build on a GPU node...

```
/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA/lib/python3.10/site-packages/signalp/model_weights
Converting to CPU.
Converting distilled_model_signalp6.pt .
```