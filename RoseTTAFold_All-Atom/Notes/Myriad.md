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

--- Update 22:04 on Friday 28th of February in the year of our Lord Two Thousand and Twenty Five.

I've got an interactive job running on a GPU node so re-running steps 5. onwards (not setting that `CONDA_OVERRIDE_CUDA` variable since we have a GPU in this box).

Ok interesting, it's still doing:

```
/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA/lib/python3.10/site-packages/signalp/model_weights
Converting to CPU.
Converting distilled_model_signalp6.pt .
```

FWIW:

```
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > nvidia-smi
Fri Feb 28 23:08:28 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:58:00.0 Off |                    0 |
| N/A   35C    P0             37W /  250W |       1MiB /  32768MiB |      1%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```

That's done.

As required in the install instructions: 

```
mv $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/distilled_model_signalp6.pt $CONDA_PREFIX/lib/python3.10/site-packages/signalp/model_weights/ensemble_model_signalp6.pt
```

Learned my lesson: do not question.

And then of course I question - the last install step says to "run `bash install_dependencies.sh`". That makes me uncomfy. What does the script do?

```
#!/bin/bash
# From: https://github.com/RosettaCommons/RoseTTAFold

# install external program not supported by conda installation
case "$(uname -s)" in
    Linux*)     platform=linux;;
    Darwin*)    platform=macosx;;
    *)          echo "unsupported OS type. exiting"; exit 1
esac
echo "Installing dependencies for ${platform}..."

# the cs-blast platform descriptoin includes the width of memory addresses
# we expect a 64-bit operating system
if [[ ${platform} == "linux" ]]; then
    platform=${platform}64
fi

# download cs-blast
echo "Downloading cs-blast ..."
wget http://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_${platform}.tar.gz -O csblast-2.2.3.tar.gz
mkdir -p csblast-2.2.3
tar xf csblast-2.2.3.tar.gz -C csblast-2.2.3 --strip-components=1
```

Right, so basically it works out if you are on a Mac or on Linux and then unpacks that the appropriate version of `csblast-2.2.3.tar.gz`

We can not like this for various reasons - we are downloading a random binary and running it, but to add to that, it's over an http link.

Note if we modify that link to https it works fine:

```
wget https://wwwuser.gwdg.de/~compbiol/data/csblast/releases/csblast-2.2.3_linux64.tar.gz
```

This makes me generally unhappy.

It appears csblast is this: https://en.wikipedia.org/wiki/CS-BLAST

That Wiki page refers both to the https://wwwuser.gwdguser.de/~compbiol/data/csblast/releases/ URL which is similar but crucially not the same as the above and a GitHub Repo https://github.com/soedinglab/csblast. On investigation, https://wwwuser.gwdg.de/ redirects to https://wwwuser.gwdguser.de/ which is fine and in no way upsetting at 22:36 on a Friday.

Interestingly, this is itself a fork of https://github.com/cangermueller/csblast, which is ahead by three commits which are updates to the readme to say it depends on some godawful Google software.

So on a whim, I looked and csblast is in bioconda, which is a channel they use in the original mamba setup. FFS

```
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > mamba search bioconda::csblast
Loading channels: done
# Name                       Version           Build  Channel             
csblast                        2.2.3      h4ac6f70_3  bioconda            
csblast                        2.2.3      h7d875b9_0  bioconda            
csblast                        2.2.3      h9948957_4  bioconda            
csblast                        2.2.3      h9f5acd7_1  bioconda            
csblast                        2.2.3      h9f5acd7_2  bioconda            
```

So I did

```
mamba install bioconda::csblast
```

This slightly upgraded some things (e.g. certificates and OpenSSL which seem like a good idea!)

Looking at the `.tar.gz`, to replicate their install instructions, we want to:

```
mkdir -p csblast-2.2.3/bin
mkdir -p csblast-2.2.3/data
ln -s ${CONDA_PREFIX}/data/K4000.crf csblast-2.2.3/data
ln -s ${CONDA_PREFIX}/data/K4000.lib csblast-2.2.3/data
ln -s ${CONDA_PREFIX}/bin/csblast csblast-2.2.3/bin
ln -s ${CONDA_PREFIX}/bin/csbuild csblast-2.2.3/bin
```

Inside the RoseTTAFold-All-Atom directory.

```
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > find csblast-2.2.3/
csblast-2.2.3/
csblast-2.2.3/data
csblast-2.2.3/data/K4000.crf
csblast-2.2.3/data/K4000.lib
csblast-2.2.3/bin
csblast-2.2.3/bin/csblast
csblast-2.2.3/bin/csbuild
```

Sweet!