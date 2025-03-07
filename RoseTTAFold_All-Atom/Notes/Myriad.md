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

So on a whim, I looked and csblast is in bioconda, which is a channel they use in the original mamba setup.

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

It's 23:11, time to take my dried frog pills and go to bed.

--- Update 15:06 on Sunday 2nd of March in the year of our Lord Two Thousand and Twenty Five.

I'm seriously struggling with disk space challenges - between 1TiB of ACFS and 1TiB of Scratch I do not have enough space to have the databases for both applications, on the basis that RoseTTAFold All-Atom is the most likely to work I've ditched the databases for AlphaFold3 for now while I try to unpack the RoseTTAFold All-Atom.

BLAST Legacy - while the various files are decompressing, the final "code bit" that's needed is BLAST legacy. As with CS-BLAST this is actually in bioconda:

```
(RFAA) Myriad [login12] ~ :) > mamba search bioconda::blast-legacy
Loading channels: done
# Name                       Version           Build  Channel             
blast-legacy                  2.2.19               0  bioconda            
blast-legacy                  2.2.22               0  bioconda            
blast-legacy                  2.2.22               1  bioconda            
blast-legacy                  2.2.26               0  bioconda            
blast-legacy                  2.2.26               1  bioconda            
blast-legacy                  2.2.26               2  bioconda            
blast-legacy                  2.2.26      h9ee0642_3  bioconda            
blast-legacy                  2.2.26      h9ee0642_4  bioconda            
blast-legacy                  2.2.26      hf7ff83a_5  bioconda         
```

So let's try and do the same trick.

Interestingly, it looks like it's already installed. Go figure.

Very unhelpfully, the URL given, `https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz` for the package is wrong.

```
(RFAA) Myriad [login12] temp :) > wget https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
--2025-03-02 15:19:32--  https://ftp.ncbi.nlm.nih.gov/blast/executables/legacy.NOTSUPPORTED/2.2.26/blast-2.2.26-x64-linux.tar.gz
Resolving ftp.ncbi.nlm.nih.gov (ftp.ncbi.nlm.nih.gov)... failed: Name or service not known.
wget: unable to resolve host address ‘ftp.ncbi.nlm.nih.gov’
```

We do however have this installed in Myriad anyway, so to save sanity and space:

```
ln -s /shared/ucl/apps/blast/blast-2.2.26 .
```

--- Update 18:36 on Sunday 2nd of March in the year of our Lord Two Thousand and Twenty Five.

`bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz` is still unpacking and I'm about to run out of space. I clearly need more thant 1 TiB quota.

Looking at the output of `tar -vtf`, one file is over 1.5 TiB.

--- Update 10:37 on Tuesday 4th of March in the year of our Lord Two Thousand and Twenty Five.

Heather increased my quota in Scratch temporarily to 6TiB and I unpacked the RoseTTAFold databases to it.

I've reserved a GPU node on Myriad and tried running it and

```
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > python3 -m rf2aa.run_inference --config-name protein.yaml
/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'protein.yaml': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
Using the cif atom ordering for TRP.
./make_msa.sh examples/protein/7u7w_A.fasta 7u7w_protein/A 4 64  pdb100_2021Mar03/pdb100_2021Mar03
Predicting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:03<00:00,  3.87s/sequences]
Running HHblits against UniRef30 with E-value cutoff 1e-10

```

Looking at `nvidia-smi` something is happening:

```

Tue Mar  4 10:40:29 2025       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:58:00.0 Off |                    0 |
| N/A   36C    P0             37W /  250W |    1898MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-PCIE-32GB           Off |   00000000:D8:00.0 Off |                    0 |
| N/A   32C    P0             26W /  250W |       4MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A    249383      C   python3                                      1894MiB |
+-----------------------------------------------------------------------------------------+

```

Which bodes well.

There are definitely some problems with centralising the install though. For example, the `protein.yaml` is relative to `rf2aa/config/inference/` which is quite annoying.

Also, this is a multi-stage pipeline so let's see what happens.

As things continue: 

```
Running HHblits against UniRef30 with E-value cutoff 1e-10
- 10:50:06.264 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 10:50:06.264 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov75.a3m

- 10:50:06.493 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 10:50:39.946 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 10:50:39.947 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov50.a3m

- 10:50:40.172 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

Running PSIPRED
Running hhsearch
cat: 7u7w_protein/A/t000_.ss2: No such file or directory

```

Which looks not good, worse, the pipeline continues?

Perhaps it's not necessary as it's pumped out a PDB and a Pytorch file as required:

```
7u7w_protein         7u7w_protein.pdb  blast-2.2.26   environment.yaml  img          input_prep               LICENSE      outputs           README.md  RFAA_paper_weights.pt
7u7w_protein_aux.pt  bfd               csblast-2.2.3  examples          __init__.py  install_dependencies.sh  make_msa.sh  pdb100_2021Mar03  rf2aa      UniRef30_2020_06
```

Ah, see this issue: https://github.com/baker-laboratory/RoseTTAFold-All-Atom/issues/106

The fix (`chmod +x input_prep/make_ss.sh`) in that issue doesn't make a lot of sense but let's see if it works.

It did indeed work:

```
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > python3 -m rf2aa.run_inference --config-name protein
/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'protein': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
Using the cif atom ordering for TRP.
./make_msa.sh examples/protein/7u7w_A.fasta 7u7w_protein/A 4 64  pdb100_2021Mar03/pdb100_2021Mar03
Predicting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.32sequences/s]
Running HHblits against UniRef30 with E-value cutoff 1e-10
- 11:24:25.339 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 11:24:25.339 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov75.a3m

- 11:24:25.568 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 11:24:59.027 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 11:24:59.027 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov50.a3m

- 11:24:59.258 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

Running PSIPRED
Running hhsearch
(RFAA) Myriad [node-e96a-001] RoseTTAFold-All-Atom :) > 
```


--- Update 15:40 on Thursday 6th of March

(note to James, I'm waiting for a long ansible deploy of the trino dependencies to happen)

Over the last few days I've been copying stiff as `ccspapp` over so that the RosettaFold All-Atom databases are visible to all.

The first thing I want to do is "un-mamba" my install which will make life easier doing a central install. Looking at the environment, the only meaningful variable `mamba activate` sets is `$PATH`.

Doing a run to prove this is fine.

```
- 17:20:46.810 ERROR: In /opt/conda/conda-bld/hhsuite_1709621322429/work/src/hhalignment.cpp:223: Read:

- 17:20:46.810 ERROR:   sequence ss_pred contains no residues.

Error executing job with overrides: []
Traceback (most recent call last):
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 206, in main
    runner.infer()
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 153, in infer
    self.parse_inference_config()
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 46, in parse_inference_config
    protein_input = generate_msa_and_load_protein(
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 93, in generate_msa_and_load_protein
    return load_protein(str(msa_file), str(hhr_file), str(atab_file), model_runner)
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 66, in load_protein
    xyz_t, t1d, mask_t, _ = get_templates(
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 30, in get_templates
    ) = parse_templates_raw(ffdb, hhr_fn=hhr_fn, atab_fn=atab_fn)
  File "/lustre/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/parsers.py", line 628, in parse_templates_raw
    for l in open(atab_fn, "r").readlines():
FileNotFoundError: [Errno 2] No such file or directory: '7u7w_protein/A/t000_.atab'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.

```

So not fine. No idea why this happening. It makes no sense and it's definitely due to the difference between mamba activating the RFAA environment and just setting the `$PATH`. Will retrun to to this later.

On another track, I thought initially that this code expected its input files to be in a specific place but it actually uses something called "hydra" to set its environment.

It looks like we can configure the input directory with `--config-path` so:

```
CUDA_VISIBLE_DEVICES=1 python3 -m rf2aa.run_inference --config-path=$(pwd)/config/inference --config-name owain
```

The problem is it still wants a lot of things in the CWD...


```
lrwxrwxrwx 1 uccaoke uccapc3   44 Mar  6 17:53 bfd -> /home/uccaoke/Scratch/protein_shake/rfaa/bfd
drwx------ 3 uccaoke uccapc3 4.0K Mar  6 17:48 config
drwx------ 5 uccaoke uccapc3 4.0K Mar  6 17:55 examples
lrwxrwxrwx 1 uccaoke uccapc3   53 Mar  6 17:57 make_msa.sh -> /home/uccaoke/Source/RoseTTAFold-All-Atom/make_msa.sh
lrwxrwxrwx 1 uccaoke uccapc3   57 Mar  6 17:53 pdb100_2021Mar03 -> /home/uccaoke/Scratch/protein_shake/rfaa/pdb100_2021Mar03
lrwxrwxrwx 1 uccaoke uccapc3   70 Mar  6 17:54 RFAA_paper_weights.pt -> /home/uccaoke/ACFS/Datasets/RoseTTaFold_All-Atom/RFAA_paper_weights.pt
lrwxrwxrwx 1 uccaoke uccapc3   57 Mar  6 17:53 UniRef30_2020_06 -> /home/uccaoke/Scratch/protein_shake/rfaa/UniRef30_2020_06
```

I'm not sure if this is the "minimum" either as the test job us running.

That recreates good old:

```
cat: 7u7w_protein/A/t000_.ss2: No such file or directory
```

So presumably we need to link in that as well.

```
ln -s ~/Source/RoseTTAFold-All-Atom/input_prep .
```

Assuming this is the state of the world and I can't work out out to de-Mamba the Mamba environment I think the central install process would look like:

1. Environment module adds a directory to the `$PATH` which contains two scripts.
2. Script 1 -> "blesses" a directory with all the necessary sym-links.
3. Script 2 -> is a "sourceme" which sources the two `.sh` files to enable Mamba, and activates the `RFAA` environemnt, and sets the `$PYTHONPATH` to the central install.

So when setting up a new run, users would load the module, "bless" the directory they want to put the files in, put their config + input files in, and then source the sourceme script.  They should then run RFAA as normal.

If we can de-Mamba the environment sucessfully, all of 3. can be done replaced by the module.

If we can control where RFAA looks for things (hydra???) then we can lose 2.

Well that test run failed with the same error as we see above.

```
CUDA_VISIBLE_DEVICES=1 python3 -m rf2aa.run_inference --config-path=$(pwd)/config/inference --config-name owain
/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA/lib/python3.10/site-packages/hydra/_internal/defaults_list.py:251: UserWarning: In 'owain': Defaults list is missing `_self_`. See https://hydra.cc/docs/1.2/upgrades/1.0_to_1.1/default_composition_order for more information
  warnings.warn(msg, UserWarning)
Using the cif atom ordering for TRP.
./make_msa.sh examples/protein/7u7w_A.fasta 7u7w_protein/A 4 64  pdb100_2021Mar03/pdb100_2021Mar03
Predicting: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.32sequences/s]
Running HHblits against UniRef30 with E-value cutoff 1e-10
- 18:13:49.653 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 18:13:49.653 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov75.a3m

- 18:13:49.897 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 18:14:24.089 INFO: Input file = 7u7w_protein/A/hhblits/t000_.1e-10.a3m

- 18:14:24.089 INFO: Output file = 7u7w_protein/A/hhblits/t000_.1e-10.id90cov50.a3m

- 18:14:24.323 WARNING: Maximum number 100000 of sequences exceeded in file 7u7w_protein/A/hhblits/t000_.1e-10.a3m

Running PSIPRED
Running hhsearch
- 18:16:38.763 ERROR: In /opt/conda/conda-bld/hhsuite_1709621322429/work/src/hhalignment.cpp:223: Read:

- 18:16:38.763 ERROR:   sequence ss_pred contains no residues.

Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 206, in main
    runner.infer()
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 153, in infer
    self.parse_inference_config()
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/run_inference.py", line 46, in parse_inference_config
    protein_input = generate_msa_and_load_protein(
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 93, in generate_msa_and_load_protein
    return load_protein(str(msa_file), str(hhr_file), str(atab_file), model_runner)
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 66, in load_protein
    xyz_t, t1d, mask_t, _ = get_templates(
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/protein.py", line 30, in get_templates
    ) = parse_templates_raw(ffdb, hhr_fn=hhr_fn, atab_fn=atab_fn)
  File "/home/uccaoke/Source/RoseTTAFold-All-Atom/rf2aa/data/parsers.py", line 628, in parse_templates_raw
    for l in open(atab_fn, "r").readlines():
FileNotFoundError: [Errno 2] No such file or directory: '7u7w_protein/A/t000_.atab'

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
```

That's bad because it means that we can't use the approach I suggested, but good because it's the *same* as the un-Mambaing error which implies that the fix for one is the fix for both.

--- Update 13:36 on Friday 7th of March.

I found this issue will poking about the RoseTTAfold-AA repo (https://github.com/baker-laboratory/RoseTTAFold-All-Atom/issues/105) and so I decided to fix this.

I've just done a successful run with:

* sourced conda + mamba settings
* set PATH to include the RFAA conda, the root of the RoseTTAFold-AA, and the input_prep sub-dir
* The `./` removed from that line.
* `export CONDA_PREFIX=/lustre/scratch/scratch/uccaoke/miniforge3/envs/RFAA`

This is really good progress!

I'm removing sym-links one at a time to see what we can get away with, but while it was bleak before I have more hope.

I'm now trying with the `PYTHONPATH` set to include the rfaa folder.

--- Update 15:52

It worked!