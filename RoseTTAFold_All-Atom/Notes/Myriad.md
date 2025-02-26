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

5. 