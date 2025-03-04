# Notes on trying to build the Alphafold 3 container.

## Build system

Deployed standard "Docker" plan + playbooks on Condenser.

Source repo: https://github.com/google-deepmind/alphafold3

I've applied for access to the model parameters so that I can run tests. I am not allowed to share them with the rest of the Organisation because someone higher up than me needs to have the authority to do that.

Created `Scripts/download_databases.sh` to download databases to ACFS.

Need to apply fixes to Dockerfile:

1. Use apt-get instead of apt (this is the intended tool for scripts)
2. Use apt-get to instal hmmr - hmmr is served from an http URL and they don't even check it. If the Ubuntu one doesn't work we can get a valid checksum from somewhere.
3. Investigate wtf is the "deadsnakes" ppa and whether we trust it.
   Lol, I'm not alone: https://askubuntu.com/questions/1398568/installing-python-who-is-deadsnakes-and-why-should-i-trust-them


Cloned repo onto build machine and substituted our Dockerfile.

Note: Apparently you can't ln -s Dockerfiles. THE PAIN.

Now running `docker build -t alphafold3 .`

That doesn't work so you have to go up a directory and do:

`docker build -t alphafold3 -f docker/Dockerfile .`

This got OOM killed.

Boosted RAM of VM to 26GiB.

Container built.

Exported with `docker save alphafold3 -o alphafold3-proteinshake.tar`

Gzipped the resulting `.tar` and checksummed:

```
4d6e7b9c47ec89470c27c57df7bde104f5a9b7729dd0bc4eaf3abce93c4574ccd6daaba8828aca37fb165b7376ad150247cd4c76e2aa73048a7bba06e2a18eb4  alphafold3-proteinshake.tar.gz
```

I'll put that somewhere so that I can install it on something once I have the weights + once the databases have downloaded.

As part of this I've discovered that Myriad + Kathleen are just about the only machines at UCL that can't see my Condenser S3.  Networks.

--- Update 22:04 on Friday 28th of February in the year of our Lord Two Thousand and Twenty Five.

Google sent me the Weights today so I've put them on the ACFS.

Since I was well enough to be back on campus I was able to use my laptop as an in-between to copy the Container to the ACFS as well (due to the network problem).

I've converted the Container to a `.sif` container (and put it in ACFS).

I realised that the database files are compressed with "Zstandard" which is some Facebook compression tool. I had to compile and install it which was a PITA because it uses CMake in an unusual and creative way.

I've set a job off on Myriad decompressing them to scratch:

```
#!/bin/bash -l

#$ -l mem=5G
#$ -l h_rt=48:0:0

#$ -N AF3_DB_decompress

#$ -cwd

module load zstd

cp ${HOME}/ACFS/Datasets/AlphaFold3/*.zst .

for a in $(ls *.zst)
do
        unzstd --rm ${a}
done
```

I copy them to Scratch first because `unzstd` decompresses files to the directory the `.zst` is in, seems to have no way of setting an output directory and of course the ACFS is read only on compute nodes. The `--rm` flag should delete the `.zst` files after decompressing them.

Running dangerously close to my quota on basically all filesystems at this point.

--- Update 15:06 on Sunday 2nd of March in the year of our Lord Two Thousand and Twenty Five.

I'm seriously struggling with disk space challenges - between 1TiB of ACFS and 1TiB of Scratch I do not have enough space to have the databases for both applications, on the basis that RoseTTAFold All-Atom is the most likely to work I've ditched the databases for AlphaFold3 for now while I try to unpack the RoseTTAFold All-Atom.

--- Update 11:40 on Tuesday 4th of March

Quota temporarily increased to 6TiB. Unpacking databases to scratch.

I forget to get pdb_2022_09_28_mmcif_files.tar.zst so I'm getting/unpacking that too.

*Edit*  - I didn't forget, I deleted it by doing `rm *.zstd` earlier because my script doesn't unpack it.

```
Myriad [node-e96a-001] AlphaFold3 :( > apptainer exec  --nv --no-home alphafold3-proteinshake.sif sh -c 'nvidia-smi'
Tue Mar  4 11:48:22 2025
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.127.05             Driver Version: 550.127.05     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla V100-PCIE-32GB           Off |   00000000:58:00.0 Off |                    0 |
| N/A   36C    P0             37W /  250W |       1MiB /  32768MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
|   1  Tesla V100-PCIE-32GB           Off |   00000000:D8:00.0 Off |                    0 |
| N/A   34C    P0             36W /  250W |       1MiB /  32768MiB |      1%      Default |
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

Our test command is:

```
apptainer exec  --nv --bind /home/uccaoke/Source/AlphaFold3/home/af_input:/root/af_input --bind /home/uccaoke/Source/AlphaFold3/home/af_output:/root/af_output --bind /home/uccaoke/ACFS/Datasets/AlphaFold3/Weights:/root/models --bind /home/uccaoke/Scratch/protein_shake/af3:/root/public_databases --no-home --no-mount bind-paths  alphafold3-proteinshake.sif sh -c "XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter' python3 /app/alphafold/run_alphafold.py --json_path=/root/af_input/fold_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output --flash_attention_implementation=xla"
```

This gives us:

```
Running AlphaFold 3. Please note that standard AlphaFold 3 model parameters are
only available under terms of use provided at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.
If you do not agree to these terms and are using AlphaFold 3 derived model
parameters, cancel execution of AlphaFold 3 inference with CTRL-C, and do not
use the model parameters.

Found local devices: [CudaDevice(id=0), CudaDevice(id=1)], using device 0: cuda:0
Building model from scratch...
Checking that model parameters can be loaded...
2025-03-04 12:55:32.285132: W external/xla/xla/service/gpu/nvptx_compiler.cc:930] The NVIDIA driver's CUDA version is 12.4 which is older than the PTX compiler version 12.6.77. Because the driver is older than the PTX compiler version, XLA is disabling parallel compilation, which may slow down compilation. You should update your NVIDIA driver or use the NVIDIA-provided CUDA forward compatibility packages.

Running fold job 2PV7...
Output will be written in /root/af_output/2pv7
Running data pipeline...
RRunning data pipeline for chain A...                                                                                                                        
I0304 12:55:37.168841 47694066176640 pipeline.py:82] Getting protein MSAs for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVV
IVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAE
LYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG                                                                                   
I0304 12:55:37.175542 47704449242688 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLK
PYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIE
TLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG                                                                                                    
I0304 12:55:37.175703 47740615312960 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLK
PYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIE
TLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG                                                                                                    
I0304 12:55:37.176079 47740617414208 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLK
PYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIE
TLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG                                                                                                    
I0304 12:55:37.176443 47740619515456 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLK
PYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIE
TLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG                                                                                                    
I0304 12:55:37.176863 47704449242688 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpue32utdc/output.sto --noali --
F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpue32utdc/query.fasta /root/public_databases/uniref90_2022_05.fa"               
I0304 12:55:37.176935 47740615312960 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpmmf15n4m/output.sto --noali --
F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpmmf15n4m/query.fasta /root/public_databases/mgy_clusters_2022_05.fa"           
I0304 12:55:37.177148 47740617414208 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpx0ebv6lz/output.sto --noali --
F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpx0ebv6lz/query.fasta /root/public_databases/bfd-first_non_consensus_sequences.f
asta"                                                                                                                                                       
I0304 12:55:37.177576 47740619515456 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpvd0ltelm/output.sto --noali --
F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpvd0ltelm/query.fasta /root/public_databases/uniprot_all_2021_04.fa"            
I0304 12:57:02.762822 47740617414208 subprocess_utils.py:97] Finished Jackhmmer (bfd-first_non_consensus_sequences.fasta) in 85.585 seconds                 
I0304 13:01:14.081996 47704449242688 subprocess_utils.py:97] Finished Jackhmmer (uniref90_2022_05.fa) in 336.905 seconds                                    
I0304 13:02:56.852339 47740619515456 subprocess_utils.py:97] Finished Jackhmmer (uniprot_all_2021_04.fa) in 439.675 seconds                                 
I0304 13:05:37.514951 47740615312960 subprocess_utils.py:97] Finished Jackhmmer (mgy_clusters_2022_05.fa) in 600.337 seconds
I0304 13:05:37.610370 47694066176640 pipeline.py:115] Getting protein MSAs took 600.44 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0304 13:05:37.610510 47694066176640 pipeline.py:121] Deduplicating MSAs for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0304 13:05:37.632619 47694066176640 pipeline.py:134] Deduplicating MSAs took 0.02 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG, found 8506 unpaired sequences, 7080 paired sequences
I0304 13:05:37.640418 47694066176640 pipeline.py:40] Getting protein templates for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0304 13:05:37.883495 47694066176640 subprocess_utils.py:68] Launching subprocess "/usr/bin/hmmbuild --informat stockholm --hand --amino /tmp/tmp9vetwmmi/output.hmm /tmp/tmp9vetwmmi/query.msa"
I0304 13:05:38.452481 47694066176640 subprocess_utils.py:97] Finished Hmmbuild in 0.569 seconds
I0304 13:05:38.455605 47694066176640 subprocess_utils.py:68] Launching subprocess "/usr/bin/hmmsearch --noali --cpu 8 --F1 0.1 --F2 0.1 --F3 0.1 -E 100 --incE 100 --domE 100 --incdomE 100 -A /tmp/tmp9xedbyvk/output.sto /tmp/tmp9xedbyvk/query.hmm /root/public_databases/pdb_seqres_2022_09_28.fasta"
I0304 13:05:48.859328 47694066176640 subprocess_utils.py:97] Finished Hmmsearch (pdb_seqres_2022_09_28.fasta) in 10.404 seconds
I0304 13:05:49.322858 47694066176640 pipeline.py:52] Getting 4 protein templates took 11.68 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
Running data pipeline for chain A took 612.25 seconds
Running data pipeline for chain B...
Running data pipeline for chain B took 0.09 seconds
Writing model input JSON to /root/af_output/2pv7/2pv7_data.json
Predicting 3D structure for 2PV7 with 1 seed(s)...
Featurising data with 1 seed(s)...
Featurising data with seed 1.
I0304 13:05:57.122287 47694066176640 pipeline.py:164] processing 2PV7, random_seed=1
I0304 13:05:57.164265 47694066176640 pipeline.py:257] Calculating bucket size for input with 596 tokens.
I0304 13:05:57.164416 47694066176640 pipeline.py:263] Got bucket size 768 for input with 596 tokens, resulting in 172 padded tokens.
Featurising data with seed 1 took 10.84 seconds.
Featurising data with 1 seed(s) took 18.41 seconds.
Running model inference and extracting output structure samples with 1 seed(s)...
Running model inference with seed 1...
Running model inference with seed 1 took 277.36 seconds.
Extracting inference results with seed 1...
Extracting 5 inference samples with seed 1 took 0.63 seconds.
Running model inference and extracting output structures with 1 seed(s) took 277.99 seconds.
Writing outputs with 1 seed(s)...
Fold job 2PV7 done, output written to /root/af_output/2pv7

Done running 1 fold jobs.

```