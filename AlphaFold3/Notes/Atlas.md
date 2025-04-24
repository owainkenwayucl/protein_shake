# ATLAS

Atlas is the Cray 8-way Mi300x box we have on loan from HPE.

I think it would be neat to get Alphafold 3 working on it.

Steps I think:

1. test AMD's rocm JAX containers to see if they are a suitable base.
2. Strip hashes, JAX and nvidia stuff from Alphafold3's dev-requirements.txt
3. Work out which container is the correct one (`rocm/jax` vs the various `rocm/jax-community` contianers).
4. Build container.
5. Test!

## 1. Running JAX containers on ATLAS.

Like a lot of our test RCNIC systems, we prefer `podman` to `docker` because the former handles multi-user better than the latter.

Experimentation shows that with the `rocm/jax` container we can do:

```
podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/podmanhome:/home/uccaoke:Z docker.io/rocm/jax:latest /bin/bash -l
```

And get a shell.

Some notes. It appears `--ipc=host` and `--group-add keep-groups` are required for JAX to actually be able to see the devices.

```
root@6494dfd737ad:/home/uccaoke# python3 
Python 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)]
>>>
root@6494dfd737ad:/home/uccaoke# python3 jaxpi.py 160000000
Estimating Pi with:
  160000000 slices
  1 devices(s)

Estimated value of Pi: 3.141592502593994
Time taken: 8.435225248336792 seconds.

root@6494dfd737ad:/home/uccaoke# 
```

Sweet.

Tested with the `rocm/jax-community:latest` which is JAX 0.5.0 rather than 0.4.31 and it also works (and is quicker!).

## 2. Prepare the requirements.txt

I removed every line with `nvidia` or `hash=sha256` in it, as well as setting the JAX version to 0.5.0 as it is in the community container. `pip install`ing this worked as far at least as the `pip install` step.

There are requirements files in the `Pip` folder for the three possible versions of JAX.

## 3. Work out which base continer to use.

Although 0.5.0 is faster than 0.4.34 (which is the one installed in the nvidia container) I think we want to use as close to versions of things as the nvidia container because we don't know what other nasties are in store for us.

There is a `rocm/jax-community:rocm6.2.4-jax0.4.35-py3.11.10` which would be at most a minor change over the version used in the nvidia container, and `rocm/jax-community:rocm6.2.3-jax0.4.34-py3.11.10` which is exactly the same version but a fractionally older ROCm. Choices!

I am pulling both images.

I think I will go with 0.4.34 image as the software I think is most flimsy in this equation is AlphaFold3 and or `chex` which I know from other contexts has ... breaking changes with minor changes in JAX.

## 4. Building the container.

For reasons, `podman` uses the directory the `Dockerfile` is in as `.` so we need to put our `Dockerfile` in the `alphafold3` folder rather than overwrite the one in `docker/`

Then do:

```
podman build -t alphafold3:proteinshake -f Dockerfile.rocm
```

This fails with:

```
      [184/235] Building CXX object _deps/cifpp-build/test/CMakeFiles/test-main.dir/test-main.cpp.o
      FAILED: _deps/cifpp-build/test/CMakeFiles/test-main.dir/test-main.cpp.o
      /usr/bin/g++  -pthread -DCACHE_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/var/cache/libcifpp\" -DCATCH22=1 -DDATA_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/share/libcifpp\" -I/tmp/tmpzydwn6cp/build/_deps/cifpp-s
rc/include -I/tmp/tmpzydwn6cp/build/_deps/catch2-src/single_include -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3 -DNDEBUG -std=gnu++2a -fPIC -MD -MT _deps/cifpp-build/test/CMakeFi
les/test-main.dir/test-main.cpp.o -MF _deps/cifpp-build/test/CMakeFiles/test-main.dir/test-main.cpp.o.d -o _deps/cifpp-build/test/CMakeFiles/test-main.dir/test-main.cpp.o -c /tmp/tmpzydwn6cp/build/_deps/cif
pp-src/test/test-main.cpp
      In file included from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/test/test-main.cpp:5:
      /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/utilities.hpp:193:2: error: ‘requires’ does not name a type
        193 |  requires std::is_assignable_v<std::string_view, T>
            |  ^~~~~~~~
      [185/235] Building CXX object CMakeFiles/cpp.dir/src/alphafold3/structure/cpp/string_array_pybind.cc.o
      [186/235] Building CXX object _deps/cifpp-build/test/CMakeFiles/unit-3d-test.dir/unit-3d-test.cpp.o
      FAILED: _deps/cifpp-build/test/CMakeFiles/unit-3d-test.dir/unit-3d-test.cpp.o
      /usr/bin/g++  -pthread -DCACHE_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/var/cache/libcifpp\" -DCATCH22=1 -DDATA_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/share/libcifpp\" -I/tmp/tmpzydwn6cp/build/_deps/my-eigen3-src -I/tmp/tmpzydwn6cp/build/_deps/cifpp-src/include -I/tmp/tmpzydwn6cp/build/_deps/catch2-src/single_include -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3 -DNDEBUG -std=gnu++2a -fPIE -MD -MT _deps/cifpp-build/test/CMakeFiles/unit-3d-test.dir/unit-3d-test.cpp.o -MF _deps/cifpp-build/test/CMakeFiles/unit-3d-test.dir/unit-3d-test.cpp.o.d -o _deps/cifpp-build/test/CMakeFiles/unit-3d-test.dir/unit-3d-test.cpp.o -c /tmp/tmpzydwn6cp/build/_deps/cifpp-src/test/unit-3d-test.cpp
      In file included from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/test/unit-3d-test.cpp:31:
      /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/utilities.hpp:193:2: error: ‘requires’ does not name a type
        193 |  requires std::is_assignable_v<std::string_view, T>
            |  ^~~~~~~~
      [187/235] Building CXX object CMakeFiles/cpp.dir/src/alphafold3/parsers/cpp/cif_dict_pybind.cc.o
      [188/235] Building CXX object _deps/cifpp-build/CMakeFiles/cifpp.dir/src/symmetry.cpp.o
      FAILED: _deps/cifpp-build/CMakeFiles/cifpp.dir/src/symmetry.cpp.o
      /usr/bin/g++  -pthread -DBOOST_REGEX_STANDALONE=1 -DCACHE_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/var/cache/libcifpp\" -DDATA_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/share/libcifpp\" -DUSE_BOOST_REGEX=1 -I/tmp/tmpzydwn6cp/build/_deps/cifpp-src/include -I/tmp/tmpzydwn6cp/build/_deps/boost-rx-src/include -I/tmp/tmpzydwn6cp/build/_deps/my-eigen3-src -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3 -DNDEBUG -std=gnu++2a -fPIC -MD -MT _deps/cifpp-build/CMakeFiles/cifpp.dir/src/symmetry.cpp.o -MF _deps/cifpp-build/CMakeFiles/cifpp.dir/src/symmetry.cpp.o.d -o _deps/cifpp-build/CMakeFiles/cifpp.dir/src/symmetry.cpp.o -c /tmp/tmpzydwn6cp/build/_deps/cifpp-src/src/symmetry.cpp
      In file included from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/item.hpp:32,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/row.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/condition.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/category.hpp:31,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/datablock.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/src/symmetry.cpp:28:
      /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/utilities.hpp:193:2: error: ‘requires’ does not name a type
        193 |  requires std::is_assignable_v<std::string_view, T>
            |  ^~~~~~~~
      [189/235] Building CXX object _deps/cifpp-build/CMa      FAILED: _deps/cifpp-build/CMakeFiles/cifpp.dir/src/pdb/cif2pdb.cpp.o
      /usr/bin/g++  -pthread -DBOOST_REGEX_STANDALONE=1 -DCACHE_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/var/cache/libcifpp\" -DDATA_DIR=\"/tmp/tmpzydwn6cp/wheel/platlib/share/libcifpp\" -DUSE_BOOST_REGEX=1 -I/tmp/tmpzydwn6cp/build/_deps/cifpp-src/include -I/tmp/tmpzydwn6cp/build/_deps/boost-rx-src/include -I/tmp/tmpzydwn6cp/build/_deps/my-eigen3-src -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers -O3 -DNDEBUG -std=gnu++2a -fPIC -MD -MT _deps/cifpp-build/CMakeFiles/cifpp.dir/src/pdb/cif2pdb.cpp.o -MF _deps/cifpp-build/CMakeFiles/cifpp.dir/src/pdb/cif2pdb.cpp.o.d -o _deps/cifpp-build/CMakeFiles/cifpp.dir/src/pdb/cif2pdb.cpp.o -c /tmp/tmpzydwn6cp/build/_deps/cifpp-src/src/pdb/cif2pdb.cpp
      In file included from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++.hpp:29,
                       from /tmp/tmpzydwn6cp/build/_deps/cifpp-src/src/pdb/cif2pdb.cpp:27:
      /tmp/tmpzydwn6cp/build/_deps/cifpp-src/include/cif++/utilities.hpp:193:2: error: ‘requires’ does not name a type
        193 |  requires std::is_assignable_v<std::string_view, T>
            |  ^~~~~~~~
      ninja: build stopped: subcommand failed.

      *** CMake build failed
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for alphafold3
ERROR: Failed to build installable wheels for some pyproject.toml based projects (alphafold3)
Error: building at STEP "RUN pip3 install --no-deps .": while running runtime: exit status 1
```

Which is obviously a problem.

It looks like this is this - so we need a newer GCC? https://github.com/google-deepmind/alphafold3/issues/223

If this is indeed caused by too old a version of GCC (9.4) then the newest we can install from `apt` on Ubuntu 22.04 which is the base of the JAX container is GCC 10.

So trying that.

Adding `gcc-10` and `g++-10` and setting `CC` and `CXX` appropriately seems to allow this to build.

Nice, look how *massive* these images are!

```
[uccaoke@ip-10-134-25-2 alphafold3]$ podman image ls
REPOSITORY                     TAG                            IMAGE ID      CREATED         SIZE
localhost/alphafold3           proteinshake                   889555914640  57 seconds ago  35 GB
docker.io/rocm/jax             latest                         d949265c6ac2  3 weeks ago     33.5 GB
docker.io/rocm/jax-community   latest                         193ba487b999  6 weeks ago     31.9 GB
docker.io/library/hello-world  latest                         74cc54e27dc4  2 months ago    27.1 kB
docker.io/rocm/jax-community   rocm6.2.4-jax0.4.35-py3.11.10  ef50d5181ba5  3 months ago    30.5 GB
docker.io/rocm/jax-community   rocm6.2.3-jax0.4.34-py3.11.10  b229479e4af8  5 months ago    30.4 GB
```

## 5. Test

I'm downloading the massive databases to ATLAS. Wish me luck.

```
podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/af_input:/root/af_input:Z -v $HOME/af_output:/root/af_output:Z -v $HOME/Datasets/alphafold3/weights:/root/models:Z -v $HOME/Datasets/alphafold3/databases:/root/public_databases  localhost/alphafold3:proteinshake sh -c "XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter' python3 /app/alphafold/run_alphafold.py --json_path=/root/af_input/fold_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output --flash_attention_implementation=xla"
```

Sad beep:

```
Traceback (most recent call last):
  File "/alphafold3_venv/lib/python3.11/site-packages/jax_triton/__init__.py", line 37, in <module>
    get_compute_capability = gpu_triton.get_compute_capability
                             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: module 'jaxlib.gpu_triton' has no attribute 'get_compute_capability'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/app/alphafold/run_alphafold.py", line 45, in <module>
    from alphafold3.jax.attention import attention
  File "/alphafold3_venv/lib/python3.11/site-packages/alphafold3/jax/attention/attention.py", line 17, in <module>
    from alphafold3.jax.attention import flash_attention as attention_triton
  File "/alphafold3_venv/lib/python3.11/site-packages/alphafold3/jax/attention/flash_attention.py", line 22, in <module>
    import jax_triton as jt
  File "/alphafold3_venv/lib/python3.11/site-packages/jax_triton/__init__.py", line 40, in <module>
    raise ImportError(
ImportError: jax-triton requires JAX to be installed with GPU support. The installation page on the JAX documentation website includes instructions for installing a supported version:
https://jax.readthedocs.io/en/latest/installation.html
```

Well that sucks. It looks like installing on the base of the community jax means the test in `jaxlib.gpu_triton` is failing. On the older JAC, 0.4.31 in the "non-community release", we can do: 

```
root@e549fdfdc21a:/# python3
Python 3.10.12 (main, Feb  4 2025, 14:57:36) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
<module 'jax_rocm60_plugin._triton' from '/opt/venv/lib/python3.10/site-packages/jax_rocm60_plugin/_triton.so'>
>>> 
```

On the community containers, it does e.g:

```
Python 3.10.16 (main, Feb 17 2025, 01:40:07) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
>>> 
```

We can see it's failing this test:

https://github.com/jax-ml/jax/blob/bd9220838f4758bda14150a99d80293dc4dc0be4/jaxlib/gpu_triton.py#L20

```python
_hip_triton = import_from_plugin("rocm", "_triton")
```

Oh well https://github.com/ROCm/jax/issues/339

I've built a container with a horrible kludge in it to use jac 0.4.31.

```
podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/af_input:/root/af_input:Z -v $HOME/af_output:/root/af_output:Z -v $HOME/Datasets/alphafold3/weights:/root/models:Z -v $HOME/Datasets/alphafold3/databases:/root/public_databases  localhost/alphafold3:proteinshake-old sh -c "XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter' python3 /app/alphafold/run_alphafold.py --json_path=/root/af_input/fold_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output --flash_attention_implementation=xla"
I0403 15:25:16.346253 140180234358912 xla_bridge.py:897] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
Traceback (most recent call last):
  File "/app/alphafold/run_alphafold.py", line 829, in <module>
    app.run(main)
  File "/alphafold3_venv/lib/python3.12/site-packages/absl/app.py", line 308, in run
    _run_main(main, args)
  File "/alphafold3_venv/lib/python3.12/site-packages/absl/app.py", line 254, in _run_main
    sys.exit(main(argv))
             ^^^^^^^^^^
  File "/app/alphafold/run_alphafold.py", line 717, in main
    compute_capability = float(
                         ^^^^^^
ValueError: could not convert string to float: 'gfx942'
[uccaoke@ip-10-134-25-2 alphafold3]$ 
```

We know what this is, this is the check for Volta cards which need to be treated special. So we should be able to modify `run_alphafold.py` to fix it.

```
[uccaoke@ip-10-134-25-2 alphafold3]$ podman run -it --rm --group-add keep-groups --device /dev/kfd:rwm --device /dev/dri:rwm --ipc=host -v $HOME/af_input:/root/af_input:Z -v $HOME/af_output:/root/af_output:
Z -v $HOME/Datasets/alphafold3/weights:/root/models:Z -v $HOME/Datasets/alphafold3/databases:/root/public_databases  localhost/alphafold3:proteinshake-old sh -c "XLA_FLAGS='--xla_disable_hlo_passes=custom-k
ernel-fusion-rewriter' python3 /app/alphafold/run_alphafold.py --json_path=/root/af_input/fold_input.json --model_dir=/root/models --db_dir=/root/public_databases --output_dir=/root/af_output --flash_attent
ion_implementation=xla"
I0403 15:32:51.938552 140574038458496 xla_bridge.py:897] Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory

Running AlphaFold 3. Please note that standard AlphaFold 3 model parameters are
only available under terms of use provided at
https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md.
If you do not agree to these terms and are using AlphaFold 3 derived model
parameters, cancel execution of AlphaFold 3 inference with CTRL-C, and do not
use the model parameters.

Found local devices: [RocmDevice(id=0), RocmDevice(id=1), RocmDevice(id=2), RocmDevice(id=3), RocmDevice(id=4), RocmDevice(id=5), RocmDevice(id=6), RocmDevice(id=7)], using device 0: rocm:0
Building model from scratch...
Checking that model parameters can be loaded...

Running fold job 2PV7...
Output will be written in /root/af_output/2pv7
Running data pipeline...
Running data pipeline for chain A... 
I0403 15:33:05.314379 140574038458496 pipeline.py:82] Getting protein MSAs for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVL
GLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:33:05.318038 140189536085696 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQV
VVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:33:05.318757 140189527692992 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQV
VVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:33:05.319428 140189519300288 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:33:05.319853 140189536085696 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpegeja4hy/output.sto --noali --F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpegeja4hy/query.fasta /root/public_databases/uniref90_2022_05.fa"
I0403 15:33:05.319942 140189510907584 jackhmmer.py:78] Query sequence: GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:33:05.320207 140189527692992 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpi3y6x7q0/output.sto --noali --F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpi3y6x7q0/query.fasta /root/public_databases/mgy_clusters_2022_05.fa"
I0403 15:33:05.320554 140189519300288 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpm81q7p8q/output.sto --noali --F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpm81q7p8q/query.fasta /root/public_databases/bfd-first_non_consensus_sequences.fasta"
I0403 15:33:05.321136 140189510907584 subprocess_utils.py:68] Launching subprocess "/usr/bin/jackhmmer -o /dev/null -A /tmp/tmpw6k_a9p1/output.sto --noali --F1 0.0005 --F2 5e-05 --F3 5e-07 --cpu 8 -N 1 -E 0.0001 --incE 0.0001 /tmp/tmpw6k_a9p1/query.fasta /root/public_databases/uniprot_all_2021_04.fa"
I0403 15:34:09.994639 140189519300288 subprocess_utils.py:97] Finished Jackhmmer (bfd-first_non_consensus_sequences.fasta) in 64.674 seconds
I0403 15:37:38.072148 140189536085696 subprocess_utils.py:97] Finished Jackhmmer (uniref90_2022_05.fa) in 272.752 seconds
I0403 15:39:42.036951 140189510907584 subprocess_utils.py:97] Finished Jackhmmer (uniprot_all_2021_04.fa) in 396.716 seconds
I0403 15:41:53.874885 140189527692992 subprocess_utils.py:97] Finished Jackhmmer (mgy_clusters_2022_05.fa) in 528.554 seconds
I0403 15:41:53.952558 140574038458496 pipeline.py:115] Getting protein MSAs took 528.64 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:41:53.952702 140574038458496 pipeline.py:121] Deduplicating MSAs for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:41:53.976438 140574038458496 pipeline.py:134] Deduplicating MSAs took 0.02 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG, found 8506 unpaired sequences, 7080 paired sequences
I0403 15:41:53.979389 140574038458496 pipeline.py:40] Getting protein templates for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
I0403 15:41:54.053283 140574038458496 subprocess_utils.py:68] Launching subprocess "/usr/bin/hmmbuild --informat stockholm --hand --amino /tmp/tmpfuhvudia/output.hmm /tmp/tmpfuhvudia/query.msa"
I0403 15:41:54.385916 140574038458496 subprocess_utils.py:97] Finished Hmmbuild in 0.332 seconds
I0403 15:41:54.388776 140574038458496 subprocess_utils.py:68] Launching subprocess "/usr/bin/hmmsearch --noali --cpu 8 --F1 0.1 --F2 0.1 --F3 0.1 -E 100 --incE 100 --domE 100 --incdomE 100 -A /tmp/tmpr9p4kc34/output.sto /tmp/tmpr9p4kc34/query.hmm /root/public_databases/pdb_seqres_2022_09_28.fasta"
I0403 15:42:00.482367 140574038458496 subprocess_utils.py:97] Finished Hmmsearch (pdb_seqres_2022_09_28.fasta) in 6.094 seconds
I0403 15:42:00.770663 140574038458496 pipeline.py:52] Getting 4 protein templates took 6.79 seconds for sequence GMRESYANENQFGFKTINSDIHKIVIVGGYGKLGGLFARYLRASGYPISILDREDWAVAESILANADVVIVSVPINLTLETIERLKPYLTENMLLADLTSVKREPLAKMLEVHTGAVLGLHPMFGADIASMAKQVVVRCDGRFPERYEWLLEQIQIWGAKIYQTNATEHDHNMTYIQALRHFSTFANGLHLSKQPINLANLLALSSPIYRLELAMIGRLFAQDAELYADIIMDKSENLAVIETLKQTYDEALTFFENNDRQGFIDAFHKVRDWFGDYSEQFLKESRQLLQQANDLKQG
Running data pipeline for chain A took 535.51 seconds
Running data pipeline for chain B... 
Running data pipeline for chain B took 0.05 seconds
Writing model input JSON to /root/af_output/2pv7/2pv7_data.json
Predicting 3D structure for 2PV7 with 1 seed(s)...
Featurising data with 1 seed(s)...   
Featurising data with seed 1.
I0403 15:42:05.221202 140574038458496 pipeline.py:166] processing 2PV7, random_seed=1
I0403 15:42:05.252050 140574038458496 pipeline.py:259] Calculating bucket size for input with 596 tokens.
I0403 15:42:05.252220 140574038458496 pipeline.py:265] Got bucket size 768 for input with 596 tokens, resulting in 172 padded tokens.
Featurising data with seed 1 took 7.07 seconds.
Featurising data with 1 seed(s) took 11.39 seconds.
Running model inference and extracting output structure samples with 1 seed(s)...
Running model inference with seed 1...
Running model inference with seed 1 took 129.31 seconds.
Extracting inference results with seed 1...
Extracting 5 inference samples with seed 1 took 0.45 seconds.
Running model inference and extracting output structures with 1 seed(s) took 129.76 seconds.
Writing outputs with 1 seed(s)...
Fold job 2PV7 done, output written to /root/af_output/2pv7

Done running 1 fold jobs.
```

There is a patch for `run_alphafold.py` in `Patches` - it needs to be applied before you build the container.

## Returning after notes

https://github.com/ROCm/jax/issues/339#issuecomment-2825420288

Let's see if we can get the newer JAX to play ball.

On the "ubuntu" dev image:

```
root@9f852385265d:/# pip install jax==0.4.38 jaxlib==0.4.38 jax_rocm60_plugin==0.4.35 jax_rocm60_pjrt==0.4.35
Collecting jax==0.4.38
  Downloading jax-0.4.38-py3-none-any.whl (2.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 38.7 MB/s eta 0:00:00
Collecting jaxlib==0.4.38
  Downloading jaxlib-0.4.38-cp310-cp310-manylinux2014_x86_64.whl (101.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.7/101.7 MB 62.9 MB/s eta 0:00:00
Collecting jax_rocm60_plugin==0.4.35
  Downloading jax_rocm60_plugin-0.4.35-cp310-cp310-manylinux_2_28_x86_64.whl (6.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.7/6.7 MB 9.7 MB/s eta 0:00:00
Collecting jax_rocm60_pjrt==0.4.35
  Downloading jax_rocm60_pjrt-0.4.35-py3-none-manylinux_2_28_x86_64.whl (104.0 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 104.0/104.0 MB 7.7 MB/s eta 0:00:00
Collecting ml_dtypes>=0.4.0
  Downloading ml_dtypes-0.5.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 165.9 MB/s eta 0:00:00
Collecting opt_einsum
  Downloading opt_einsum-3.4.0-py3-none-any.whl (71 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 71.9/71.9 KB 60.2 MB/s eta 0:00:00
Collecting numpy>=1.24
  Downloading numpy-2.2.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (16.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 16.4/16.4 MB 175.6 MB/s eta 0:00:00
Collecting scipy>=1.10
  Downloading scipy-1.15.2-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (37.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 37.6/37.6 MB 91.6 MB/s eta 0:00:00
Installing collected packages: jax_rocm60_pjrt, opt_einsum, numpy, jax_rocm60_plugin, scipy, ml_dtypes, jaxlib, jax
Successfully installed jax-0.4.38 jax_rocm60_pjrt-0.4.35 jax_rocm60_plugin-0.4.35 jaxlib-0.4.38 ml_dtypes-0.5.1 numpy-2.2.5 opt_einsum-3.4.0 scipy-1.15.2
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
root@9f852385265d:/# python3
Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
root@9f852385265d:/# rocm-smi


============================================ ROCm System Management Interface ============================================
====================================================== Concise Info ======================================================
Device  Node  IDs              Temp        Power     Partitions          SCLK    MCLK    Fan  Perf  PwrCap  VRAM%  GPU%  
              (DID,     GUID)  (Junction)  (Socket)  (Mem, Compute, ID)                                                  
==========================================================================================================================
0       4     0x74a1,   28851  50.0°C      133.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
1       5     0x74a1,   50724  44.0°C      127.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
2       3     0x74a1,   60940  43.0°C      128.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
3       2     0x74a1,   22683  50.0°C      135.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
4       8     0x74a1,   53458  50.0°C      132.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
5       9     0x74a1,   52940  44.0°C      125.0W    NPS1, SPX, 0        136Mhz  900Mhz  0%   auto  750.0W  0%     0%
6       7     0x74a1,   59108  51.0°C      133.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
7       6     0x74a1,   39291  44.0°C      126.0W    NPS1, SPX, 0        137Mhz  900Mhz  0%   auto  750.0W  0%     0%
==========================================================================================================================
================================================== End of ROCm SMI Log ===================================================
root@9f852385265d:/# python3
Python 3.10.12 (main, Nov  6 2024, 20:22:13) [GCC 11.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
>>> 
```

Sad beep.

Ok, let's try the `rocm/jax` image:

```
root@0ba725805946:/# pip install jax==0.4.38 jaxlib==0.4.38 jax_rocm60_plugin==0.4.31 jax_rocm60_pjrt==0.4.31
Collecting jax==0.4.38
  Downloading jax-0.4.38-py3-none-any.whl.metadata (22 kB)
Collecting jaxlib==0.4.38
  Downloading jaxlib-0.4.38-cp312-cp312-manylinux2014_x86_64.whl.metadata (1.0 kB)
Requirement already satisfied: jax_rocm60_plugin==0.4.31 in /opt/venv/lib/python3.12/site-packages (0.4.31)
Requirement already satisfied: jax_rocm60_pjrt==0.4.31 in /opt/venv/lib/python3.12/site-packages (0.4.31)
Requirement already satisfied: ml_dtypes>=0.4.0 in /opt/venv/lib/python3.12/site-packages (from jax==0.4.38) (0.5.1)
Requirement already satisfied: numpy>=1.24 in /opt/venv/lib/python3.12/site-packages (from jax==0.4.38) (1.26.4)
Requirement already satisfied: opt_einsum in /opt/venv/lib/python3.12/site-packages (from jax==0.4.38) (3.4.0)
Requirement already satisfied: scipy>=1.10 in /opt/venv/lib/python3.12/site-packages (from jax==0.4.38) (1.15.2)
Downloading jax-0.4.38-py3-none-any.whl (2.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.2/2.2 MB 65.7 MB/s eta 0:00:00
Downloading jaxlib-0.4.38-cp312-cp312-manylinux2014_x86_64.whl (101.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 101.8/101.8 MB 52.9 MB/s eta 0:00:00
Installing collected packages: jaxlib, jax
  Attempting uninstall: jaxlib
    Found existing installation: jaxlib 0.4.31
    Uninstalling jaxlib-0.4.31:
      Successfully uninstalled jaxlib-0.4.31
  Attempting uninstall: jax
    Found existing installation: jax 0.4.31
    Uninstalling jax-0.4.31:
      Successfully uninstalled jax-0.4.31
Successfully installed jax-0.4.38 jaxlib-0.4.38
root@0ba725805946:/# python3
Python 3.12.3 (main, Feb  4 2025, 14:48:35) [GCC 13.3.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> from jaxlib import gpu_triton
>>> gpu_triton._hip_triton
<module 'jax_rocm60_plugin._triton' from '/opt/venv/lib/python3.12/site-packages/jax_rocm60_plugin/_triton.so'>
```

Much better news - as long as we pin the plugin versions at 0.4.31 which is the version with the fix, we can slightly update JAX.