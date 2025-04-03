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