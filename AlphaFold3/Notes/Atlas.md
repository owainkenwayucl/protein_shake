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

## 2. Prepare the requirements.tx

I removed every line with `nvidia` or `hash=sha256` in it, as well as setting the JAX version to 0.5.0 as it is in the community container. `pip install`ing this worked as far at least as the `pip install` step.

## 3. Work out which base continer to use.

Although 0.5.0 is faster than 0.4.34 (which is the one installed in the nvidia container) I think we want to use as close to versions of things as the nvidia container because we don't know what other nasties are in store for us.

There is a `rocm/jax-community:rocm6.2.4-jax0.4.35-py3.11.10` which would be at most a minor change over the version used in the nvidia container, and `rocm/jax-community:rocm6.2.3-jax0.4.34-py3.11.10` which is exactly the same version but a fractionally older ROCm. Choices!

I am pulling both images.