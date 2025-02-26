# Notes on trying to build the Alphafold 3 container.

## Build system

Deployed standard "Docker" plan + playbooks on Condenser.

Source repo: https://github.com/google-deepmind/alphafold3

I've applied for access to the model parameters so that I can run tests. I am not allowed to share them with the rest of the Organisation because someone higher up than me needs to have the authority to do that.

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