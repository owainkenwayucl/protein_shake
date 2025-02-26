# Notes on trying to build the Alphafold 3 container.

## Build system

Deployed standard "Docker" plan + playbooks on Condenser.

Source repo: https://github.com/google-deepmind/alphafold3

Need to apply fixes to Dockerfile:

1. Use apt-get instead of apt (this is the intended tool for scripts)
2. Use apt-get to instal hmmr - hmmr is served from an http URL and they don't even check it. If the Ubuntu one doesn't work we can get a valid checksum from somewhere.
3. Investigate wtf is the "deadsnakes" ppa and whether we trust it.
   Lol, I'm not alone: https://askubuntu.com/questions/1398568/installing-python-who-is-deadsnakes-and-why-should-i-trust-them