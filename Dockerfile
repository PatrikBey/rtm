# Interactive Python environment for the rtm (Reaction Time Modelling) project.
# Does not build an app or copy code — intended for interactive use:
#   docker run --rm -it -v $(pwd):/workspace rtm-env python

FROM python:3.11-slim

# System-level build dependencies required by PyMC / PyTensor and matplotlib
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgfortran5 \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Pin all packages to known-compatible versions.
# PyMC 5.x requires Python >=3.10; pylater requires Python <3.12; using 3.11.
# arviz version is aligned with the pymc 5.28.x release series.
RUN pip install --no-cache-dir \
        "numpy==2.3.5" \
        "pandas==2.2.3" \
        "matplotlib==3.10.3" \
        "scipy>=1.14,<2" \
        "pytensor==2.38.3" \
        "pymc==5.28.5" \
        "arviz==0.23.4" \
        "pylater @ git+https://github.com/unimelbmdap/pylater.git"

# pylater calls pt.repeat(x=...) but pytensor 2.38.x renamed that keyword
# argument from 'x' to 'a' (following numpy's API).  Patch the installed file.
RUN python - <<'EOF'
import os, pylater
path = os.path.join(os.path.dirname(pylater.__file__), "model.py")
src = open(path).read()
src = src.replace("x=sigma,", "sigma,").replace("x=k,", "k,")
open(path, "w").write(src)
EOF

WORKDIR /workspace
