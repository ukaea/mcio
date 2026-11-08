# Installation

## End user installation

MCIO can be installed from source using the below commands. The `cad` and `jupyter` extras install the
dependencies for CAD-to-CSG decomposition and Jupyter; they are not needed for building CSG models directly.

```bash

git clone https://github.com/ukaea/mcio.git

cd mcio

python -m pip install ".[cad,jupyter]"

```

## Dev installation

For those wishing to contribute to development, install the dependencies for testing, linting and building these docs.

```bash

git clone -b dev https://github.com/ukaea/mcio.git

cd mcio

python -m pip install ".[dev,docs,cad,jupyter]"

```

The docs can then be built locally with:

```bash
python -m sphinx -b html docs/source docs/build/html
```

## OpenMC

Writing OpenMC XML files requires the OpenMC Python API; see the
[OpenMC installation guide](https://docs.openmc.org/en/stable/quickinstall.html). The test suite also needs it, and
CI installs it at a fixed commit with:

```bash
python -m pip install "openmc @ git+https://github.com/openmc-dev/openmc.git@83a30f6860c79fac8fd58584489cc0d1abcb0376"
```

## Conda installation for CadQuery-OCP

Some operating systems don't have pre-compiled wheels for the CadQuery-OCP binaries, such as clusters running older
versions of Linux. Conda can be installed without super user access, as below:

```bash
# create a directory
mkdir -p ~/software/conda && cd ~/software/conda

# pull the mini-conda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# run the installer - provide this path when requested: ~/software/conda/miniconda3
bash Miniconda3-latest-Linux-x86_64.sh
```

Add the below to your `.bashrc` file. The `module` lines are for clusters that use environment modules; change the
GCC version to one available on your system, or remove them if your system doesn't use modules.

```bash
function load_conda {
    echo "loading conda"
    declare -a arr=(
        "GCC/13.3.0"
    )
    deactivate 2> /dev/null
    module purge
    for i in "${arr[@]}"; do
        module add "${i}" -s
    done
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$("$HOME/software/conda/miniconda3/bin/conda" 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "$HOME/software/conda/miniconda3/etc/profile.d/conda.sh" ]; then
            . "$HOME/software/conda/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="$HOME/software/conda/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
}
export -f load_conda
```

Now install the cadquery-ocp package in conda:

```bash
# reload your .bashrc file
. ~/.bashrc

# load the base conda environment
load_conda

# install cadquery and ocp
conda create --name ocp -c conda-forge cadquery ocp python=3.13 pip -y

# activate the ocp conda environment
conda activate ocp
```

Once in the Conda environment, you can install MCIO as described in [End user installation](#end-user-installation).
