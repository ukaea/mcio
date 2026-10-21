# Installation

## End user installation

MCIO can be installed from source using the below commands

```bash

git clone git@gitlab.ukaea.uk:rworrall/mcio.git

cd mcio

python -m pip install .[cad,jupyter]

```

## Dev installation

For those wishing to contribute to development, you can install all dependencies for docs.

```bash

git clone -b dev git@gitlab.ukaea.uk:rworrall/mcio.git

cd mcio

python -m pip install .[dev,docs,cad,jupyter]

```

## Conda installation for CadQuery-OCP (Open Cascade python wrapper used for CAD decomposition and inversion)

Some operating systems don't have pre-compiled wheels for the CadQuery-OCP binaries, such as on clusters with older
version of linux. Conda can be installed without the need for super user access, as below:

```bash
# create a directory
mkdir -p ~/software/conda && cd ~/software/conda

# pull the mini-conda installer
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# run the installer - provide path when requested (/lustre/home/<USERNAME>/software/conda/miniconda3)
bash Miniconda3-latest-Linux-x86_64.sh

```

Add the below to your .bashrc file, replacing your user name with the <USERNAME> string.

```bash
# to add to your .bashrc, swapping out <USERNAME> for your 

function load_conda {
    echo "loading conda"
    declare -a arr=(
        "GCC/13.3.0"
    )
    deactivate
    module purge
    for i in "${arr[@]}"; do
        module add "${i}" -s
    done
    # >>> conda initialize >>>
    # !! Contents within this block are managed by 'conda init' !!
    __conda_setup="$('/lustre/home/<USERNAME>/software/conda/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
    if [ $? -eq 0 ]; then
        eval "$__conda_setup"
    else
        if [ -f "/lustre/home/<USERNAME>/software/conda/miniconda3/etc/profile.d/conda.sh" ]; then
            . "/lustre/home/<USERNAME>/software/conda/miniconda3/etc/profile.d/conda.sh"
        else
            export PATH="/lustre/home/<USERNAME>/software/conda/miniconda3/bin:$PATH"
        fi
    fi
    unset __conda_setup
    # <<< conda initialize <<<
    }
export -f load_conda
```

Now install the cadquery-ocp package in conda

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

Once in the Conda environment, you can install MCIO as described above in the installation section.