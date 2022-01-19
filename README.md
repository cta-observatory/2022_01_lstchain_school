# 2022_01_lstchain_school

LST analysis school January 2022
https://indico.cta-observatory.org/event/3687/

### For doing the hands-on exercises in your own computer (recommended option):

#### Download the school repository to your computer:

```
git clone https://github.com/cta-observatory/2022_01_lstchain_school.git
```

#### Install

- You will need to install [mamba](https://github.com/conda-forge/miniforge#mambaforge) :
```
conda install -c conda-forge -n base mamba
```

then, in the folder where you cloned your repo (2022_01_lstchain_school):

```
mamba env create -f environment.yml
```
This will install the version 0.8.4 of lstchain in the environment "lst-school-2022-01".

Anytime you want to use this environment in a new terminal, you have to activate it using:
```
conda activate lst-school-2022-01
```

### Instructions for the MAGIC+LST1 analysis session

Due to different versions of `ctapipe` (v0.12 against v0.11) used for this session, we need to work with a different `conda` environment.

If you want to work at the IT container, simply activate the `lst-school-2022-01-magic-lst1` environment:

```bash
conda activate lst-school-2022-01-magic-lst1
```

and run the notebooks from there.

If you want to run the examples locally (recommended), you have to create a separate environment. Please do:

```bash
git clone https://github.com/cta-observatory/magic-cta-pipe.git
cd magic-cta-pipe
conda env create -n lst-school-2022-01-magic-lst1 -f environment.yml
conda activate lst-school-2022-01-magic-lst1
pip install .
```

Then follow the notebooks that are available in this repo (directory `lst1_magic`).

The data used for the MAGIC+LST1 session are available at the IT container, see below. So you just need to synchronize again in order to download them (around 500 MB).


### Downloading the LST data for the exercises

We collected a subset of LST data that you should copy to your computer
and that is needed to run the notebooks locally.

To download the data, run this command from the base directory of this repository:

```
rsync -a --info=progress2 cp01:/fefs/aswg/workspace/analysis-school-2022/ data/
```

If you are on macOS, either install a more recent rsync using brew or leave out the --info option.

