# 2022_01_lstchain_school

LST analysis school January 2022

### For doing the hands-on exercises in your own computer (recommended option):

#### Download the school repository to your computer:

```
git clone https://github.com/cta-observatory/2022_01_lstchain_school.git
```

#### Install

- You will need to install [mamba](https://github.com/conda-forge/miniforge#mambaforge) 


then, in the folder where you cloned your repo:

```
mamba env create -f environment.yml
mamba activate lst-school-2022-01
```

This will install and activate the version 0.8.4 of lstchain in the environment "lst-school". 

#### Downloading the LST data for the exercises

We collected a subset of LST data that you should copy to your computer
and that is needed to run the notebooks locally.

To download the data, run this command from the base directory of this repository:

```
rsync -a --info=progress2 cp01:/fefs/aswg/workspace/analysis-school-2022/ data/
```

If you are on macOS, either install a more recent rsync using brew or leave out the --info option.

