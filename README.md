# 2022_01_lstchain_school

LST analysis school January 2022

Download the repository to your computer:

```
git clone https://github.com/cta-observatory/2022_01_lstchain_school.git
```

## Install

- You will need to install [mamba](https://github.com/conda-forge/miniforge#mambaforge) 


then, in the folder where you cloned your repo:

```
mamba env create -f environment.yml
mamba activate lst-school
```


## LST data

We collected a subset of LST data that you should copy to your computer
and that is needed to run the notebooks locally.

To download the data, run this command from the base directory of this repository:

```
rsync -a --info=progress2 ctan-cp01:/fefs/aswg/workspace/analysis-school-2022/ data/
```
