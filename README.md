# 2022_01_lstchain_school

LST analysis school January 2022

Download the repository to your computer:

```
git clone https://github.com/cta-observatory/2022_01_lstchain_school.git
```

## Install

- You will need to install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (recommended) or [anaconda](https://www.anaconda.com/distribution/#download-section) first. 


### In your computer

```
conda env create -n -f environment.yml
conda activate lst-school
```

### In the IT cluster

```
source /fefs/aswg/software/conda/etc/profile.d/conda.sh
conda env create -n -f environment.yml
conda activate lst-school
```


## LST data

We collected a subset of LST data that you should copy to your computer
and that is needed to run the notebooks locally.

To download the data, run this command from the base directory of this repository:

```
rsync -a --info=progress2 ctan-cp01:/fefs/aswg/workspace/analysis-school-2022/ data/
```
