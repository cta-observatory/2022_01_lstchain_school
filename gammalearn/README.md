# Install  instruction:

```
mamba env create -f environment_glearn.yml
conda activate glearn
```
You may replace mamba with conda.


## Test install

Running `gammalearn --version` should return 0.8


## Known issues

### M1 chip or no GPU
If you have a mac with the M1 chip or don't plan to run on GPU, simply remove the `cudatoolkit` line from the environment file.



# Working at La Palma


If you do NOT have your own installation of conda / miniconda:
```
 source /fefs/aswg/workspace/gammalearn/software/source_conda.sh
 conda activate glearn-v0.8
```
If you have your own conda / miniconda installation:
```
 conda deactivate
 source /fefs/aswg/workspace/gammalearn/software/source_conda.sh
 hash -r
 conda activate
 conda activate glearn-v0.8
 ```
