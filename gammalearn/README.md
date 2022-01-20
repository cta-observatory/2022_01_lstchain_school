# Install  instruction:

```
mamba env create -f environment_glearn.yml
conda activate glearn
```

You may replace mamba with conda.


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
