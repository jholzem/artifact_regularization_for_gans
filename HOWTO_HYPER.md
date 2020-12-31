# HYPERPARAMETER TUNING

## Getting started

Make sure to be up-to-date on the Git branch 'fixedLatents':
```bash
git checkout fixedLatents
git pull
```

Download the pairs of real FFHQ .png images and their corresponding optimized latent codes:
```bash
# mkdir $SCRATCH/data
cd $SCRATCH/data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-" -O real_latent.zip && rm -rf /tmp/cookies.txt
unzip -q real_latent.zip
rm -r __MACOSX
rm real_latent.zip
```

Download the weights for CNNDetection:
```bash
cd $HOME/artifact_regularization_for_gans/genforce
bash weights/download_weights.sh
```

If not done already, remove the 'data' folder from 'genforce'

```bash
cd $HOME/artifact_regularization_for_gans/genforce
rm -r data
```

If not done already, re-install the conda environment (new enviroment more storage efficient):
```bash
cd $HOME/artifact_regularization_for_gans
conda deactivate
conda remove -n DL --all
conda env create -f environment.yml
conda activate DL
```

## Perform Search

Make sure to be up-to-date on the Git branch 'fixedLatents':
```bash
git checkout fixedLatents
git pull
```

Check that you have sufficient storage in your $HOME directory (16 GB is the limit):
```bash
cd $HOME/..
du -sh <NETHZ>
```

Clean up your scratch folder (always before executing hyper.sh, see below):
```bash
# mkdir $SCRATCH/results
cd $SCRATCH/results
mkdir bak
find . -maxdepth 1 -name "*.pth" -exec mv "{}" ./bak \;
find . -maxdepth 1 -name "*.txt" -exec mv "{}" ./bak \;
```

Start hyperparameter search:
```bash
cd $HOME/artifact_regularization_for_gans/genforce
bash scripts/hyper.sh <LAMBDAS> <METRIC> <LRS> <NETHZ>
```
example:

```bash
bash scripts/hyper.sh "1 1e-1 5e-2 " "cos" "1e-3 1e-4 1e-5 1e-6 1e-7 1e-8" "mschaller"
```

### Strategy

Lambda | 1 | 0.5 | 0.1 | 0.05 | 0.01 | 0.005 

Metric | '2' | 'cos'

Learning rate | 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 | 1e-8

Maximum number of epochs: 20, with checkpoints after every epoch.


### Division

Amir:   Lambda: 1, 5e-1, 1e-1,          Metric: 2   LR: all,    nethz: hadzica

Jonas:  Lambda: 5e-2, 1e-2, 5e-3,       Metric: 2   LR: all,    nethz: jholzem

Max:    Lambda: 1, 5e-1, 1e-1,          Metric: cos   LR: all,    nethz: mschaller

Oli:    Lambda: 5e-2, 1e-2, 5e-3,       Metric: cos   LR: all,    nethz: steffeol
