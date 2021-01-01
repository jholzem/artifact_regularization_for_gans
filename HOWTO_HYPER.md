# HYPERPARAMETER TUNING

## Getting started

Make sure to be up-to-date on the Git branch 'fixedLatents':
```bash
git checkout fixedLatents
git pull
```

Configure ETH proxy:
```bash
cd $HOME
module load eth_proxy
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
***In addition, make sure that no unnecessary .zip files with data / weights are stored anywhere.***

If not done already, re-install the conda environment (new enviroment more storage efficient):
```bash
cd $HOME/artifact_regularization_for_gans
conda deactivate
conda remove -n DL --all
conda env create -f environment.yml
conda activate DL
module load eth_proxy
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
bash scripts/hyper.sh "1 5e-1 1e-1 5e-2" "cos" "1e-3 1e-4 1e-5 1e-6" "mschaller"
```

### Strategy

Lambda | 1 | 0.5 | 0.1 | 0.05 | 0.01 | 0.005 | 0.001 | 0.0005

Metric | '2' | 'cos'

Learning rate | 1e-3 | 1e-4 | 1e-5 | 1e-6

Maximum number of epochs: 20, with checkpoints after every epoch.


### Division

Amir:   Lambda: 1, 5e-1, 1e-1, 5e-2,         Metric: 2   LR: all,    nethz: hadzica

Jonas:  Lambda: 1e-2, 5e-3, 1e-3, 5e-4,      Metric: 2   LR: all,    nethz: jholzem

Max:    Lambda: 1, 5e-1, 1e-1, 5e-2,          Metric: cos   LR: all,    nethz: mschaller

Oli:    Lambda: 1e-2, 5e-3, 1e-3, 5e-4,       Metric: cos   LR: all,    nethz: steffeol

## Download the results

To download the .txt accuracy files and log files, execute the following commands from a local terminal:
```bash
scp <nethz>@login.leonhard.ethz.ch:/cluster/scratch/<nethz>/results/*.txt <localFolder>
cd <repo>/genforce
bash download_log.sh <LAMBDAS> <METRIC> <LRS> <NETHZ> <LOCALFOLDER>
```

example:
```bash
bash download_log.sh "1 5e-1 1e-1 5e-2" "cos" "1e-3 1e-4 1e-5 1e-6" "mschaller" "/Users/max/Desktop/log"
```

