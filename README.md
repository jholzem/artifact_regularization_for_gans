## Installation

Clone github repo
```bash
git clone https://github.com/hadzica/artifact_regularization_for_gans.git
```

For easier handling, we use conda environments. Install conda using the following commands
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Inside the artifact...-folder, run
```bash
conda env create -f environment.yml
conda activate DL
```

Restart the shell after the environment has been installed. Make sure that you always activate `dl_env` before running scripts, files etc.

Additionally, the weights of the pretrained models need to be added. Create a directory such that you have the following structure: `artifact_regularization_for_gans/genforce/models/pretrain`. In this directory download the files using:

```bash
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=18hkGLc_0WxzQvnQiU2NDy5Q9P5N3EBtX' -O vgg16.pth
```
```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd" -O styleganinv_ffhq256_encoder.pth && rm -rf /tmp/cookies.txt
```
```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C" -O styleganinv_ffhq256_generator.pth && rm -rf /tmp/cookies.txt
```

## How to optimize latent codes on the leonhard cluster

### Getting started

Download the FFHQ **.png** images to the folder structure on the cluster:
```bash
cd $HOME/artifact_regularization_for_gans/genforce/data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS" -O FFHQ_256.zip && rm -rf /tmp/cookies.txt
unzip FFHQ_256.zip
rm -r __MACOSX
```

### Run the job

Run either *realZfake.sh* on the cluster:
```bash
cd $HOME/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -W 24:00 scripts/realZfake.sh
```

### Download the results

After the job is finished, download the result files *lat\<X\>.p*, *fak\<X\>.p*, *los\<X\>.p* to your computer with a local shell (not logged into your leonhard account), for A: A00-A02, for B: A03-A05, for C: A06-A08, for D: A09-A10, here shown for A:
```bash
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/lat<X>.p /<localPath>/lat<X>.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/fak<X>.p /<localPath>/fak<X>.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/los<X>.p /<localPath>/los<X>.p
```
where you should replace \<nethz\> and \<localPath\>. Finally, please upload all the .p files to the folder 'p' on our Google Drive.

## Training

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

Start training:
```bash
cd $HOME/artifact_regularization_for_gans
bash scripts/train_eval.sh <LAMBDA> <METRIC> <LR> <NETHZ>
```


