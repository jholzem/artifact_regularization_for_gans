## Installation

Clone the GitHub repository in the cluster's $SCRATCH folder and update its cited submodules
```bash
cd $SCRATCH
git clone https://github.com/hadzica/artifact_regularization_for_gans.git
git submodule update --remote
```

Install conda using the following commands
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```

Create and activate the conda environment
```bash
cd $SCRATCH/artifact_regularization_for_gans
conda env create -f environment.yml
conda activate DL
```

Restart the shell after the environment has been installed. Make sure that you always activate `DL` before running scripts, files etc.

Download pre-trained weights and FFHQ data
```bash
cd $SCRATCH/artifact_regularization_for_gans
bash scripts/download.sh
```

## Preparation

As described in the report, we compute the optimized latent codes of the in-domain GAN inversion prior to the actual fine-tuning of the StyleGAN generator. The following steps can be followed to reproduce the results that have been downloaded already in the previous step.

### Create triplets of real images, latent codes, and fake images

Run *realZfake.sh* on the cluster:
```bash
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -W 24:00 scripts/realZfake.sh
```

### Download the results

After the job is finished, download the result files *lat\<X\>.p*, *fak\<X\>.p*, *los\<X\>.p*
```bash
scp  <nethz>@login.leonhard.ethz.ch:/cluster/scratch/<nethz>/artifact_regularization_for_gans/lat<X>.p /<localPath>/lat<X>.p
scp  <nethz>@login.leonhard.ethz.ch:/cluster/scratch/<nethz>/artifact_regularization_for_gans/fak<X>.p /<localPath>/fak<X>.p
scp  <nethz>@login.leonhard.ethz.ch:/cluster/scratch/<nethz>/artifact_regularization_for_gans/los<X>.p /<localPath>/los<X>.p
```
where you should replace \<nethz\>, \<X\> and \<localPath\>.

### Post-process the results

Convert the .p files into .png files containing real and fake images and .csv files containing latent codes with
```bash
TODO
```

## Fine-tuning the StyleGAN generator

Configure ETH proxy:
```bash
cd $SCRATCH
module load eth_proxy
```

Start training:
```bash
cd $SCRATCH/artifact_regularization_for_gans
bash scripts/train_eval.sh <LAMBDA> <METRIC> <LR> <NETHZ>
```

...
