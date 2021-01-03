## Installation

Clone the GitHub repository in the cluster's $SCRATCH folder and update its cited submodules
```bash
cd $SCRATCH
git clone --branch submission https://github.com/hadzica/artifact_regularization_for_gans.git
cd artifact_regularization_for_gans
git submodule update --init
```

Install conda using the following commands
```bash
cd $SCRATCH
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Specify the installation path when asked to. Restart the shell after conda has successfully been installed.

Create and activate the conda environment
```bash
cd $SCRATCH/artifact_regularization_for_gans
conda env create -f environment.yml
conda activate DL
```
Make sure that you always activate `DL` before running scripts, files etc.

Download pre-trained weights, FFHQ images, and optimized latent codes
```bash
bash scripts/download.sh
```


## Fine-tuning the StyleGAN generator

Load ETH proxy:
```bash
module load eth_proxy
```

Start training and subsequent evaluation with CNNDetection:
```bash
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 4:00 scripts/train_eval.sh 1e-3 2 1e-6 10000 <NETHZ>
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 4:00 scripts/train_eval.sh 3e-1 cos 1e-6 10000 <NETHZ>
```


## Appendix

<details><summary>Create triplets of real images, latent codes, and fake images</summary>
<p>

As described in the report, we compute the optimized latent codes of the in-domain GAN inversion prior to the actual fine-tuning of the StyleGAN generator. The following steps can be followed to reproduce the results that have been downloaded already in the previous step.

Download FFHQ data:
Download pre-trained weights and FFHQ data
```bash
bash scripts/download_FFHQ.sh
```

Run *realZfake.sh* on the cluster:
```bash
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -W 120:00 scripts/realZfake.sh
```

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

</p>
</details>
...
