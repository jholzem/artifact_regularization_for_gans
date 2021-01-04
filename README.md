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

Download pre-trained weights, and pairs of FFHQ images and their corresponding optimized latent codes as described in the report.
```bash
bash scripts/download.sh
```


## Fine-Tuning the StyleGAN Generator

Load ETH proxy:
```bash
module load eth_proxy
```

Start training and subsequent evaluation with CNNDetection, where you should specify your \<NETHZ\>:
```bash
mkdir $SCRATCH/results
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 0 2 1e-6 10000 <NETHZ>
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 1e-3 2 1e-6 10000 <NETHZ>
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 3e-1 cos 1e-6 10000 <NETHZ>
```


## Appendix

<details><summary>Create triplets of real images, latent codes, and fake images</summary>
<p>

As described in the report, we compute the optimized latent codes of the in-domain GAN inversion prior to the actual fine-tuning of the StyleGAN generator. The following steps can be followed to reproduce the results that have been downloaded already in the previous steps. All **Installation** steps should be finished beforehand.

Download FFHQ data:
```bash
cd $SCRATCH/artifact_regularization_for_gans
bash scripts/download_FFHQ.sh
```

Utilize in-domain GAN inversion to optimize latent codes for real FFHQ images and pass those through the StyleGAN generator to retrieve associated "fake" images.
```bash
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -W 120:00 scripts/realZfake.sh 11
```

</p>
</details>

<details><summary>Analyze dissimilarity of Fourier spectra of real/generated images</summary>
<p>

To determine the frequency range of interest, we analyze Fourier dissimilarity values for different truncations of the spectra of real and generated images. Since a Jupyter notebook is included in the subsequent steps, it might be convenient to follow these on a machine where you can open .ipynb files with a GUI. All **Installation** steps should be finished.

First, download the pairs of real and generated images:
```bash
bash scripts/download_FFHQ.sh
```

Add the conda environment to the notebook
```bash
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=DL
```

Select `DL` as kernel and follow the steps in `fourier_analysis.ipynb`

</p>
</details>
