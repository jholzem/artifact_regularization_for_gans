Recent research has shown that GAN-generated images are easy to detect for dedicated detection algorithms. They rely on artifacts in the Fourier representation of   the generated images. As the goal for image synthesis is to produce images which are indistinguishable from real ones, this is a major drawback. We propose a novel training framework which penalizes synthesis artifacts by computing a Fourier dissimilarity between synthesized and real images. In this work, we investigate in which bandwidth in the Fourier domain the artifacts occur and show that the accuracy of StyleGAN images being detected as fake can be reduced by using an appropriate training strategy.

To reproduce our results, please follow the subsequent tutorial. The code is tested for the use on the Leonhard cluster and therefore we recommend to run the code on the Leonhard cluster as well. If you wish to run the code on another machine, please reach out to us - mschaller@ethz.ch (a few file paths will need to be adjusted).

## Installation

Clone the GitHub repository in the Leonhard cluster's $SCRATCH folder and update its cited submodules
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
where you should specify the installation path when asked to and type `yes` when asked if the installer should intiliaze Miniconda3. Restart the shell after conda has been successfully installed.

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

Start training and subsequent evaluation with the fake detection network, where you should specify your \<NETHZ\>:
```bash
mkdir $SCRATCH/results
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 0 1e3 cos 1e-4 10000 <NETHZ>
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 1 1e3 cos 1e-5 10000 <NETHZ>
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 24:00 scripts/train_eval.sh 1 0 cos 1e-6 10000 <NETHZ>
```
Info: The first argument of `train_eval.sh` is the weight of the adversarial loss, while the second argument is the weight of the Fourier loss, for which the type of dissimilarity measure is chosen through the third argument. The fourth argument specifies the learning rate und the fifth argument denotes the number of images that are synthesized for determining the detection accuracy.

## Visualize the Results

```bash
cd $SCRATCH/artifact_regularization_for_gans
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 1:00 scripts/visualize.sh 0_1e3_cos_1e-4 17 1_1e3_cos_1e-5 10 1_0_cos_1e-6 5 <NETHZ>
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
bsub -R "rusage[mem=32768,ngpus_excl_p=1]" -W 120:00 < scripts/realZfake.sh
```

Info: the results will be saved into `data/reproduced`

</p>
</details>
