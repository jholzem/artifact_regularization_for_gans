## How to use the leonhard cluster

### Getting started

Connect to the cluster using (VPN required)
```bash
ssh username@login.leonhard.ethz.ch
```
On your personal computer, create ssh keys for advanced safety and enter password when prompted
```bash
ssh-keygen -t ed25519 -f $HOME/.ssh/id_ed25519_leonhard
```
Copy public key to cluster
```bash
ssh-copy-id -i $HOME/.ssh/id_ed25519_leonhard.pub username@login.leonhard.ethz.ch
```
Add the following two lines to your `$HOME/.ssh/config` file (on your personal computer):
```bash
Host login.leonhard.ethz.ch
IdentityFile ~/.ssh/id_ed25519_leonhard
```

### Preparing for operation

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
The other necessary files need to be downloaded to your own computer manually and uploaded using `scp` (Miscellaneous)

To download the ffhq-dataset, create a folder s.t. `genforce/data`. Then:
```bash
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wejdbBYespDiLLudtyiMHz_zqt0tPIot' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wejdbBYespDiLLudtyiMHz_zqt0tPIot" -O ffhq.zip && rm -rf /tmp/cookies.txt
```


### Miscellaneous

* The cluster uses Linux as its operating system. Terminal commands are just normal linux commands.
* To copy files from your personal computer to the cluster, use `scp filename username@login.leonhard.ethz.ch:`. The file is uploaded to your home directory. You can change its location using the `mv` command.
* Programmes and python-files are best run through shell scripts. The simplest way to run them is through the command `bsub < filename.sh`. There are many options to adjust the way the script is run. Sometimes these options are necessary to be specified in advance. For more information, please refer to https://scicomp.ethz.ch/wiki/Using_the_batch_system
