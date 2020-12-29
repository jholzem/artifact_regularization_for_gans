## How to optimize latent codes on the leonhard cluster

### Getting started

Follow the tutorial *HOWTO_leonhard.md* and make sure to download the FFHQ **.png** images to the folder structure on the cluster:
```bash
cd $HOME/artifact_regularization_for_gans/genforce/data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS" -O FFHQ_256.zip && rm -rf /tmp/cookies.txt
unzip FFHQ_256.zip
rm -r __MACOSX
```
Make sure that the GitHub clone is up-to-date:
```bash
git pull origin master
```

### Run the job

Run either *realZfakeA.sh*, *realZfakeB.sh*, *realZfakeC.sh* or *realZfakeD.sh* on the cluster, examplarily for A:
```bash
cd $HOME/artifact_regularization_for_gans
bsub -R "rusage[mem=8192,ngpus_excl_p=1]" -W 24:00 < genforce/scripts/realZfakeA.sh
```

### Download the results

After the job is finished, download the result files *lat\<X\>.p*, *fak\<X\>.p*, *los\<X\>.p* to your computer with a local shell (not logged into your leonhard account), for A: A00-A02, for B: A03-A05, for C: A06-A08, for D: A09-A10, here shown for A:
```bash
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/latA00.p /<localPath>/latA00.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/fakA00.p /<localPath>/fakA00.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/losA00.p /<localPath>/losA00.p

scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/latA01.p /<localPath>/latA01.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/fakA01.p /<localPath>/fakA01.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/losA01.p /<localPath>/losA01.p

scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/latA02.p /<localPath>/latA02.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/fakA02.p /<localPath>/fakA02.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/losA02.p /<localPath>/losA02.p
```
where you should replace \<nethz\> and \<localPath\>
