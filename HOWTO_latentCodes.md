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

Run either *realZfakeA.sh* or *realZfakeB.sh* on the cluster, examplarily for A:
```bash
cd $HOME/artifact_regularization_for_gans
bsub -I -R "rusage[mem=8192,ngpus_excl_p=1]" < genforce/scripts/realZfakeA.sh
```

### Download the results

After the job is finished, download the result files *lat.p*, *fak.p*, *los.p* to your computer with a local shell (not logged into your leonhard account):
```bash
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/lat.p /<localPath>/lat.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/fak.p /<localPath>/fak.p
scp  <nethz>@login.leonhard.ethz.ch:artifact_regularization_for_gans/los.p /<localPath>/los.p
```
where you should replace \<nethz\> and \<localPath\>
