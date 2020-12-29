## How to train with fixed latent codes on the leonhard cluster

### Getting started

Follow the tutorial *HOWTO_leonhard.md* up to the point where you have downloaded all the pretrained weights. Then, download the pairs of real FFHQ .png images and their corresponding optimized latent codes:
```bash
cd $HOME/artifact_regularization_for_gans/genforce/data
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-" -O real_latent.zip && rm -rf /tmp/cookies.txt
unzip -q real_latent.zip
rm -r __MACOSX
```

...

