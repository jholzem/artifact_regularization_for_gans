#!/usr/bin/env bash

# weights for CNNDetection
wget https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=0 -O CNNDetection/weights/blur_jpg_prob0.1.pth

# weights for StyleGAN
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=18hkGLc_0WxzQvnQiU2NDy5Q9P5N3EBtX' -O idinvert_pytorch/models/pretrain/vgg16.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd" -O idinvert_pytorch/models/pretrain/styleganinv_ffhq256_encoder.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C" -O idinvert_pytorch/models/pretrain/styleganinv_ffhq256_generator.pth && rm -rf /tmp/cookies.txt

# download FFHQ images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS" -O data/FFHQ_256.zip && rm -rf /tmp/cookies.txt
unzip -q data/FFHQ_256.zip -d data
rm -r data/__MACOSX
rm data/FFHQ_256.zip

# download pairs of FFHQ images and latent codes
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1xuXvFYXcm01Z1OBcd8BhSeK7bEIwZk7-" -O data/real_latent.zip && rm -rf /tmp/cookies.txt
unzip -q data/real_latent.zip -d data
rm -r data/__MACOSX
rm data/real_latent.zip
