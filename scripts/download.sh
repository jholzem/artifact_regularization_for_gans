#!/usr/bin/env bash

# weights for CNNDetection
wget https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=0 -O CNNDetection/weights/blur_jpg_prob0.1.pth

# weights for StyleGAN
wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=18hkGLc_0WxzQvnQiU2NDy5Q9P5N3EBtX' -O idinvert_pytorch/models/pretrain/vgg16.pth
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1hOszkKUp1faDgMpxSg_HNl4pxDM2ALOd" -O idinvert_pytorch/models/pretrain/styleganinv_ffhq256_encoder.pth && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Dz7AnbhPfHVMIBQTWoSqiCWhqJWq-8_C" -O idinvert_pytorch/models/pretrain/styleganinv_ffhq256_generator.pth && rm -rf /tmp/cookies.txt

# download triplets of real/generated FFHQ images and latent codes
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1fQRBLu6TYuHmUb_UKxCc5ktn5iAUpZZm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1fQRBLu6TYuHmUb_UKxCc5ktn5iAUpZZm" -O data/real_latent_fake.zip && rm -rf /tmp/cookies.txt
unzip -q data/real_latent_fake.zip -d data
rm -r data/__MACOSX
rm data/real_latent_fake.zip
