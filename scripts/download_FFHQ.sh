#!/usr/bin/env bash

# download FFHQ images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TVfX2dy39agfUfRjoryLnDG9kbB4jerS" -O data/FFHQ_256.zip && rm -rf /tmp/cookies.txt
unzip -q data/FFHQ_256.zip -d data
rm -r data/__MACOSX
rm data/FFHQ_256.zip
