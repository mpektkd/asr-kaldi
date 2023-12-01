#!/usr/bin/env bash

#download slp_lab2_data
echo "downloadind 'slp_lab2_data'"
sudo wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1M5p17VySiWANsKfuZyaXs9i2vPer0mTd' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1M5p17VySiWANsKfuZyaXs9i2vPer0mTd" -O slp_lab2 && rm -rf /tmp/cookies.txt

#unzip file and place it in correct directory
unzip slp_lab2 -d slp_data
mv slp_data/slp_lab2_data .
rm -r slp_data
rm -f slp_lab2

#create subdirectories
mkdir data data/train data/dev data/test
#run scripts for stpes 3.4.3 and 3.4.4 from preparation 
python3 step_3.4.3.py
python3 step_3.4.4.py

#rename the files to follow rule for the file names
rm -r data/train/text data/dev/text data/test/text
cp data/train/text2 data/train/text
cp data/dev/text2 data/dev/text
cp data/test/text2 data/test/text
rm -r data/train/text2 data/dev/text2 data/test/text2

#step 4.1.1 --> files cmd.sh, path.sh were included in submitted files

#step 4.1.2
#make sodt links with util and steps in our working directory
ln -s ../wsj/s5/steps steps
ln -s ../wsj/s5/utils utils

#step 4.1.3
mkdir local
cd ./local
ln -s ../../wsj/s5/steps/score_kaldi.sh score.sh
cd ..

#step 4.1.4
mkdir conf
echo "placing mfcc.conf from kaldi/egs/usc to kaldi/egs/usc/conf"
cp mfcc.conf conf/mfcc.conf
rm -r mfcc.conf

#step 4.1.5
mkdir data/lang data/local data/local/dict data/local/lm_tmp data/local/nist_lm