#!/usr/bin/env bash

#step 4.2.1
python3 step_4.2.1.py

#step 4.2.2
source path.sh
ln -s ../../tools/irstlm/scripts/build-lm.sh build-lm.sh
./build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_train_unigram.ilm.gz
./build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_train_bigram.ilm.gz

#step 4.2.3
compile-lm data/local/lm_tmp/lm_train_unigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_ug.arpa.gz
compile-lm data/local/lm_tmp/lm_train_bigram.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > data/local/nist_lm/lm_phone_bg.arpa.gz

#step 4.2.4
utils/prepare_lang.sh data/local/dict "<oov>" data/local/lang data/lang

#step 4.2.5
sort data/train/wav.scp -o data/train/wav.scp
sort data/train/text -o data/train/text
sort data/train/utt2spk -o data/train/utt2spk
sort data/dev/wav.scp -o data/dev/wav.scp
sort data/dev/text -o data/dev/text
sort data/dev/utt2spk -o data/dev/utt2spk
sort data/test/wav.scp -o data/test/wav.scp
sort data/test/text -o data/test/text
sort data/test/utt2spk -o data/test/utt2spk

#step 4.2.6
utils/utt2spk_to_spk2utt.pl data/train/utt2spk > data/train/spk2utt
utils/utt2spk_to_spk2utt.pl data/dev/utt2spk > data/dev/spk2utt
utils/utt2spk_to_spk2utt.pl data/test/utt2spk > data/test/spk2utt

#step 4.2.7 , timit was includeed in the submitted files
chmod +x timit_format_data.sh
./timit_format_data.sh

#question 1 
mkdir data/local/lm_tmp
source path.sh
./build-lm.sh -i data/local/dict/lm_train.text -n 1 -o data/local/lm_tmp/lm_train_unigram.ilm.gz
./build-lm.sh -i data/local/dict/lm_train.text -n 2 -o data/local/lm_tmp/lm_train_bigram.ilm.gz

compile-lm data/local/lm_tmp/lm_train_unigram.ilm.gz -eval=data/dev/text
compile-lm data/local/lm_tmp/lm_train_unigram.ilm.gz -eval=data/test/text
compile-lm data/local/lm_tmp/lm_train_bigram.ilm.gz -eval=data/dev/text
compile-lm data/local/lm_tmp/lm_train_bigram.ilm.gz -eval=data/test/text

#step 4.3
steps/make_mfcc.sh data/train
steps/make_mfcc.sh data/dev
steps/make_mfcc.sh data/test

steps/compute_cmvn_stats.sh data/train
steps/compute_cmvn_stats.sh data/dev
steps/compute_cmvn_stats.sh data/test

#question 3
feat-to-len scp:data/train/feats.scp ark,t:data/train/feats.lengths
feat-to-dim scp:data/train/feats.scp -

#step 4.4.1
steps/train_mono.sh data/train data/lang exp/mono

#step 4.4.2
utils/mkgraph.sh data/lang_phones_ug exp/mono exp/mono/ug_graph
utils/mkgraph.sh data/lang_phones_bg exp/mono exp/mono/bg_graph

#step 4.4.3
steps/decode.sh exp/mono/ug_graph data/dev exp/mono/decode_dev_ug
steps/decode.sh exp/mono/ug_graph data/test exp/mono/decode_test_ug
steps/decode.sh exp/mono/bg_graph data/dev exp/mono/decode_dev_bg
steps/decode.sh exp/mono/bg_graph data/test exp/mono/decode_test_bg

#step 4.4.4
local/score.sh data/test exp/mono/ug_graph exp/mono/decode_test_ug > exp_mono_ug/decode_test/scoring_kaldi/best_wer
local/score.sh data/test exp/mono/bg_graph exp/mono/decode_test_bg > exp_mono_bg/decode_test/scoring_kaldi/best_wer	

#step 4.4.5
steps/align_si.sh data/train data/lang exp/mono exp/mono_ali
steps/train_deltas.sh 2000 10000 data/train data/lang exp/mono_ali exp/tri

utils/mkgraph.sh data/lang_phones_ug exp/tri exp/tri/ug_graph
utils/mkgraph.sh data/lang_phones_bg exp/tri exp/tri/bg_graph

steps/decode.sh exp/tri/ug_graph data/dev exp/tri/decode_dev_ug
steps/decode.sh exp/tri/ug_graph data/test exp/tri/decode_test_ug
steps/decode.sh exp/tri/bg_graph data/dev exp/tri/decode_dev_bg
steps/decode.sh exp/tri/bg_graph data/test exp/tri/decode_test_bg

#step 4.5.1
steps/align_si.sh data/train data/lang exp/tri exp/tri_ali_train
steps/align_si.sh data/dev data/lang exp/tri exp/tri_ali_dev
steps/align_si.sh data/test data/lang exp/tri exp/tri_ali_test

#step 4.5.(2,3,4,5)
chmod +x decode_dnn.sh
chmod +x run_dnn.sh
./run_dnn.sh

