#!/usr/bin/python

# create a dictionary with form [number]=phrase
dic = {}  
i=1
with open('slp_lab2_data/transcription.txt', 'r') as f:
    for line in f:
      line = line.strip("\n")
      dic[format(i, '03')] = line   #number in form 'nnn' 
      i+=1

# create files for train set
with open('data/train/uttids','w') as uttids:
    with open('data/train/utt2spk','w') as utt2spk:
        with open('data/train/wav.scp','w') as wav:
            with open('data/train/text','w') as text:
                with open('slp_lab2_data/filesets/train_utterances.txt','r') as f:
                    for l in f:
                      l = l.strip("\n")
                      label1, label2, person, number = l.split("_")
                      uttids.write('{}\n'.format(l))
                      utt2spk.write('{} {}\n'.format(l,person))
                      wav.write('{} slp_lab2_data/wav/{}/{}.wav\n'.format(l,person,l))
                      text.write('{} {}\n'.format(l,dic[number]))

# create files for validation set
with open('data/dev/uttids','w') as uttids:
    with open('data/dev/utt2spk','w') as utt2spk:
        with open('data/dev/wav.scp','w') as wav:
            with open('data/dev/text','w') as text:
                with open('slp_lab2_data/filesets/validation_utterances.txt','r') as f:
                    for l in f:
                      l = l.strip("\n")
                      label1, label2, person, number = l.split("_")
                      uttids.write('{}\n'.format(l))
                      utt2spk.write('{} {}\n'.format(l,person))
                      wav.write('{} slp_lab2_data/wav/{}/{}.wav\n'.format(l,person,l))
                      text.write('{} {}\n'.format(l,dic[number]))

# create files for test set
with open('data/test/uttids','w') as uttids:
    with open('data/test/utt2spk','w') as utt2spk:
        with open('data/test/wav.scp','w') as wav:
            with open('data/test/text','w') as text:
                with open('slp_lab2_data/filesets/test_utterances.txt','r') as f:
                    for l in f:
                      l = l.strip("\n")
                      label1, label2, person, number = l.split("_")
                      uttids.write('{}\n'.format(l))
                      utt2spk.write('{} {}\n'.format(l,person))
                      wav.write('{} slp_lab2_data/wav/{}/{}.wav\n'.format(l,person,l))
                      text.write('{} {}\n'.format(l,dic[number]))