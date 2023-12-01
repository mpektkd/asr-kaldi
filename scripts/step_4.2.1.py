#!/usr/bin/python
import re

#create silence_phones.txt
with open('data/local/dict/silence_phones.txt', 'w') as f:
  f.write('sil\n')

#create optional_silence.txt
with open('data/local/dict/optional_silence.txt', 'w') as f:
  f.write('sil\n')

phones = [] # list to save all phones

with open('slp_lab2_data/lexicon.txt', 'r') as f:
  for l in f:
    phones_temp = []  # empty temporary list
    junk,ph=l.split(' ',1)
    ph = ph.strip() # strip leading / trailing
    ph = ph.strip('\n') # remove '\n'
    phones_temp = ph.split(' ',-1) # create a temporary list to save phones of each phrase
    phones_temp.sort()
    for p in phones_temp:
      #p = re.sub(r"[^a-z\s]","",p)  # keep only lowercase letters
      if p not in phones:
        phones.append(p)  # append all phonetics that are not contained

if 'sil' in phones:
  phones.remove('sil') # remove sil if it exists

phones.sort()   # sort all phones

#create nonsilence_phones.txt (it already contails 'sil' because it was on lexicon.txt)
with open('data/local/dict/nonsilence_phones.txt', 'w') as f:
  for p in phones:
    f.write('{}\n'.format(p))

phones.append('sil') # append sil in list so as to create lexicon.txt
phones.sort() # sort all phones

#create lexicon.txt
with open('data/local/dict/lexicon.txt', 'w') as f:
  for p in phones:
    f.write('{} {}\n'.format(p,p))


# add <s> and </s> for train text
with open('data/local/dict/lm_train.text','w') as f_write:
  with open('data/train/text', 'r') as f_read:
    for l in f_read:
      first,last=l.split(' ',1) #split in 2 parts
      last = last.strip('\n')
      f_write.write('<s> {} </s>\n'.format(last))  # add <s> and </s> 

# add <s> and </s> for dev text
with open('data/local/dict/lm_dev.text','w') as f_write:
  with open('data/dev/text', 'r') as f_read:
    for l in f_read:
      first,last=l.split(' ',1) #split in 2 parts
      last = last.strip('\n')
      f_write.write('<s> {} </s>\n'.format(last))  # add <s> and </s> 

# add <s> and </s> for test text
with open('data/local/dict/lm_test.text','w') as f_write:
  with open('data/test/text', 'r') as f_read:
    for l in f_read:
      first,last=l.split(' ',1) #split in 2 parts
      last = last.strip('\n')
      f_write.write('<s> {} </s>\n'.format(last))  # add <s> and </s> 

open('data/local/dict/extra_questions.txt','w') #create extra_questions.txt