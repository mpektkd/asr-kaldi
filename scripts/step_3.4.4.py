#!/usr/bin/python

import re
from string import digits

def clean_text(s):
    s = s.strip()  # strip leading / trailing 
    s = s.strip('\t')   # remove \t
    s = s.lower()  # convert to lowercase
    s = re.sub("\s+", " ", s)  # strip multiple whitespace
    s = re.sub(r"[^a-z\s'\s]", " ", s)  # keep only lowercase letters, ' and spaces
    return s

def pron(s):
    s = clean_text(s)   # convert phrase in acceptable form
    w = s.split()   # split into each word
    r = []
    for word in w:
        r.append(dic[word]) # find pronunciation for each word
    
    ph = ' '
    ph = ph.join(r) # make all into one string
    return ph

# create a dictionary with form [word]=pronunciation
dic = {}
with open('slp_lab2_data/lexicon.txt', 'r') as f:
  for l in f:
    k,v=l.split(' ',1)    # split in word, pronunciation
    k = clean_text(k)     # remove all unnecessary symbols 
    k = k.strip()         # strip leading / trailing
    k = k.strip('\t')     # remove \t
    v = clean_text(v)     # remove all unnecessary symbols
    v = v.strip()         # strip leading / trailing
    v = v.strip('\n')     #remove \n
    dic[k]=v

# do it for train set
with open('data/train/text2','w') as text2:
    with open('data/train/text','r') as text:
        for line in text:
            label, phrase = line.split(' ',1)    # split each line in label and phrase
            phrase = phrase.strip('\n')
            text2.write('{} sil {} sil\n'.format(label, pron(phrase)))

# do it for validation set
with open('data/dev/text2','w') as text2:
    with open('data/dev/text','r') as text:
        for line in text:
            label, phrase = line.split(' ',1)    # split each line in label and phrase
            phrase = phrase.strip('\n')
            text2.write('{} sil {} sil\n'.format(label, pron(phrase)))

# do it for test set
with open('data/test/text2','w') as text2:
    with open('data/test/text','r') as text:
        for line in text:
            label, phrase = line.split(' ',1)    # split each line in label and phrase
            phrase = phrase.strip('\n')
            text2.write('{} sil {} sil\n'.format(label, pron(phrase)))