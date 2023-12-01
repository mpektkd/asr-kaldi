# asr-kaldi
NTUA Projects

HOW TO RUN SCIPTS:
  In the folder called 'scripts', there are all the files needed for the implementation of the task. The steps are the following:
					 
    1) Creation of the subdirectory inside 'kaldi' root folder that we will work i.e. kaldi/egs/usc
    2) Move all the files from folder 'scripts' into kaldi/egs/usc 
    3) Env creation:
        a) chmod +x create_env.sh
        b) ./create_env.sh
    4) Run the experiments
        a) chmod +x run_scripts.sh
        b) ./run_scripts.sh

*The above steps are essential, as they will create the appropriate soft links.