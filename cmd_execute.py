import os

path = "D:/level4_sem1/project_interim2/speech-recognition-neural-network-master"
os.chdir(path)
cmd1 = "python save_wav2.py"
cmd2 = "python predict_run.py"
cmd3 = "python predict_intent.py"
os.system(cmd1)
os.system(cmd2)
os.system(cmd3)
