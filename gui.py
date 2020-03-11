import os
from tkinter import *
from PIL import ImageTk, Image

root = Tk()
root.title('Speech App')
root.iconbitmap('D:/level4_sem1/project_interim2/speech-recognition-neural-network-master/images/chatbot.ico')
root.geometry("400x600")
# root.configure(bg="#617C58")


def hello():
    path = "D:/level4_sem1/project_interim2/speech-recognition-neural-network-master"
    os.chdir(path)
    cmd1 = "python save_wav2.py"
    cmd2 = "python predict_run.py"
    cmd3 = "python predict_intent.py"
    os.system(cmd1)
    os.system(cmd2)
    os.system(cmd3)


my_img = ImageTk.PhotoImage(Image.open("D:/level4_sem1/project_interim2/speech-recognition-neural-network-master/images"
                                       "/innovation-brain.jpg"))
my_label = Label(image=my_img)

my_label.grid(row=2, column=0)

myButton = Button(root, text="Speak", padx=40, pady=10, bg="#ADD8E6", command=hello)
myButton.grid(row=20, column=0)

root.mainloop()
