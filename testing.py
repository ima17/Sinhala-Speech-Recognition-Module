import json
import cv2
import playsound

from rasa_nlu.model import Interpreter
from gtts import gTTS

interpreter = Interpreter.load('./models/current/nlu')
f = open("D:/level4_sem1/project_interim2/output/transcriptions.txt", "r", encoding="utf-8")


def predict_intent(text):
    results = interpreter.parse(text)
    print(json.dumps({
        "intent": results["intent"]
    }, indent=2))
    print(results['intent']['name'])
    if results['intent']['name'] == "ask_currency_note_value":
        print("INTENT IS THERE")
        # tts = gTTS(text="මුදල් නෝට්ටුව ඉදිරිපත් කරන්න", lang='si')
        # tts.save("good.mp3")
        # os.system("start good.mp3")
        playsound.playsound('D:/level4_sem1/project_interim2/mp3/good.mp3', True)
        # camera_port = 0
        # camera = cv2.VideoCapture(camera_port)
        # return_value, image = camera.read()
        # cv2.imwrite("image.jpeg", image)
        # camera.release()
        # cv2.destroyAllWindows()

        # cv2.namedWindow("preview")
        # vc = cv2.VideoCapture(0)
        #
        # if vc.isOpened():  # try to get the first frame
        #     rval, frame = vc.read()
        # else:
        #     rval = False
        #
        # while rval:
        #     cv2.imshow("preview", frame)
        #     rval, frame = vc.read()
        #     key = cv2.waitKey(20)
        #     if key == 27:  # exit on ESC
        #         break
        # cv2.destroyWindow("preview")


    else:
        print("INTENT IS NOT THERE")
        # tts = gTTS(text="විධානය හදුනාගත නොහැක. නැවත උත්සහ කරන්න", lang='si') tts.save("good.mp3") os.system("start
        # good.mp3")
        playsound.playsound('D:/level4_sem1/project_interim2/mp3/bad.mp3', True)

    # text = input('Enter a message: ')


text = f.readline()
print(text)
predict_intent(text)
