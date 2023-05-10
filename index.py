from flask import Flask, render_template, Response
#from flask_ngrok import run_with_ngrok
import cv2
import numpy as np
import tensorflow as tf


model = tf.keras.models.load_model('model.h5')
LABELS = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,'Z':25}  #Labels and their corresponding integers

def recognize(img):
    img = np.resize(img, (28,28,1))
    img = np.expand_dims(img, axis=0)
    img = np.asarray(img)
    classes = model.predict(img)
    pred_id = tf.argmax(classes[0])
    #print(pred_id[0])
    for i in LABELS:
        if pred_id == LABELS[i]:
            p = LABELS[i]   
    return p



global output
output=""
#word=[]


cam = cv2.VideoCapture(0)       #video capture opbject for the build-in camera

def gen():                          #generator
    global img_name, char_op
    while(True):
        success, frame=cam.read()    
        rec_start=(100,160)
        rec_end=(420,560)
        rec_col=(255,0,0)
        frame=cv2.rectangle(frame,rec_start,rec_end,rec_col,thickness=2)   
        '''
        sucess: 
            boolean value. 
            returns true if python is able to capture video
        frame:
            a numpy array that represents the first image captured by the the VideoCamera
        '''
        crop_sign=frame[rec_start[1]:rec_end[1], rec_start[0]:rec_end[0]]
        img=cv2.cvtColor(crop_sign, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (5,5), 0)
        pred_img = cv2.resize(img, (28,28), interpolation=cv2.INTER_AREA)
        y_pred = recognize(pred_img)
        char_op = chr(y_pred + 65)
        cv2.rectangle(frame, (80,600), (680,680), (0,0,0), -1)
        cv2.putText(frame, "Predicted Sign: "+char_op, (100,660), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,0), 2) 

        if not success:
            print("Capture not Successful")
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
            # concat frame one by one and show result
            
            #yield lets the execution to continue and generates from until alive

def pr():
    global output
    output+=char_op
    return(output)

'''
==================================
||            FLASK              ||
||           ROUTES              ||
==================================

'''

app=Flask(__name__, template_folder='template')
#run_with_ngrok(app)  # Start ngrok when app is run

@app.route('/')  #landing
def index():
    return render_template("index.html")

@app.route('/video')
def video():
    return Response(gen(),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture')
def capture():
    return render_template("index.html", output=pr())

@app.route('/reset')
def reset():
    global output
    output=""
    return render_template("index.html", output=output)

@app.route('/del_last')
def del_last():
    global output
    output=output[:-1]
    return render_template("index.html", output=output)


if __name__=="__main__":
    app.run()