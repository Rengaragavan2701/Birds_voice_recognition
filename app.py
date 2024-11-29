from flask import Flask, render_template, request, redirect, url_for
import librosa
import glob
import os
import time
import numpy as np
import pandas as pd

UPLOAD_FOLDER = 'static/file/'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

emotions={
  '01':'amebit',
  '02':'amerco',
  '03':'ameavo',
  '04':'amegfi',
  '05':'aldfly'
}

tess_emotions=['aldfly','amebit','amerco','ameavo','amegfi']
ravdess_emotions=['amebit','amerco','aldfly', 'ameavo','amegfi',]
observed_emotions = ['amegfi','aldfly','ameavo','amebit','amerco']

def extract_feature(file_name, mfcc):
    X, sample_rate = librosa.load(os.path.join(file_name), res_type='kaiser_fast')
    if mfcc:
        mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result=np.hstack((mfccs))
        return result
    else:
        return None

def dataset_options():
    # choose datasets
    ravdess = True
    tess = True
    ravdess_speech = False
    ravdess_song = False
    data = {'ravdess':ravdess, 'ravdess_speech':ravdess_speech, 'ravdess_song':ravdess_song, 'tess':tess}
    print(data)
    return data

def load_data(path, test_size=0.2): 
    x,y=[],[]
    
    mfcc = True
    
    data = dataset_options()
    paths = []
    if data['ravdess']:
        paths.append(path)
        
    for path in paths:
        for file in glob.glob(path):
            file_name=os.path.basename(file)
            emotion=emotions[file_name.split("-")[2]]
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc)
            x.append(feature)
            y.append(emotion)
    if data['tess']:
        for file in glob.glob(path):
            file_name=os.path.basename(file)
            emotion=file_name.split("-")[:-4]
            if emotion == 'ameavo':
                emotion = 'ameavo'
            if emotion not in observed_emotions:
                continue
            feature=extract_feature(file, mfcc)
            x.append(feature)
            y.append(emotion)
    return {"X":x,"y":y}


@app.route('/')
def login():
    return render_template('login.html')

@app.route('/validate', methods = ['POST','GET'])
def validate():
    if request.method == 'POST':
        if request.form.get('username') == 'admin' and request.form.get('password') == '1234':
            return render_template('index.html')
        else:
            return render_template('login.html', msg = 'Invalid Data')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method == 'POST':
        starting_time = time.time()
        # data = pd.read_csv("static/RAVTESS_MFCC_Observed.csv")
        # print("data loaded in " + str(time.time()-starting_time) + "ms")
        # print(data.head())
        speech_file = request.files['speech_file']
        path = os.path.join(app.config['UPLOAD_FOLDER'], speech_file.filename)
        speech_file.save(path)
        start_time = time.time()
        Trial_dict = load_data(path, test_size = 0.4)
        print("--- Data loaded. Loading time: %s seconds ---" % (time.time() - start_time))
        X = pd.DataFrame(Trial_dict["X"])
        y = pd.DataFrame(Trial_dict["y"])
        res=y[0]
        fin=res[0]
        print(res[0])

        return render_template('wrong.html', msg = fin)

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == "__main__":
    app.run(debug=True, port=7001)
