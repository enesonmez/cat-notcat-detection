from flask import Flask, render_template, flash, redirect, url_for, session, request, send_from_directory
from werkzeug.utils import secure_filename
import os
import random
from functools import wraps
import scipy
from PIL import Image
from scipy import ndimage
#import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv
import sys
sys.path.append('../')
from nn_model.nn_algorithm import *
from nn_model.model import file_parameter_read

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/upload'
app.secret_key = "catapp"
image = np.array([])

#Dosya uzantı kontrolü
def extension_control(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ['jpg','jpeg','png','bmp','ico','tiff','jfif']

@app.route("/")
def index():
    return render_template("index.html", predict=False)

@app.route("/prediction",methods=['POST', 'GET'])
def prediction():
    #Klasördeki resimleri siliyoruz.
    imagefiles = os.listdir(app.config['UPLOAD_FOLDER']) #Uzantıdaki dosyaların isimlerini getirir.
    if len(imagefiles) != 0:
        for file in imagefiles:
            os.remove(app.config['UPLOAD_FOLDER'] + '/' + file) #Dosya siler.
            
    #Post Metodu
    if request.method == 'POST':
        file = request.files['files']
        #Dosya seçilip seçilmediği veya boş isimli dosya gelmiş mi kontrol edilir.
        if file.filename == '':
            flash("Error!!! No File Selected","red")
            return redirect(url_for('index'))
        #Dosya kontrolü ve upload
        random1 = random.randint(20000,32000) 
        random2 = random.randint(20000,32000) 
        random3 = random.randint(20000,32000) # Resimlerde benzersiz isim oluşturulur.
        random4 = random.randint(20000,32000)                                 #Dosya güvenlik kontrolü
        newfile = str(random1) + str(random2) + str(random3) + str(random4) + secure_filename(file.filename) 
        if file and extension_control(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'],newfile))
            
            #Upload ettiğimiz resimde kedi olup olmadığını tahmin ediyoruz.
            parameter, conditional = file_parameter_read("../nn_model/parameter.txt")
            if conditional: # parameter.txt varsa parametreler okunup tahmin yapılır.
                global image
                fname = app.config['UPLOAD_FOLDER'] + '/' + newfile # İşlenecek görüntünün dosya yolu
                image = np.array(ndimage.imread(fname, flatten=False)) # Resim okunur ve matrise çevrilir
                image = image[:,:,0:3] # Resmin rgb matrisleri alınır.
                sha = image.shape
                image = scipy.misc.imresize(image, size=(64,64)).reshape(1,-1).T #Resim 64x64  boyutuna getirilir.
                image = image/255 # Tüm pikseller feature olduğu için (64*64*3,1) boyutlarına getirilir.  
                predictValue = predict(image,parameter) # nn_algorithm>predict ile tahmin yapılır.
                predictValue = np.squeeze(predictValue) # Tek veri gelir ve bu numpy'dan çıkartılır.
            else:
                flash("Eroor, nothing predict (model train, none parameter file)","red") # parameter.txt yoktur. model.py eğitilmeidir.
                return render_template("index.html", predict=False, fileUrl = newfile)   

            flash("Congratulations, predict successful","green") # Tahmin başarıyla yapılmıştır.
            return render_template("index.html", predict=True, fileUrl = newfile, predictValue=predictValue,sha=sha)
        #Dosya uzantısı eşleşmeme durumu
        else:
            flash('Error!!! Disallowed file extension','red')
            return redirect(url_for('index'))
    #Get Metodu    
    else:
        return redirect(url_for('index'))

@app.route('/answer/<string:button>')
def answer(button):
    if button == "correct":
        cvsWrite(1)
        return redirect(url_for('index'))
    elif button == "wrong":
        cvsWrite(0)
        return redirect(url_for('index'))


# CSV dosyasına resim feature'larını ve label'ını kaydetme
def cvsWrite(imgAnswer):
    haveFile = False
    fieldnames = []
    if os.path.isfile('../nn_model/datasets/predictDataCatvsNotCat.csv'):
        haveFile = True
    with open('../nn_model/datasets/predictDataCatvsNotCat.csv', mode='a', newline='') as csv_file:
        for i in range(12288):
            fieldnames.append("x" + str(i+1))    
        fieldnames.append("y")
        value = image.T
        dic = {}
        for i in range(12288):
            dic.__setitem__('x'+str(i+1),value[0][i])
        dic.__setitem__('y',imgAnswer)

        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not haveFile:
            writer.writeheader()
        writer.writerow(dic)


if __name__ == "__main__":
    app.run(debug=True)