from django.shortcuts import render,redirect
import tensorflow as tf
import tensorflow_text as text
import numpy as np

MODEL=tf.keras.models.load_model('email_models/no_dropouts')
# Create your views here.
result=[]
size=0

def home(request):
    return render(request,'home.html',{'result':result})

def predict(request):
    if request.method=='POST':
        mail=request.POST.get('mail')
        pred=MODEL.predict([mail])[0][0]
        print(pred)
        if pred>0.5:
            pred_class='SPAM'
            conf=round(pred*100,2)
        else:
            pred_class='REAL'
            conf=100 - round(pred*100,2)

        
        global size
        size=size+1
        result.append({'sn':size,'mail':mail,'class':pred_class,'conf':conf})
        return redirect('/')
    
    return redirect('/')

    