from django.shortcuts import render,redirect
from .forms import Textform
from .util import pred

def home(request):
    context ={}
    if request.method=='POST':
        form=Textform(request.POST)
        if form.is_valid():
            text = request.POST['Text']
            context['text'] = text
            images = pred([text],int(request.POST['samples']))
            print(images)
            context['images'] = images
            return render(request,'result.html',context=context)
    else:
        form = Textform()
        context['form']=form
    return render(request,'home.html',context=context)