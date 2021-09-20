from django.shortcuts import render,redirect
from .forms import Textform

def home(request):
    context ={}
    if request.method=='POST':
        form=Textform(request.POST)
        if form.is_valid():
            text = request.POST['Text']
            context['text'] = text
            # synthesize image
            context['image'] = 'test.jfif'
            return render(request,'result.html',context=context)
    else:
        form = Textform()
        context['form']=form
    return render(request,'home.html',context=context)