from django.shortcuts import render
from django.http import HttpResponse
from .models import Post

def home(request):
    context = {
        'posts': Post.objects.all()
    }
    # return HttpResponse('<h1>Blog Home </h1>')
    return render(request, 'blog/home.html', context)
def home2(request):
    
    return HttpResponse('<h1>Seth Nyawacha </h1>')
def about(request):
    return render(request, 'blog/about.html', {'title':'About'})
    # return HttpResponse('<h1>Blog about </h1>')






