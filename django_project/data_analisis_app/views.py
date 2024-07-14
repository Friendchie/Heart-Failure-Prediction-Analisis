from django.shortcuts import render

'''
creating webpage backend
'''
# Create your views here.

def home(request):
    #funcion que recibe al usuario
    return render(request, 'home.html')