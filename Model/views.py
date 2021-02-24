from django.http import HttpResponse

from Model.model import Model

model = Model()


# Create your views here.

def ping(request):
    return HttpResponse("Pong!")


def index(request):
    t = request.POST['text']
    return HttpResponse(model.predict(t))
