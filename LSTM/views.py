from django.http import HttpResponse

from LSTM.LSTM import LSTM

model = LSTM()


# Create your views here.

def ping(request):
    return HttpResponse("Pong!")


def index(request):
    t = request.POST['text']
    return HttpResponse(model.predict(t))
