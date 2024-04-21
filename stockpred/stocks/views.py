from django.shortcuts import render
from .stock_model import *
import pickle

# Create your views here.
def index(request):

    if request.method == 'POST':
        sname = request.POST['search']

        data, d = DataImport(sname)

        OneYear_data(data)

        LastMonth_data(data)

        LastWeek_data(data)

        model, scaler = modelBuilding(data)

        pred, next_data = predictData(data, d, model, scaler)

        data_final = CombineData(data, pred, next_data)

        plotPredict(data_final, sname)


        return render(request, 'stocks/index.html', {
            "disp": True,
        })

    return render(request, 'stocks/index.html', {
        "disp": False,
    })