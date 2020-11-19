import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"

def cost(area, prices, theta):
    n = len(prices)
    yp = area.dot(theta)
    error = (yp - prices)**2
    
    return 1/(2*n)*(numpy.sum(error) + numpy.sum(theta)**2)
    

def GradientDescent(area, prices, theta, alpha, iteractions):
    m = len(prices)
    costs = []
    for i in range(iteractions):
        yPred = area.dot(theta)
        error = numpy.dot(area.transpose(), (yPred - prices))
        theta -= alpha*(1/m)*error
        costs.append(cost(area, prices, theta))
    return theta, costs
    
def Predict(area, theta):    
    yPred = numpy.dot(theta.transpose(), area)
    return yPred

def predict_price(area, prices) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    alpha = 0.01
    n_iteration = 1000
    n_samples = len(area)
    print(area)
    n_features = numpy.size((len(area), 1))
    param = numpy.zeros((n_features+1, 1))
    coef = None
    intercept = None
    theta = numpy.zeros(area.shape)
    
    theta, costs = GradientDescent(area, prices, theta, alpha, n_iteration)
    yd = Predict(area, theta)
    
    return yd
    
    ...


if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    areas = areas/(max(areas)-min(areas))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas, prices)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
