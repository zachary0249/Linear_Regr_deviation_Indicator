import numpy as np
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
import pandas_datareader as web
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt


# epsilon (ε) = the distance both above and below a given standard deviation of linear regression

def epsilon(start_string, symbol, periods=100, first=False, upper_deviation=1, lower_deviation=1, plot=False, legend=False,
            summary=False, second=True, second_up=2, second_low=2, third=True, third_up=3, third_low=3,
            print_array=False):

    # equation to counteract the loss of days due to weekends not being open for the market - there will still be some descrepency
    days = int(round((periods / 7) * 2 + periods)) + 10

    # defining constants
    END = dt.strptime(start_string, '%Y-%m-%d')  # converting the string into dt object
    START = END - relativedelta(days=days)
    SOURCE = 'yahoo'
    STOCK = [str(symbol)]

    # gathering and formatting the stock data
    df = web.DataReader(STOCK, start=START, end=END, data_source=SOURCE)  # symbol data
    df.drop(['Close', 'Open', 'High', 'Low', 'Volume'], inplace=True, axis=1)
    df.fillna(-99999, inplace=True)
    #df.sort_values(by=['Date'], ascending=False, inplace=True)
    end = np.subtract(END, relativedelta(days=days))
    df = df.loc[end:start_string]


    y = df.values # defining response var
    x = [_ for _ in range(len(df.index))] # defining explanatory var
    x = sm.add_constant(x)  # add constant to predictor variables
    usable_x = [_ for _ in range(len(df.index))]
    model = sm.OLS(y, x).fit()  # fitting linear regression model
    model_summary = model.summary()  # the model stats
    if summary:
        print(model_summary)

    model_summary_html = model_summary.tables[1].as_html()  # converts to html
    model_df = pd.read_html(model_summary_html, header=0, index_col=0)[0]  # converts html to pandas df
    slope = model_df.iloc[1, 0]  # value for the slope in equation
    yint = model_df.iloc[0, 0]  # value for the y - intercept in equation
    linreg_coordinates = np.array([(var * slope) + yint for var in usable_x])  # y - values for the lin reg

    # calculating the deviation channels equations and coordinates
    std = np.std(linreg_coordinates)  # standard deviation of lin reg y coordinates
    pos_deviation = std * upper_deviation
    neg_deviation = std * (-lower_deviation)
    # returns two solutions and one is extraneous
    b_upper = abs(pos_deviation * (sqrt(slope ** 2 + 1)) + yint)  # solving for the yint of upper deviation channel line
    b_lower = abs(neg_deviation * (sqrt(slope ** 2 + 1)) + yint)  # solving for the yint of lower deviation channel line
    upper_deviation_coordinates = [z * slope + b_upper for z in usable_x]
    low_deviation_coordinates = [z * slope + b_lower for z in usable_x]

    if second:
        another_pos_deviation = std * second_up
        another_neg_deviation = std * (-second_low)
        another_b_upper = abs(another_pos_deviation * (
            sqrt(slope ** 2 + 1)) + yint)  # solving for the yint of upper deviation channel line
        another_b_lower = abs(another_neg_deviation * (
            sqrt(slope ** 2 + 1)) + yint)  # solving for the yint of lower deviation channel line
        another_upper_deviation_coordinates = [z * slope + another_b_upper for z in usable_x]
        another_low_deviation_coordinates = [z * slope + another_b_lower for z in usable_x]

    if third:
        third_pos_deviation = std * third_up
        third_neg_deviation = std * (-third_low)
        third_b_upper = abs(third_pos_deviation * (sqrt(slope ** 2 + 1)) + yint)
        third_b_lower = abs(third_neg_deviation * (sqrt(slope ** 2 + 1)) + yint)
        third_upper_deviation_coordinates = [z * slope + third_b_upper for z in usable_x]
        third_low_deviation_coordinates = [z * slope + third_b_lower for z in usable_x]

    # plotting the data
    if plot:
        plt.plot([z for z in range(len(df.index))], y, label='Price')
        plt.plot(usable_x, linreg_coordinates, label='Linear regression', color='green')

        if first:
            plt.plot(usable_x, upper_deviation_coordinates, label=str(upper_deviation) + 'SD', color='red')
            plt.plot(usable_x, low_deviation_coordinates, label=str(-lower_deviation) + 'SD', color='red')

        if second:
            plt.plot(usable_x, another_upper_deviation_coordinates, label=str(second_up) + 'SD', color='red')
            plt.plot(usable_x, another_low_deviation_coordinates, label=str(-second_low) + 'SD', color='red')

        if third:
            plt.plot(usable_x, third_upper_deviation_coordinates, label=str(third_up) + 'SD', color='red')
            plt.plot(usable_x, third_low_deviation_coordinates, label=str(-third_low) + 'SD', color='red')

        plt.title(symbol + ' Linear Regression with STD Channels')
        plt.xlabel('X')
        plt.ylabel('Price')
        if legend:
            plt.legend()
        plt.show()


    # finding the % distance the current price is from given standard deviation
    most_recent_price = df.iloc[-1:].values # most recent price in df
    # must index [0] because the other value is extraneous
    d1 = ((most_recent_price - upper_deviation_coordinates[-1]) / upper_deviation_coordinates[-1] * 100)
    dneg1 = ((most_recent_price - low_deviation_coordinates[-1]) / low_deviation_coordinates[-1] * 100)
    p = np.array(most_recent_price)

    d3 = ((most_recent_price - third_upper_deviation_coordinates[-1]) / third_upper_deviation_coordinates[-1] * 100)
    dneg3 = ((most_recent_price - third_low_deviation_coordinates[-1]) / third_low_deviation_coordinates[
                -1] * 100)

    d2 = ((most_recent_price - another_upper_deviation_coordinates[-1]) / another_upper_deviation_coordinates[-1] * 100)
    dneg2 = ((most_recent_price - another_low_deviation_coordinates[-1]) / another_low_deviation_coordinates[
            -1] * 100)

    if all([first, second, third]):
        upper = np.array([d3, d2, d1])
        upper.shape = (3, 1)
        p.shape = (1, 1)
        lower = np.array([dneg1, dneg2, dneg3])
        lower.shape = (3, 1)

    elif second and third:
        upper = np.array([d3, d2])
        upper.shape = (2, 1)
        p.shape = (1, 1)
        lower = np.array([dneg2, dneg3])
        lower.shape = (2, 1)

    elif first and second:
        upper = np.array([d1, d2])
        upper.shape = (2, 1)
        p.shape = (1, 1)
        lower = np.array([dneg1, dneg2])
        lower.shape = (2, 1)

    else:
        upper = np.array([d1])
        upper.shape = (1, 1)
        p.shape = (1, 1)
        lower = np.array([dneg1])
        lower.shape = (1, 1)


    if print_array:
        print(upper)
        print()
        print(p)
        print()
        print(lower)

    array = np.concatenate([upper, lower])
    dimensions = len(upper) + len(lower)
    array.shape = (dimensions, 1)

    return array


# sample function call below, can be used as a template if you wish, just uncomment it
#epsilon('2021-2-25', 'AAPL', 100, first=False, upper_deviation=1, lower_deviation=1, second_up=2, second_low=2, plot=True, second=True, legend=False, print_array=True, summary=False)
