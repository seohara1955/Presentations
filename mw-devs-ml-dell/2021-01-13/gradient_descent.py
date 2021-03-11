import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size': 18})

import os
## Get the directory address of current python file
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_dir) ## Set the current directory as working directory

## The package used to create gif files
import numpngw

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGB values
    """
    # draw the renderer
    fig.canvas.draw()
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.array(fig.canvas.renderer._renderer)
    return buf

def plot_original_data(spending, sales):
    plt.scatter(spending, sales, color='#1f77b4',marker = 'o')
    plt.xlabel("Ad Spending, M$")
    plt.ylabel("Sales, Units")
    plt.title("Sales vs Ad Spending")
    #plt.show()
    axes = plt.gca()
    axes.set_xlim([0,50])
    axes.set_ylim([0,35])
    plt.tight_layout()
    fig1 = plt.gcf()
    nfig = fig2data(fig1)
    seq.append(nfig)
    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    fig1.savefig('./figs/gradient_descent-1.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

def plot_data(spending, sales, w, b, epoch, image_counter):
    plt.figure(image_counter)
    plt.xlabel("Ad Spending, M$")
    plt.ylabel("Sales, Units")
    axes = plt.gca()
    axes.set_xlim([0,50])
    axes.set_ylim([0,35])
    plt.scatter(spending,sales,color='#1f77b4',marker='o' )
    X_plot = np.linspace(0,50,50)
    plt.plot(X_plot,X_plot*w + b)
    heading = 'epoch = '+str(epoch)+' loss = '+str(round(loss(spending,sales,w,b), ndigits=1))
    plt.title(heading)
    #plt.show()
    plt.tight_layout()
    fig1 = plt.gcf()
    nfig = fig2data(fig1)
    seq.append(nfig)
    fig1.subplots_adjust(top = 0.98, bottom = 0.1, right = 0.98, left = 0.08, hspace = 0, wspace = 0)
    fig1.savefig('./figs/gradient_descent-' + str(image_counter) + '.png', dpi=1000, bbox_inches = 'tight', pad_inches = 0)

def update_w_and_b(spending, sales, w, b, alpha):
    dr_dw = 0.0
    dr_db = 0.0
    N = len(spending)
    for i in range(N):
        dr_dw += -2*spending[i]*(sales[i] - (w*spending[i] + b))
        dr_db += -2*(sales[i] - (w*spending[i] + b))
    # update w and b
    w = w - (dr_dw/float(N))*alpha
    b = b - (dr_db/float(N))*alpha
    return w,b

def loss(spending,sales,w,b):
    N = len(spending)
    total_error = 0.0
    for i in range(N):
        total_error += (sales[i] - (w*spending[i] + b))**2
    return total_error/N

def train(spending, sales, w, b, alpha, epochs):
    image_cnt = 2
    plot_original_data(spending, sales)

    ### main loop ###
    for e in range(epochs):
        w, b = update_w_and_b(spending,sales,w,b,alpha)
    ### main loop ###

        # log the progress
        if (e==0) or (e<3000 and e%400==0) or (e%3000==0):
            print()
            print("epoch: ", str(e), "loss: "+str(loss(spending,sales,w,b)))
            print("w, b: ",w,b)
            plot_data(spending, sales, w, b, e, image_cnt)
            image_cnt += 1

    return w,b

def predict(x,w,b):
    return w*x + b

# train
seq = []
df = pd.read_csv("advertising.csv")
w,b = train(df['radio'],df['sales'], 0.0, 0.0, 0.001, 16000)

# predict
x_new = 23.0
y_new = predict(x_new, w, b)
print(y_new)

# create gif
numpngw.write_apng('gradient_descent.png', seq, delay=750)
