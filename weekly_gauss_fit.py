import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import os
import sys

import scipy as sp
from scipy import stats
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp


plt.rcParams["figure.figsize"] = (12,8)
plt.rcParams['font.size'] = 14

purp = '#440154'
teal = '#21918c'
yellow = '#fde725'


df = pd.read_csv('en_climate_daily_AB_3075858_2001_P1D.csv')
df=df.rename(columns={"Mean Temp (°C)": "meanTemp"})
df=df.rename(columns={"Min Temp (°C)": "minTemp"})
df=df.rename(columns={"Max Temp (°C)": "maxTemp"})


df =df[['Date/Time', 'Year', 'Month', 'Day', 
    'Data Quality', 'maxTemp','minTemp','meanTemp',
    'Total Rain (mm)','Total Rain Flag', 'Total Snow (cm)', 
    'Total Snow Flag','Total Precip (mm)'
    ]]


date = pd.to_datetime(df['Date/Time'])
week= date.dt.isocalendar().week
df.insert(4, 'Week', week)


maxTemp= df[['Week','maxTemp']]
minTemp= df[['Week','minTemp']]
meanTemp= df[['Week','meanTemp']]


maxTemp_week_mean = maxTemp.groupby(by=["Week"],as_index=False).mean()
minTemp_week_mean = minTemp.groupby(by=["Week"],as_index=False).mean()
meanTemp_week_mean = meanTemp.groupby(by=["Week"],as_index=False).mean()

maxTemp_week_std = maxTemp.groupby(by=["Week"],as_index=False).std()
minTemp_week_std = minTemp.groupby(by=["Week"],as_index=False).std()
meanTemp_week_std = meanTemp.groupby(by=["Week"],as_index=False).std()


## fit Gauss
def norm(data,a,b):
    num=((data- np.min(data))*(b-a))
    denom = (np.max(data)-np.min(data))
    return a+ (num/denom)

def standardize(data):
    return (data-np.mean(data))/(np.std(data))

def Gauss(x, A, B,H): 
    y = H + A*np.exp(-1*B*x**2) 
    return y

# def deNorm(data,xmin,xmax):
#     d= (data*(xmax-xmin))+xmin
#     return d


def deNorm(norm, xmin,xmax):
    a= np.min(norm)
    b= np.max(norm)
    return ( ((norm-a) * ( xmax-xmin ))/(b-a) ) + xmin

def deStandard(data,mean,std):
    return data * std + mean


# mean vs Std
def plot_mean_std(dfMean,dfStd, title):

    plt.figure(figsize=(14,8))

    xlen= len(dfMean.iloc[:,1])
    
    xdata= norm(range(0,xlen),0,48)

    plt.plot(xdata,dfMean.iloc[:,1],'*-',label='mean',color=purp,linewidth=2.5)
    plt.plot(xdata,dfStd.iloc[:,1],'*-',label='std',color=teal,linewidth=2.5)

    plt.ylabel('Temp (°C)')
    
    plt.xticks(ticks=[0,4,8,12,16,20,24,28,32,36,40,44,48], labels=['Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sept.','Oct.','Nov.','Dec.',''])
    plt.grid(axis="x",color='black', linestyle='--', linewidth=0.8,alpha= 0.5)
    
    #plt.

    pltStr= title+ ' Temperature'
    saveStr= title+'temp_mean_VS_std.pdf'

    plt.legend()

    plt.title(pltStr)
    plt.savefig(saveStr)
    #plt.show()
    return 



## corr
def corr():
    corr_file= 'temp_mean_std_corr.txt'

    mean_corr= meanTemp_week_mean['meanTemp'].corr(meanTemp_week_std['meanTemp'])
    max_corr = maxTemp_week_mean['maxTemp'].corr(maxTemp_week_std['maxTemp'])
    min_corr = minTemp_week_mean['minTemp'].corr(minTemp_week_std['minTemp'])


    mean_str= f"Correlation of mean and std of mean temp is: {mean_corr:.3f}"
    min_str= f"Correlation of mean and std of min temp is: {min_corr:.3f}"
    max_str= f"Correlation of mean and std of max temp is: {max_corr:.3f}"

    with open(corr_file, 'w') as f:
        f.write(mean_str)
        f.write('\n')
        f.write(min_str)
        f.write('\n')
        f.write(max_str)
        f.write('\n')
    return





def fit_gauss(data,title):

    x=data.iloc[:,0]
    y=data.iloc[:,1]

    xmin=np.min(x)
    xmax= np.max(x)

    xdata= norm(x,-2,2)

    ymin=np.min(y)
    ymax=np.max(y)

    ydata= norm(y,0,1)

    parameters, covariance = curve_fit(Gauss, xdata, ydata) 

    fit_A = parameters[0] 
    fit_B = parameters[1]
    fit_H = parameters[2] 

    fit_y = Gauss(xdata, fit_A, fit_B,fit_H) 

    xdata= deNorm(xdata, xmin,xmax)
    ydata= deNorm(ydata, ymin,ymax)
    fit_y= deNorm(fit_y,ymin,ymax)

    plt.figure(figsize=(14,8))
    plt.title(str(title))

    plt.plot(norm(xdata,0,48), ydata, '-*', label='Original Data',color=purp,linewidth=2.5) 
    plt.plot(norm(xdata,0,48), fit_y, '-', label='Gaussian fit',color= teal,linewidth=2.5) 

    plt.xticks(ticks=[0,4,8,12,16,20,24,28,32,36,40,44,48], labels=['Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sept.','Oct.','Nov.','Dec.',''])
    plt.grid(axis="x",color='black', linestyle='--', linewidth=0.8,alpha= 0.5)
    
    
    plt.ylabel('Temp (°C)')
    
    saveStr= 'temp'+title+ '_Gauss.pdf'
    titleStr = title + ' Temperature'
    
    plt.title(titleStr)
    plt.legend()
    plt.savefig(saveStr)

    return



if __name__ == '__main__':


    homeDir= sys.path[0]
    os.chdir(homeDir)
    outDir = homeDir+'/gauss_fit_out_test'

    isExist = os.path.exists(outDir)
    if isExist == False:
        os.makedirs(outDir)
    os.chdir(outDir)

    print('saving files in ', outDir)

    plot_mean_std(maxTemp_week_mean,maxTemp_week_std, 'Max')
    plot_mean_std(minTemp_week_mean,minTemp_week_std, 'Min')
    plot_mean_std(meanTemp_week_mean,meanTemp_week_std, 'Mean')

    fit_gauss(maxTemp_week_mean,'Max')
    fit_gauss(minTemp_week_mean,'Min')
    fit_gauss(meanTemp_week_mean,'Mean')
    
    corr()
        
