import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

import scipy as sp
from scipy.stats import skew
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from scipy.stats import skew
import os
import sys

plt.rcParams["figure.figsize"] = (14,8)
plt.rcParams['font.size'] = 14


#Global color variables
month_hex= ['#1a2795','#1a2795','#7b366c','#9b2a4b', '#bd2a40', '#D22B2B','#D22B2B' ,'#D22B2B','#9b2a4b','#7b366c' ,'#54307c','#1a2795']
purp = '#440154'
teal = '#21918c'
yellow = '#fde725'

def import_data():
    df = pd.read_csv('en_climate_daily_AB_3075858_2001_P1D.csv')
    df=df.rename(columns={"Mean Temp (°C)": "meanTemp"})
    df=df.rename(columns={"Min Temp (°C)": "minTemp"})
    df=df.rename(columns={"Max Temp (°C)": "maxTemp"})


    df =df[['Date/Time', 'Year', 'Month', 'Day', 
        'Data Quality', 'maxTemp','minTemp','meanTemp',
        'Total Rain (mm)','Total Rain Flag', 'Total Snow (cm)', 
        'Total Snow Flag','Total Precip (mm)'
        ]]

    a=-1
    b=1
    df['mean_Temp_Norm'] = df.groupby('Month')['meanTemp'].transform(lambda x: ( a +   (( (x-x.min())*(b-a))  /  (x.max()-x.min()) )  ))
    df['mean_Temp_stand'] = df.groupby('Month')['meanTemp'].transform(lambda x: (  x-x.mean() )/ (x.std() )  )

    return df


## fit Gauss
def norm(data,a=-1,b=1):
    num=((data- np.min(data))*(b-a))
    denom = (np.max(data)-np.min(data))
    return a+ (num/denom)

def standardize(data):
    return (data-np.mean(data))/(np.std(data))

def deNorm(data,xmin,xmax):
    d= (data*(xmax-xmin))+xmin
    return d

def deStandard(data,mean,std):
    return data * std + mean


def get_violin(df,col,title,saveStr):
    mean_of_means = list(df[['Month',col]].groupby(by= 'Month',as_index=False).mean()[col])

    plt.figure(figsize= (14,8))
    ax=sns.violinplot(data=df, y=col, x="Month",palette=month_hex,inner= 'quart')
    plt.xlabel('')
    plt.ylabel('Mean Temp (°C)')
    plt.title(title)
    plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11], labels=['Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sept.','Oct.','Nov.','Dec.'])
    plt.scatter(x=range(12),y=mean_of_means,c="w")

    for l in ax.lines:
        l.set_linestyle('dotted')
        l.set_linewidth(2)
        l.set_color('w')
        l.set_alpha(0.9)

    saveStr= saveStr+'.pdf'
    plt.savefig(saveStr)
    #plt.show()

    return 



def get_norm_skew(col,title,saveStr):

    monthly_means= df[['Month','maxTemp','minTemp','meanTemp']].groupby(by='Month',as_index=False).mean()
    df_temp= df[['Month','maxTemp','minTemp','meanTemp']]

    mean_of_means = np.array(df[['Month',col]].groupby(by= 'Month',as_index=False).mean()[col])
    med_of_means=  np.array(df[['Month',col]].groupby(by= 'Month',as_index=False).median()[col])
    dif_mean_med= mean_of_means-med_of_means

    sk= df.groupby('Month')[col].transform(lambda x: ( skew(x) ) )
    skewness= sk.unique()

    months= range(1,13)
    above=[]
    for m in months:
        mean = float(monthly_means[monthly_means['Month']==m].maxTemp)
        days= len(df_temp[(df_temp['Month']== m) & (df_temp['maxTemp'] <= mean)])
        above.append(days)

    a=np.min(dif_mean_med)
    b=np.max(dif_mean_med)

    fig, ax1 = plt.subplots()
    l1,=ax1.plot(dif_mean_med,label= 'Mean minus Median',color= teal,linewidth=2.5,linestyle='--')
    l2,=ax1.plot(norm(skewness,a,b),label='Normed Skewness',color= teal,linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=teal)
    ax1.axhline(0,color=teal,linestyle='--', linewidth=1.5,alpha= 0.5)
    ax1.grid(axis="x",color='black', linestyle='--', linewidth=0.8,alpha= 0.5)
    
    plt.xticks(ticks=[0,1,2,3,4,5,6,7,8,9,10,11], labels=['Jan.','Feb.','Mar.','Apr.','May','June','July','Aug.','Sept.','Oct.','Nov.','Dec.'])
    
    ax2 = ax1.twinx()

    l3,=ax2.plot(above,label= 'No. of Days Below Monthly Mean',color=purp,linewidth=2.5)
    ax2.tick_params(axis='y', labelcolor=purp)
    ax2.legend(handles=[l1,l2,l3],loc=(0.35,0.15),framealpha=0.8)  

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(saveStr+'.pdf')
    #plt.show()
    return


if __name__ == '__main__':
    homeDir= sys.path[0]
    os.chdir(homeDir)
    df= import_data()


    outDir = homeDir+'/monthly'

    isExist = os.path.exists(outDir)
    if isExist == False:
        os.makedirs(outDir)
    os.chdir(outDir)

    print('Saving Monthly files in ', outDir)

    get_violin(df,'meanTemp','Mean Temp Dist per Month','mean_monthly_dist')
    get_violin(df,'minTemp','Min. Temp Dist per Month','min_monthly_dist')
    get_violin(df,'maxTemp','Max. Temp Dist per Month','max_monthly_dist')

    get_norm_skew('maxTemp','Max. Temp Dist per Month','max_skew')
    get_norm_skew('minTemp','Min. Temp Dist per Month','min_skew')
    get_norm_skew('meanTemp','Mean Temp Dist per Month','mean_skew')
