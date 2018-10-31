# -*- coding: utf-8 -*-
import os,time
# import glob
from scipy import signal
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
# import scipy as sp
import scipy.stats
import numpy as np
import seaborn as sns
sns.set_style("white")

def cal(point,start,end,filenames):
    all = np.array(['file', 'Amp', 'Period', 'Phase', 'RAE'])
    allpath = []    
    currentTime = time.strftime("%Y%m%d.%H%M%S", time.localtime())
    for inputPath in filenames:
        inputPath = str(inputPath)
        parent_path,name = os.path.split(inputPath)
        name = name.replace('.csv','')
        resultPath = currentTime + "-Result"
        resultImage_path = os.path.join(parent_path,resultPath)
        if not os.path.exists(resultImage_path):
            os.makedirs(resultImage_path)
        
          
        def inputFile(inputPath):
            data = np.array(np.loadtxt(inputPath, delimiter=",", skiprows=(2), usecols=(2,3)))
            data[:,0] = (data[:,0]-data[0,0])*24
            n = len(data[:,0])
            return (data,n)
        
        (data,n) = inputFile(inputPath)# format:n*2
        rdata = data
        np.savetxt(os.path.join(resultImage_path,name+"-Raw data(disposed).csv"), rdata, delimiter=",", fmt="%s")
        
        SND1 = np.array(data[:,0])
        SND2 = np.array(signal.detrend(data[:,1]))
        data = np.hstack((SND1.reshape(n,1),SND2.reshape(n,1)))
        ##############
        # aaaa = np.vstack((SND1,SND2))
        # pppp = aaaa.T
        # np.savetxt(os.path.join(resultImage_path,name+"-Pre-detrend data.csv"), aaaa.T, delimiter=",", fmt="%s")
        ##############              
        
        '''##########
        plt.figure()
        plt.scatter(SND1,SND2,color='blue',label='Detrend Data',s=5)
        plt.savefig(os.path.join(resultImage_path,"process-1.png"))
        '''##########
        def DTR(data,n):
            M = np.zeros((n,n-point))
            for i in range(n-point):
                Y = np.array(data[i:(i+point+1),1:2])
                X = np.hstack((np.ones((point+1,1)),data[i:(i+point+1),0:1]))
                b = np.linalg.lstsq(X, Y)[0] #linear regress
                B0 = np.array(data[i:(i+point+1),0:1])
                B1 = b[0]+b[1]*B0    
                B = np.hstack((B0,B1))            
                if i>=1:
                    M[i:(i+point+1),i:i+1] = Y-B[:,1:2]        
                else:
                    M[0:point+1,i:i+1] = Y-B[0:point+1,1:2]
            detrendData1 = data[:,0]
            detrendData2 = np.sum(M,axis=1)/np.sum(M!=0,axis=1)
            
            K = np.zeros((n,n-point))
            for i in range(n-point):
                mu,sigmaa = np.mean(detrendData2[i:(i+point+1)]),np.std(detrendData2[i:(i+point+1)])
                C = (detrendData2[i:(i+point+1)]-mu)/sigmaa
                C = C.reshape((point+1,1))
                if i>=1:
                    K[i:(i+point+1),i:i+1] = C
                else:
                    K[0:point+1,i:i+1] = C
            SND1 = data[:,0]
            SND2 = np.sum(K,axis=1)/np.sum(K!=0,axis=1)    
            return (SND1,SND2,detrendData1,detrendData2)
               
        (SND1,SND2,detrendData1,detrendData2) = DTR(data,n)

        SND1 = data[:,0]
        SND2 = signal.detrend(SND2)
        ##############
        aaaa = np.vstack((SND1,SND2))
        dddd = aaaa.T
        np.savetxt(os.path.join(resultImage_path,name+"-Detrend data.csv"), aaaa.T, delimiter=",", fmt="%s")
        ##############        
        
        plt.figure(figsize=(14,7),dpi=300)
        plt.grid(axis='x')
        xtick = list(range(0,int(SND1[-1]+12),12))
        plt.xticks(xtick, fontsize=14)
        plt.xlabel("Time(Hour)", fontsize=14)
        plt.scatter(rdata[:,0:1],rdata[:,1:2],color='black',label='Raw data',s=25)
        plt.yticks(fontsize=14)
        plt.ylabel('Count of raw data(Second)', fontsize=14)
        plt.legend(loc=9, bbox_to_anchor=(0.4, 1.1), ncol=2, fontsize=14, markerscale=3., scatterpoints=1)
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9)
        plt.twinx()
        plt.scatter(SND1,SND2,color='blue',label='Detrend Data',s=25)
        plt.yticks(fontsize=14)
        plt.ylabel('Count of detrend data', fontsize=14, rotation=270, labelpad=18)
        plt.legend(loc=9, bbox_to_anchor=(0.6, 1.1), ncol=2, fontsize=14, markerscale=3., scatterpoints=1)
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9)

        plt.savefig(os.path.join(resultImage_path,name+"-PartA.png"),dpi=300)
        plt.close()
        ###########
        ub = np.argmin(np.abs(end-SND1))
        db = np.argmin(np.abs(start-SND1))
        SNDx = np.array(SND1[db:ub+1])    
        SNDy = np.array(SND2[db:ub+1])
        ###########
        # raw = pppp[db:ub+1,:]
        # np.savetxt(os.path.join(resultImage_path,name+"-Subset(pre-detrend data).csv"), raw, delimiter=",", fmt="%s")
        # raww = dddd[db:ub+1,:]
        # np.savetxt(os.path.join(resultImage_path,name+"-Subset(detrnd data).csv"), raww, delimiter=",", fmt="%s")
        ###########
      
        def fft(SNDx,SNDy,ub,db):    
            N = ub - db + 1   
            ft_SND = np.fft.fft(SNDy, axis=0)
            frequencies = np.fft.fftfreq(SNDy.shape[0], (SNDx[1] - SNDx[0]+SNDx[-1] - SNDx[-2])/2)    
            #periods,amp,phase as initial value in fitting
            periods = 1 / np.abs(frequencies)
            maxrobustNum = np.argmax(np.abs(ft_SND))
            maxT = periods[maxrobustNum]
            amp = np.abs(ft_SND[maxrobustNum])/(N/2)
            phase = np.angle(ft_SND[maxrobustNum])
            return (maxT,amp,phase)    
        
        (maxT,amp,phase) = fft(SNDx,SNDy,ub,db)
        
        SND6 = np.array(detrendData2[db:ub+1])
        SND5 = np.array(detrendData1[db:ub+1])
        plt.figure()
        plt.scatter(SND5,SND6)
        TT,ampamp,phapha = fft(SND5,SND6,ub,db)
        
        SNDy = SNDy * ampamp
        ##############
        # aaaa = np.vstack((SNDx,SNDy))
        # np.savetxt(os.path.join(resultImage_path,name+"-Restored data.csv"), aaaa.T, delimiter=",", fmt="%s")
        ##############           
        """p0 for c0
        p1 for Amp
        p2 for Period
        p3 for Phase"""
        def fun(x,p):
            return p[0]+p[3]*np.cos(2*np.pi*x/p[2]+p[1]) 
        def residuals(p,y,x):
            return y-fun(x,p)
      
        #fitting
        para = leastsq(residuals,[0,phapha,TT,ampamp],args=(SNDy,SNDx))
        pha = para[0][1]
        if pha<=0:
            pha = 2*np.pi*int(1-pha/(2*np.pi))+pha
        if pha>2*np.pi:
            pha = 2*np.pi*int(-pha/(2*np.pi))+pha
        para[0][1] = pha
        
        plt.figure(figsize=(14,7),dpi=300)
        plt.grid(axis='x')
        xtick = list(range(0,int(SND1[-1]+12),12))
        plt.xticks(xtick, fontsize=14)
        plt.yticks(fontsize=14)
        plt.xlabel("Time(Hour)", fontsize=14)
        plt.ylabel('Count of data', fontsize=14)
        plt.scatter(SNDx,SNDy,color='blue',label='Detrend Data',s=25)
        plt.scatter(SNDx,fun(SNDx,para[0]),color='red',label='Fitting Curve',s=25)
        plt.legend(loc=9, bbox_to_anchor=(0.5, 1.1), ncol=2, fontsize=14, markerscale=3., scatterpoints=1)
        plt.subplots_adjust(bottom=0.15, left=0.1, right=0.9)
        plt.savefig(os.path.join(resultImage_path,name+"-PartB.png"),dpi=300)
        plt.close()
        ##############
        aaaa = np.vstack((SNDx,fun(SNDx,para[0])))
        np.savetxt(os.path.join(resultImage_path,name+"-Fitted data.csv"), aaaa.T, delimiter=",", fmt="%s")
        ##############
        
        residualData = SNDy-fun(SNDx,para[0])
        confidence = 0.95
        def mean_confidence_interval(data, confidence):
            a = 1.0*np.array(data)
            n = len(a)# n-1 is df(degree of freedom)
            m, se = np.mean(a), scipy.stats.sem(a)
            h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
            return m, m-h, m+h
        
        (h,m1,m2) = mean_confidence_interval(residualData,confidence)
        RAE = 0.5*(m2-m1)/np.abs(para[0][3])
        
        temp = np.array((np.abs(para[0][3]),para[0][2],para[0][1]))# amp,period,phase
        if para[0][2]<16 or para[0][2]>30:
            name = "***"+name
        #########
        temp[0] = round(temp[0],4)
        temp[1] = round(temp[1],2)
        temp[2] = round(temp[2],4)
        RAE = round(RAE,4)
        #########
        temp = np.hstack((np.array(name),temp,np.array(RAE)))
        all = np.vstack((all,temp))
        allpath.append(resultImage_path)
    # save result matrix in last input path
    np.savetxt(os.path.join(resultImage_path,"summary.csv"), all, delimiter=",", fmt="%s")
    return (all, allpath)

######################################
'''
f=['F:\\luca\\xu\\C2-raw data(all).csv']  # file path
(allresult, Pathlist) = cal(144,16,80,f)
'''

