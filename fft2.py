# -*- coding: utf-8 -*-
import os,time
from scipy import signal
import matplotlib.pyplot as plt 
from scipy.optimize import leastsq
import scipy.stats
import numpy as np
import seaborn as sns
import pandas as pd
sns.set_style('white')

##############################
def cal(point, start, end, minPeriod, maxPeriod, filenames, pic):
    all = np.array(['Sample(s)', 'Amp', 'Period', 'Phase', 'RAE', 'Rhythmic'])
    allpath = []
    outputFileName = ''
    for inputPath in filenames:
        currentTime = time.strftime('%Y%m%d.%H%M%S', time.localtime())
        inputPath = str(inputPath)
        parent_path,name = os.path.split(inputPath)
        name = name.replace('.csv', '')
        resultPath = name + '-' + currentTime + '-Result'
        resultImage_path = os.path.join(parent_path, resultPath)
        if not os.path.exists(resultImage_path):
            os.makedirs(resultImage_path)

        ##############################
        def inputFile(inputPath):
            df = pd.read_table(inputPath, sep = ',', header = None)
            df = df.fillna(0.0000000000000000000000000000000000001) #replace NaN
            df = df.values
            samples = df[1:, 0]
            sample_num = len(samples)
            time_data = df[0, 1:]
            n = len(time_data)
            all_data = df[1:sample_num+1, 1:]
            return (samples, sample_num, time_data, n, all_data)
        
        (samples, sample_num, time_data, n, all_data) = inputFile(inputPath)
        times = time_data.T

        ##############################
        st = time.clock()
        for i in range(0, sample_num):
            # p = round((i + 1) * 100 / sample_num, 4)
            sample_name = str(samples[i])
            data = np.vstack((time_data, all_data[i, :]))
            data[0,:] = (data[0, :] - data[0, 0])
            rdata = data.T
            SND1 = np.array(data[0, :])
            SND2 = np.array(signal.detrend(data[1, :]))
            data = np.hstack((SND1.reshape(n, 1), SND2.reshape(n, 1)))
        
            ##############################
            def DTR(data, n):
                M = np.zeros((n, n-point))
                for i in range(n-point):
                    Y = np.array(data[i:(i+point+1), 1:2])
                    X = np.hstack((np.ones((point+1, 1)), data[i:(i+point+1), 0:1]))
                    b = np.linalg.lstsq(X, Y)[0] #linear regress
                    B0 = np.array(data[i:(i+point+1), 0:1])
                    B1 = b[0]+b[1]*B0
                    B = np.hstack((B0, B1))
                    if i >= 1:
                        M[i:(i+point+1), i:i+1] = Y-B[:, 1:2]
                    else:
                        M[0:point+1,i:i+1] = Y-B[0:point+1,1:2]
                detrendData1 = data[:, 0]
                detrendData2 = np.sum(M,axis = 1)/np.sum(M != 0, axis = 1)
            
                K = np.zeros((n, n-point))
                for i in range(n-point):
                    mu,sigmaa = np.mean(detrendData2[i:(i+point+1)]), np.std(detrendData2[i:(i+point+1)])
                    C = (detrendData2[i:(i+point+1)]-mu)/sigmaa
                    C = C.reshape((point+1, 1))
                    if i>=1:
                        K[i:(i+point+1), i:i+1] = C
                    else:
                        K[0:point+1, i:i+1] = C
                SND1 = data[:, 0]
                SND2 = np.sum(K,axis = 1)/np.sum(K != 0,axis = 1)    
                return (SND1, SND2, detrendData1, detrendData2)
               
            (SND1, SND2, detrendData1, detrendData2) = DTR(data, n)

            SND1 = data[:, 0]
            SND2 = signal.detrend(SND2)

            ##############################
            parta_x = times
            parta_y = rdata[:,1]
            parta_yy = SND2
            def partA(parta_x, parta_y, parta_yy):
                plt.figure(figsize = (14, 7), dpi = 300)
                plt.grid(axis = 'x')
                xtick = list(range(int(parta_x[0]), int(parta_x[-1])+12, 12))
                plt.xticks(xtick, fontsize = 14)
                plt.xlabel('Time(Hour)', fontsize = 14)
                if point <= 72:
                    plt.plot(parta_x, parta_y, color = 'black', label = 'Raw data', linewidth = 5)
                else:
                    plt.scatter(parta_x, parta_y, color = 'black', label = 'Raw data', s = 25)
                plt.yticks(fontsize = 14)
                plt.ylabel('Count of raw data', fontsize = 14)
                plt.legend(loc = 9, bbox_to_anchor = (0.4, 1.1), ncol = 2, fontsize = 14, markerscale  = 3., scatterpoints = 1)
                plt.subplots_adjust(bottom = 0.15, left = 0.1, right = 0.9)
                plt.twinx()
                if point <= 72:
                    plt.plot(parta_x, parta_yy, color = 'blue', label = 'Detrend Data', linewidth = 5)
                else:
                    plt.scatter(parta_x, parta_yy, color = 'blue', label = 'Detrend Data', s = 25)
                plt.yticks(fontsize=14)
                plt.ylabel('Count of detrend data', fontsize = 14, rotation = 270, labelpad = 18)
                plt.legend(loc = 9, bbox_to_anchor = (0.6, 1.1), ncol = 2, fontsize = 14, markerscale = 3., scatterpoints = 1)
                plt.subplots_adjust(bottom = 0.15, left = 0.1, right = 0.9)
                plt.savefig(os.path.join(resultImage_path, sample_name+'-PartA.png'), dpi = 300)
                plt.close()

            ##############################
            ub = np.argmin(np.abs(end-times))
            db = np.argmin(np.abs(start-times))
            subtimes = times[db:ub+1]
            SNDx = np.array(SND1[db:ub+1], dtype = float)
            SNDy = np.array(SND2[db:ub+1], dtype = float)
          
            def fft(SNDx, SNDy, ub, db):
                N = ub - db + 1   
                ft_SND = np.fft.fft(SNDy, axis = 0)
                frequencies = np.fft.fftfreq(SNDy.shape[0], (SNDx[1] - SNDx[0]+SNDx[-1] - SNDx[-2])/2)    
                #periods,amp,phase as initial value in fitting
                periods = 1 / np.abs(frequencies)
                maxrobustNum = np.argmax(np.abs(ft_SND))
                maxT = periods[maxrobustNum]
                amp = np.abs(ft_SND[maxrobustNum])/(N/2)
                phase = np.angle(ft_SND[maxrobustNum])
                return (maxT, amp, phase)    
        
            (maxT, amp, phase) = fft(SNDx, SNDy, ub, db)

            ##############################
            SND6 = np.array(detrendData2[db:ub+1])
            SND5 = np.array(detrendData1[db:ub+1])
            TT, ampamp, phapha = fft(SND5, SND6, ub, db)
            SNDy = SNDy * ampamp

           ##############################
            '''p0 for c0
            p1 for Amp
            p2 for Period
            p3 for Phase'''
            def fun(x, p):
                return p[0]+p[3]*np.cos(2*np.pi*x/p[2]+p[1])# fitting formula
            def residuals(p, y, x):
                return y-fun(x,p)
          
            #fitting
            para = leastsq(residuals, [0, phapha, TT, ampamp],args = (SNDy, SNDx))
            pha = para[0][1]
            if pha <= 0:
                pha = 2*np.pi*int(1-pha/(2*np.pi))+pha
            if pha > 2*np.pi:
                pha = 2*np.pi*int(-pha/(2*np.pi))+pha
            para[0][1] = pha

            ##############################
            partb_x = subtimes
            partb_y = SNDy
            partb_yy = fun(SNDx, para[0])
            def partB(partb_x, partb_y, partb_yy):
                plt.figure(figsize = (14,7), dpi = 300)
                plt.grid(axis = 'x')
                xtick = list(range(int(partb_x[0]), int(partb_x[-1])+12, 12))
                plt.xticks(xtick, fontsize = 14)
                plt.yticks(fontsize = 14)
                plt.xlabel('Time(Hour)', fontsize = 14)
                plt.ylabel('Count of data', fontsize = 14)
                if point <= 72:
                    plt.plot(partb_x, partb_y, color = 'blue', label = 'Detrend Data', linewidth = 5)
                    plt.plot(partb_x, partb_yy, color = 'red', label = 'Fitting Curve', linewidth = 5)
                else:
                    plt.scatter(partb_x, partb_y, color = 'blue', label = 'Detrend Data', s = 25)
                    plt.scatter(partb_x, partb_yy,color = 'red',label = 'Fitting Curve', s = 25)
                plt.legend(loc = 9, bbox_to_anchor = (0.5, 1.1), ncol = 2, fontsize = 14, markerscale = 3., scatterpoints = 1)
                plt.subplots_adjust(bottom = 0.15, left = 0.1, right = 0.9)
                plt.savefig(os.path.join(resultImage_path, sample_name+'-PartB.png'), dpi = 300)
                plt.close()
                aaaa = np.vstack((partb_x, partb_yy))
                np.savetxt(os.path.join(resultImage_path,sample_name+'-Fitted data.csv'), aaaa.T, delimiter = ',', fmt = '%s')

            ##############################
            residualData = SNDy - fun(SNDx, para[0])
            confidence = 0.95
            def mean_confidence_interval(data, confidence):
                a = 1.0*np.array(data)
                n = len(a)# n-1 is df(degree of freedom)
                m, se = np.mean(a), scipy.stats.sem(a)
                h = se * scipy.stats.t._ppf((1+confidence)/2., n-1)
                return m, m-h, m+h
            
            (h,m1,m2) = mean_confidence_interval(residualData, confidence)
            RAE = 0.5*(m2-m1)/np.abs(para[0][3])

            ##############################
            temp = np.array((np.abs(para[0][3]), para[0][2], para[0][1]))# amp,period,phase
            temp[0] = round(temp[0], 4)
            temp[1] = round(temp[1], 2)
            temp[2] = round(temp[2], 4)
            RAE = round(RAE,4)
            if para[0][2] < minPeriod or para[0][2] > maxPeriod:
                Rhythmic = ''
            else:
                Rhythmic = 'yes'
            if pic != 'no':
                partA(parta_x, parta_y, parta_yy)
                partB(partb_x, partb_y, partb_yy)

            ##############################
            temp = np.hstack((np.array(sample_name), temp, np.array(RAE), np.array(Rhythmic)))
            all = np.vstack((all, temp))
            # duration = round(time.clock() - st, 2)
            # remaining = round(duration * 100 / (0.01 + p) - duration, 2)
            # print('progress: {0}%, Time consuming: {1}s, Remaining: {2}s'.format(p, duration, remaining), end = '\n')
            # time.sleep(0.01)

        # save result matrix in last input path
        allpath.append(resultImage_path)
        np.savetxt(os.path.join(resultImage_path, 'All results.csv'), all, delimiter = ',', fmt = '%s')
    return (all, allpath)

######################################

# f = ['F:\\RCAT_WIN\\1.csv'] # file path
# pic = 'no' # do not output figures, set default, change to 'yes' if you want figures
# (allresult, Pathlist) = cal(6, 0, 48, 16, 32, f, pic)


