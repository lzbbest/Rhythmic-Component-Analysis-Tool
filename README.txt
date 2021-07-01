RCAT is mainly designed for easily analyze rhythmic components. It's simple, accurate and efficient.

fft.py is the functional code.

RCAT.py is used to activate interface.

mainwindow.ui is for designing layout of the interface.

We mainly use Anaconda (Python version 2.7.13) as the distribution for the reconstruction of algorithms and mathematical models. 

Os, Sys, Numpy, Scipy, Pandas, Matplotlib, PyQt4, Shutil and Pyinstaller are the required Python packages. 

These corresponding version of third-party packages are needed to be installed before running code.

Parameters:

1) Time points (in 24 hr): means total number of points collected in every 24 hours (Per day). For example, if data is collected every 10 minutes (1/6 hour), the time points will be 24 / (1/6) = 144. 

2) Analytic range: is to remove data with low quality. For example, if range of time points of a data is initially from 1 to 48 hours, while the data quality of the first 3 hours is unsatisfactory. In this situation, the interval will be suggested to be set as 3-48 hours to retain only data of high quality.

3) Period of interest: is to screen transcripts whose periods are within this range. For example, if period range is set as 16-32 hours, then transcripts with periods of 24 or 34 hours will be disregarded.

We list parameters of four examples as below for reference.

1.csv
Time points (in 24 hours) : 6
Analytic range : 0-48 (default)
Period of interest : 16-30 (default)

2.csv 
Time points (in 24 hours) : 12
Analytic range : 0-48 (default)
Period of interest : 16-30 (default)

3.csv
Time points (in 24 hours) : 24
Analytic range : 0-48 (default)
Period of interest : 16-30 (default)

4.csv
Time points (in 24 hours) : 24
Analytic range : 0-48 (default)
Period of interest : 16-30 (default)


An instructional video (Video S1) is also provided for users to quickly use RCAT.

Notes: The names of file and filepath with time series data should be in English. Other languages will not be successfully decoded. Besides, if  more than 70% of all values are "0" in one sample, RCAT will be unable to perform analysis. Such kind of sample should be deleted in advance.

2021.06 update: we developed function that could be able to automatically detect unqualified input files and inform the user of result. Furthermore, we set a parameter as quality control, which could set parameters subjectively by user. 

Should there be any problem, please feel free to contact: mengm5@mail2.sysu.edu.cn or liuzbbest@163.com
