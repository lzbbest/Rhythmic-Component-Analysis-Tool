# Rhythmic-Component-Analysis-Tool

Tutorial of RCAT
(i) Prepare input data in predefined format (Fig.S1 A).
(ii) Import file (s) from file path by clicking ‘File’ and ‘Import’ button in menu, and then select file (s) to analyze (Fig.S1 B, ①②③). Users can import and select one file or several files to analyze at once.
(iii) Set parameters (Fig.S1 B, ④). ‘Points per day’ means total points of raw data collected in one day. For example, we collect one data point every ten minutes so the total points per day is 144 points (set default). ‘Analysis interval’ should be set for reducing affect of poor-quality data. Here we set interval as 24 - 132 Hr for accurate analyse, and default interval is 16-80 Hr. Finally, click ‘RUN’ button to execute program (Fig.S1 B, ⑤).
(iv) Results show on the interface and save automatically in input-file path, use ‘ResultFile’ button to directly access (Fig.S1 C, ⑥). If user wants to save files in another file path, choose one of results, and click ‘ResultSave AS’ button to save (Fig.S1 C, ⑦⑧).
(v) Output files (Fig.S1 D). The result are exported as CSV format which is easy to open and analyze. And result images are saved as PNG format. ‘summary.csv’ contains Amp (Amplitude), Period, Phase, RAE (Relative Amplitude Error) which are also shown on the interface. ‘PartA.png’ is for comparing curve of ‘raw data’ and curve of ‘detrend data’. ‘PartB.png’ is for comparing curve of ‘detrend data’ and curve of ‘fitted data’. ‘Raw data (disposed).csv’ is extracted the third (Time (days)) and fourth (counts/sec) columns from raw data and transform days to hours. ‘Detrend data.csv’ is data calculated by DTRNDNAL. ‘Fitted data.csv’ is final curve data.
(vi) Change the interval multiple times to get the best results.
* For other systems, we provide source code (python) for data analysis.



