## 4th-ML100Days
第四屆機器學習百日馬拉松
### About Begin
一段機器學習的最初探險
### Day1
找到問題->初版嘗試->改進->分享->練習->實戰  
問題重要嗎?資料從哪裡拿?資料類型是什麼?用什麼指標去分析這個問題?
#### Code
產生隨機預測與誤差資料  
平均絕對誤差 Mean Absolute Error mae()  
平均平方誤差 Mean Squared Error mse()
### Day2
DL∈ML∈AI  
ML組成與應用:  
1. 監督式學習 Supervised Learning
2. 非監督式學習 Unsupervised Learning
3. 強化學習 Reinforcement Learning
### Day3
機器學習開方流程: 資料 -> 定目標和評估準則 -> 建立模型 -> 導入資料 
### Day4
EDA Exploratory Data Analysis 探索資料分析  
數據分析的流程: 數據清理 -> 特徵萃取 -> 資料視覺化 -> 建立模型 -> 驗證模型
### Day5
pandas.DataFrame  
讀取 [txt, json, mat, 圖片, npy, pkl]
### Day6
欄位變數可分: 離散變數, 連續變數  
DataFrame資料大致分: float64, int64, object:字串或其他  
Object類別分析: Label Encoding, One Hot Encoding
### Day7
資料特徵: 數值型, 類別型, 二元, 排序型, 時間型
### Day8
量化資料:
1. 計算集中趨勢: Mean, Median, Mode
2. 計算分散趨勢: Min, Max, Range, Quartiles, Variance, Standard Deviation

圖表化工具: matplotlib, seaborn
### Day9
透過分析和繪製圖表發現離群群值(Outlier)  
處理離群值的方式: 建新欄位紀錄, 取代, 整欄不用  
箱型圖 Box Plot, 經驗分布圖 ECDF Empirical Cumulative Density Plot, 直方圖 Histogram
### Day10
離群值處理方式: 捨棄離群值, 可設最大和最小範圍做取代或切割
### Day11
填補資料: median, quantiles, mode, mean  
標準化: z轉化, 空間壓縮
### Day12
填補缺失資料: median, mean, mode, 指定值, 預測值
標準化的意義: 平衡數值特徵間的影響力
### Day13
pd.DataFrame常用的操作: concat,cut,groupby...
### Day14
相關係數 Correlation Coefficient
### Day15
Correlation Coefficient實例  
當y值只有[0,1]時，可以將x,y反轉並以原y值當分組依據繪製Box Plot  
當y值範圍很大，大多數的y值都分布在偏小的範圍時，可以利用Log Scale來壓縮y值
### Day16
核密度函數 Kernel Density Estimation KDE