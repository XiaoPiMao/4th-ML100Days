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
### Day17
連續變數離散化好處: 可能性變少, 受離散值影響少  
離散化方法: 等寬劃分 pd.cut, 等頻劃分 pd.qcut, 聚類劃分, 自訂劃分
### Day18
連續變數離散化實際練習
### Day19
Subplots
### Day20
Heatmap & Grid Plot
### Day21
模型初體驗: Logistic Regression
### Day22
特徵工程: 將數據轉化為可評分的目標資料
### Day23
去偏執化: log1p, sqrt, boxcox
### Day24
One Hot Encoding 所需儲存空間和計算時間較大  
| 編碼種類 | 儲存空間/機算時間 | 適用學習模型 |
| :------------- | :----------: | :-----------: |
| Label Encoding | 小 | 非深度學習 |
| One Hot Encoding | 大 | 深度學習 |
### Day25 特徵工程-均值編碼
類別特徵與目標有明顯相關時可以使用均值編碼  
均值編碼會有Overfitting的問題  
可以用平滑式調整加以修正  

![](http://latex.codecogs.com/gif.latex?\frac{\overline{t}\times&space;tn&space;&plus;&space;\overline{a}\times&space;x}{tn&plus;x})  
### Day26 類別型特徵-進階處理
* 記數編碼: 該欄位類別比數與目標正相關時
* 特徵雜湊: 欄位類別數過多但需要列入考量的欄位(效果還是不好)
### Day27 時間型特徵
常見時間型特徵: 把時間戳記拆分成年,月,日,...  
可以用sin/cos的方式計算週期循環  
e.x.一年四季溫度正: 熱,負: 冷  

![](http://latex.codecogs.com/gif.latex?-cos((m\div&space;6&plus;d\div&space;180)\times&space;\pi))
### Day28 特徵組合-數值與數值組合
特徵工程決定機器學習的上限  
特徵工程的關鍵在領域知識
### Day29 類別與數值組合
類別與數值的組合可以用群聚編碼(Group by Encoding)
||Mean Encoding|Group by Encoding|
|:---|:---:|:---:|
|對象|Target|其他數值欄位|
|可能Overfitting|O|X|
|需要平滑化|O|X|
> 機器學習的特徵是***寧濫勿缺*** 

### Day30 特徵選擇
特徵選擇是減少特徵  
特徵選擇三種方法
* Filter
* Wrapper
* Embedded

|  |計算速度|共線性|特徵穩定性|
|:---|:---:|:---:|:---:|
|相關係數Filter|快速|無法排除|穩定|
|Lasso崁入法|快速|能排除|不穩定|
|GDBT崁入法|較慢|能排除|穩定|

### Day31 特徵評估
樹狀模型的重要性可以用分支次數、分支覆蓋度和損失降低量3種  
特徵重要性可以在缺乏領域知識時作為評估的依據  
機器學習中的優化循環:  
... -> GDBT Xgboost模型擬合 -> 根據特徵重要性做刪減 -> 交叉驗證看是否改善 -> ...  
### Day32 分類型特徵優化-葉編碼
多個分類預測結果可以取sigmoid(x)算回機率  
Leaf Encoding可以將樹狀的葉節點做離散化得到新的特徵，一般會把這個新特徵用Logistic Regression重新得到機率
### Day33 機器如何學習
1. 定義模型
2. 評估模型的好壞(Loss Function)
3. 找出最好的參數

### Day34 訓練/測試集切分
一定要分訓練和測試集，要不然完全不知道訓練的測試結果，可能是underfitting或overfitting或特徵工程有待加強或需要更換一個模組

```python
#一般的切分
sklearn.model_selection.train_test_split
```
K-fold Cross Validation
|執行|是否為驗證|是否為驗證|是否為驗證|是否為驗證|是否為驗證|
|:---:|:---:|:---:|:---:|:---:|:---:|
|1|O|||||
|2||O||||
|3|||O|||
|4||||O||
|5|||||O|

```python
#K-fold Cross Validation
sklearn.model_selection.KFold
```

### Day35 Regression vs Classifcation
回歸問題的目標是一個實數 e.x. 身高  
分類問題的目標是哪一個分類 e.x. 矮 中等 高  
Multi-label 是指每一筆資料可以為多個分類的組合

### Day36 評估指標選定
* 回歸問題: 預測值 Prediction 與實際值 Ground truth 的差異
	* MAE
	* MSE
	* R-square
* 分類問題: Prediction 與 Ground truth 的正確程度
  * AUC, Area Under Curve
  * F1-Score

    * Precision

      ![](http://latex.codecogs.com/gif.latex?\frac{TP}{TP&plus;FP})

    * Recall

      ![](http://latex.codecogs.com/gif.latex?\frac{TP}{TP&plus;FN})

### Day37 Linear Regression & Logistic Regression
Linear Regression 適用於回歸問題

![](http://latex.codecogs.com/gif.latex?\widehat{Y}_{i}=b_{0}&plus;b_{1}X_{i}})

Logistic Regression 適用於分類問題

![](http://latex.codecogs.com/gif.latex?P(y|x)=\frac{1}{1+e^{-yw^{T}x}})