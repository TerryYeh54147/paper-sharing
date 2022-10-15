# Concept Drift Detection via Boundary Shrinking

[back](../README.md)

- [Concept Drift Detection via Boundary Shrinking](#concept-drift-detection-via-boundary-shrinking)
  - [Abtract](#abtract)
  - [Introduction](#introduction)
  - [Related Work](#related-work)
  - [Methods](#methods)
    - [A. CDDBS與其飄移檢測機制](#a-cddbs與其飄移檢測機制)
    - [B. 一種訓練CDDBS中檢查器模型所特別設計的方法](#b-一種訓練cddbs中檢查器模型所特別設計的方法)
    - [C. 一種適用於CDDBS的飄移檢測和辨識演算法](#c-一種適用於cddbs的飄移檢測和辨識演算法)
  - [References](#references)
    - [*Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift*](#can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift)
    - [*Failing loudly: an empirical study of methods for detecting dataset shift*](#failing-loudly-an-empirical-study-of-methods-for-detecting-dataset-shift)
    - [*Sand: Semi-supervised adaptive novel class detection and classification over data stream*](#sand-semi-supervised-adaptive-novel-class-detection-and-classification-over-data-stream)
    - [*A semi-supervised based framework for data stream classification in non-stationary environments*](#a-semi-supervised-based-framework-for-data-stream-classification-in-non-stationary-environments)
    - [*Unsupervised drift detector ensembles for data stream mining*](#unsupervised-drift-detector-ensembles-for-data-stream-mining)

> note
> 
> 強調CDDBS可以unlabeled，並同時可抓到標移發生
> 對於資料維度的適應性高

## Abtract

數據分布的變化稱之為 **概念飄移(Concept Drift)**，該現象通常會降低訓練好的model性能。該篇主要是提出了一種非監督式的概念飄移檢測法(Concept Drift Detection)，其中有個檢查模型稱為 **Concept-Drift Detection via Boundary Shrinking(CDDBS)**。作者透過訓練讓檢查模型在某個類別的分類區域中故意縮小決策邊界，且該邊界還可以在不使用label的方式去檢測出飄移發生的位置。該論文是透過一個簡單的數值資料集、幾個公共合成基準資料集和具有高維真實圖像的CIFAR-10資料集來評估CDDBS的效果。


## Introduction

處理數據變化是確保模型品質的關鍵。而數據分布的變化稱之為 **概念飄移(Concept Drift)**，它通常會讓訓練好的模型性能降低。在近期一項[實證研究](#can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift)中表示，分布飄移會降低使用不同演算法的分類 model 精準度，而原因是模型是基於非飄移的資料集去最小化它的特定損失。

概念飄移檢測的重點除了檢測之外，還要辨識發生的地點、方法和時間，這些對re-train model很有效，都是檢測到飄移後需要仔細研究的目標。而使用多個分類器或模型的集成方法是辨識概念飄移另一種有用的方法，不過這些並不能同時滿足這種辨識特性與篇移檢測的靈敏度。[6](#failing-loudly-an-empirical-study-of-methods-for-detecting-dataset-shift)

而為了在飄移檢測中獲得上述特性，作者提出一種非監督式的概念飄移檢測方法，其具有一組 **通過邊界收縮的概念篇移檢測(CDDBS)** 的檢測器模型。

## Related Work

過去30年有很多種概念飄移檢測方法被提出，作者是專注於使用需多與CDDBS最相關的分類器或檢測器的集成飄移檢測方法。

通常，集成飄移檢測是結合分類器或檢測器得到的分數與策略去檢測和/或調整概念標移。像是Bagging, Boosting 就是經典的演算法。

然而作者研究了很多像是SAND[12](#sand-semi-supervised-adaptive-novel-class-detection-and-classification-over-data-stream), DyDaSL[13](#a-semi-supervised-based-framework-for-data-stream-classification-in-non-stationary-environments), EDFS[14](#unsupervised-drift-detector-ensembles-for-data-stream-mining)等等的檢測方法都沒有滿足飄移的領敏度和辨識特性，且都只使用低維度的數值數據進行評估。而CDDBS則可以靈敏地檢測概念飄移，又不須label，並且對維度的適應性很高。

> 補充
>
> 感覺這邊只是要強調CDDBS比其他傳統的方法強而已XD

## Methods

### A. CDDBS與其飄移檢測機制

![CDDBS overview](CDDBS%20overview.png)

當輸入一批未標記的Input之後， $f_{ori}$ 和 $\hat f_i$ 就會分別去計算分類分數。然後通過比較這些分數，CDDBS會輸出飄移檢測與辨識結果。

![explain the mechanism of drift detection and identification with the inspector models used in CDDBS by using a simple binary classification problem](./explain%20by%20binary%20classification%20problem.png)

*上半部是原模型，下半部是檢查器 $\hat f_0$，而 $\hat f_0$縮小了 $R_0$的決策邊界，因此在 $R_0$中發生的飄移反應靈敏*

通常在非監督是的分類中，Input的class label通常是透過訓練過的模型所獲得的某些分數(e.g. confidence)來預測。當input越接近boundary的時候分數就會越低。因此通過 $\hat f_0$ 獲得的分數會隨著 $R_0$ 中發生的input 分布而敏感地變化。

### B. 一種訓練CDDBS中檢查器模型所特別設計的方法

![design procedure to train the inspectormodels](explain%20by%20binary%20classification%20problem.png)

利用分類分數來挑選除了靠近原模型的決策邊界樣本跟錯誤樣本之外的樣本訓練檢查器模型。但因Confidence通常會被放定義為最後一層的輸出值，幾乎都會被變成0或1，並無線性關係，所以作者引入了種叫sureness的新度量，它可表示Input與邊界的距離。

$$
sur(X) := \phi(s_{1st}(X)) - \phi(s_{2nd}(X))
$$

此外，$\phi(s_{1st}(X))$, $\phi(s_{2nd}(X))$ 是所有類別中關於 X 的第一和第二高分類分數，而為了讓sureness在特徵空間中具有線性，兩個值得差比需要在任何空間中相等。

作者是使用損失函數來計算這些差值。而他們發現特別是如果是使用對數函數還最小化對數損失，則關於X的損失表示維 $-\log(\eta_i(X))$，其中 $\eta_i(X)$ 是將其預測為i類的機率。而這時 $\eta_i(X)$ 就具有可加性，因此他們的差異在訓練模型的特徵空間中的任何地方都會是相等的。

> note
> 
> 定義一個sureness的新度量出來，並且導證它的線性特徵

![algorithm1](algorithm%201.png)

### C. 一種適用於CDDBS的飄移檢測和辨識演算法

![algorithm2](algorithm%202.png)


## References

### *Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift*

### *Failing loudly: an empirical study of methods for detecting dataset shift*

### *Sand: Semi-supervised adaptive novel class detection and classification over data stream*

### *A semi-supervised based framework for data stream classification in non-stationary environments*

### *Unsupervised drift detector ensembles for data stream mining*