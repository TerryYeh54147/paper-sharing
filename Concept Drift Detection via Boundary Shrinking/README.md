# Concept Drift Detection via Boundary Shrinking

[back](../README.md)

- [Concept Drift Detection via Boundary Shrinking](#concept-drift-detection-via-boundary-shrinking)
  - [Abtract](#abtract)
  - [Introduction](#introduction)
  - [Related Work](#related-work)
  - [Methods](#methods)
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

## References

### *Can you trust your model’s uncertainty? Evaluating predictive uncertainty under dataset shift*

### *Failing loudly: an empirical study of methods for detecting dataset shift*

### *Sand: Semi-supervised adaptive novel class detection and classification over data stream*

### *A semi-supervised based framework for data stream classification in non-stationary environments*

### *Unsupervised drift detector ensembles for data stream mining*