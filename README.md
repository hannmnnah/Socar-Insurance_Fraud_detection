<center><img width="1000" alt="스크린샷 2021-01-20 오후 3 10 46" src="https://user-images.githubusercontent.com/42338386/105134387-b26dd680-5b31-11eb-8613-daebd01f7fdc.png"></center>
 

## Short Description
- '뒷쿵', '공갈' ,'ㄷㅋ' 등으로 불리우는 보험 사기, 지난 2019년 기준 피해액은 8090억원(출처:금융감독원)에 달합니다.  
렌터카 사고는 렌터카 업체의 보험료만 올라가고, 가해자의 자차 보험료에는 아무런 피해를 주지않는 점 때문에 렌터카가 범행의 수단이 되고 있습니다.  
본 팀은 13000여개의 사고 데이터를 분석하여 보험 사기 사고를 예측했습니다.  
  
  해당 데이터 셋은 두 클래스가 1:379 (fraud-34:normal-12879)의 비율로 수집된 데이터로, 불균형적 데이터 셋입니다.  
    
   상세한 도메인 조사 및 EDA를 통한 feature engineering, 합리적인 feature selection, fraud 데이터 셋에 적합한 Sampling 알고리즘 구축, 최적의 모델 파라미터 튜닝, fraud 데이터 학습 사례 분석에 집중하여 과제를 해결했습니다. 

## keyword
Fraud Detection, Re-Sampling, Imbalanced Data, Clustering


## Built With

- [김경한] : EDA / feature select 알고리즘 구축 / train model & tuning : prediction 2 / 발표
	- https://github.com/darenkim
- [서기현] : EDA / 도메인 research / train model & tuning : prediction 1 / 발표
	- https://github.com/seogihyun
- [장한아] : EDA / fraud 데이터 학습 사례 분석 / train model & tuning : prediction 3 / 발표 및 Readme 작성
	- https://github.com/hannmnnah
- [정현석] : Advisor | FastCampus project manager
- [조용하] : Advisor | FastCampus project manager

-------------------------------------------------------------------------------------------------------------------------------

## 목차

1. 문제 정의 
2. 구조
3. Kick Insight
4. 모델 성능 결과 분석
5. 배운 점
6. 참고문헌


## 1. 문제 정의

### - 배경  : 
- __렌터카 보험사기란?__  
	
	보험금, 합의금을 얻을 목적으로 렌터카를 이용하여 고의 사고를 내는 행위로,  2016년부터 꾸준히 8만여명씩 적발되다가 , 지난 2019년에는 9.3만명으로 역대 최고치 기록(출처:금융감독원, 보험 사기 전체 기준)하였습니다. 적발 금액은 2019년 기준 8,090억원으로 피해액이 상당합니다.  
	
	보험 사기를 적발하지 못하여 해당 사기건마다 보험료가 지급될 시 전체 보험 가입자의 보험료가 올라가는 등, 상당한 금액의 피해가 예상됩니다.  
	
	Socar의 무한한 안녕과 평안을 위해 렌터카 예약 데이터를 통해 보험 사기를 미리 예측하거나, 사고 후 집계된 데이터를 통해 보험 사고를 발견하는 모델은 보험 사기에 대응하기 위한 기반을 만들어줍니다.  
	
	<img src='https://user-images.githubusercontent.com/42338386/104930448-81ce5580-59e8-11eb-82b0-cf7fbf853ef1.png' width='70%'></img>
		
- __Fraud Detection Model__  
	  
	금융 사기, 보험 사기 등의 데이터는 한 가지 공통된 특징을 가지고 있습니다.  
	
	일반 거래 데이터와 사기 거래 데이터의 비율이 심하게 치우친 불균형한 데이터 셋이라는 것입니다.  
		
	본 팀이 분석한 데이터 셋은 트레인 데이터 셋 클래스 비율이 1:379 (fraud-34:normal-12875)인 불균형 데이터 셋입니다. fraud 클래스의 수가 현저히 작아 모델이 학습 불가능합니다. 따라서 오버 샘플링을 포함하여 적절한 Re-Sampling 모델을 선택하여 두 클래스 간의 균형을 맞춰주는 것이 중요합니다.  
	
	<img src='https://user-images.githubusercontent.com/42338386/107126537-ded86f80-68f3-11eb-8f0f-84196b2de610.png' width='50%'></img>
	
	

### - 해결과제 : 
- __'34개의 train-fraud 데이터를 어떻게 학습시킬 것인가'__
	
	1. EDA를 통해 데이터의 노이즈, 아웃라이어 등을 파악해야합니다. 노이즈라고 판단되는 데이터를 정제할 시 효과적인 under sampling의 결과를 기대할 수 있습니다.
	2. 또한 EDA를 통해 데이터의 의미를 다시 파악하고, 효과적인 방법으로 custom할 방법을 모색합니다. 연속형 데이터를 구간을 나눠 명목형으로 바꾸는 시도가 이에 속합니다.
	3. 다양한 Resampling model을 적용하고, 분석하여 데이터 셋에 최적화된 Resampling 모델을 찾아야합니다.
	4. 도메인 조사, EDA, 관련 사례 분석과 같은 선행적 자료조사 뿐만 아니라, 모델링 결과를 분석하여 데이터 셋의 특징을 파악하는 경험적 분석이 요구됩니다. raw 데이터를 사용한 것이 아니라, 오버 샘플링된 데이터셋을 학습에 사용했기 때문에, 샘플링된 데이터 셋이 raw 데이터를 정확히 반영하는지, 샘플링 과정 속에 데이터의 노이즈가 생기진 않았는지 등을 결과분석을 통해 확인해야합니다. 


### - 모델 성능 평가 지표 선정 : __" 높은 recall과 동시에 높은 accuracy"__

	* recall : 예측 보험 사기 / 실제 보험 사기
	* accuracy : 예측한 보험 사기 + 예측한 일반 사고 / 전체 데이터
		
보통 불균형한 정도가 심한 데이터에서 사용하는 모델 성능 평가 지표는 recall 입니다.   
전부 다 normal 사고라고 예측해도 accuracy는 1에 가까운 값이 나오는 탓입니다.  

따라서, 소수 클래스인 관계로 모델이 학습하기 어려운 fraud 데이터 셋을 학습시켜 fraud를 예측해내고, Recall을 올리는 것이 첫번째 목표입니다.  
Recall을 달성한 후, normal, fraud 두 클래스의 예측률을 모두 높이기 위해 섬세한 Data Cleaning, Data Sampling, 모델 파라미터 조정이 필요합니다.  
이를 평가하는 지표로 accuracy를 사용합니다.  

궁극적으로 높은 recall과 동시에 높은 accuracy를 달성하는 것이 본 팀의 목표입니다.  




## 2. 구조

![Architecture](https://user-images.githubusercontent.com/42338386/105468949-ea644d80-5cda-11eb-9caa-e490a4c60527.png)


## 3. Kick Insight

### 3-1. __20대 운전자가 많은 fraud, 20대 운전자는 Fraud 사고일까?__  

  fraud 데이터 셋만 EDA 했을 때, 대부분의 사기 데이터는 20대, 쏘카를 처음 이용하는 이용자, 법인이 아닌 개인 등의 모습을 보인다는 것을 알게 되었습니다.  
그렇다면 20대 이용자는 대부분 fraud라고 분석해도 될까요? 아닙니다. 다음의 countplot을 보면, fraud의 특성이 normal의 특성이기도 한 모습을 볼 수 있습니다.
feature 14개가 유사한 모습을 보입니다. 

  즉, '전체 이용자의 사고 경향'과 'fraud의 경향'이 유사합니다.  
  
<img src='https://user-images.githubusercontent.com/42338386/105456196-175b3500-5cc8-11eb-936f-77461b175134.png' width='50%'></img>

	
### 3-2.__일반 이용자의 사고와 fraud 이용자의 사고가 상당히 유사한 모습을 보이는 데이터 셋, 어떤 샘플링을 사용해야할까?__  

  우린 EDA를 통해 normal 데이터 셋과 fraud 데이터 셋의 특성이 유사한 모습을 보인다는 것을 확인했습니다.  

  때문에, Over-Sampling, Combined-Sampling에 속한 다양한 모델의 결과를 분석해본 결과, fraud 데이터 셋과 normal 데이터 셋의 경계를 정리해주는 샘플링 모델이 가장 적합했습니다.  
BorderlineSmote와 TomekLinks, EditedNearestNeighbours이 그에 해당합니다. 다음의 scatterplot은 본 팀의 모델에서 가장 성능이 좋았던  샘플링 모델을 요약합니다.  

  SMOTE(random_state=13, k_neighbors=26, sampling_strategy=1)는 fraud 데이터 간의 간격을 채우는 방식으로 샘플링 되어, 원 데이터와는 다른 양상을 보입니다.  
반면 BorderlineSMOTE(random_state=13, k_neighbors=10,sampling_strategy=1)의 경우  본 데이터 셋의 fraud와 가장 유사하게 샘플링되었다는 점을 알 수 있습니다.  

  두 샘플링 방법의 모델링 결과는 fraud 데이터를 가장 유사하게 샘플링하는 방법이 예측률을 높이는 방법임을 확인했습니다. 

		- {SMOTE : [DecisonTreeClassifier | acc : 0.85, recall : 0.14]}
		- {BorderlineSMOTE : [Logistic Regression | acc : 0.65, recall : 0.85]} 

![scatterplot](https://user-images.githubusercontent.com/42338386/107125670-abdfad00-68ee-11eb-8fbd-de3b5d3e44db.png) 

  다음의 scatterplot은 BorderSmote와 Under-Sampling모델을 혼합하여 사용한 결과를 요약합니다.  
BorderlineSMOTE로 오버샘플링만 했을 때보다, accuracy가 올라간 것을 보아, fraud 인접의 normal 사고를 언더샘플링해주는 것이 모델의 예측률을 높인다는 사실을 확인하였습니다.
			
		- {Tomek_all : [Logistic Regression | acc : 0.74, recall : 0.71]}
		- {Tomek : [Logistic Regression | acc : 0.67, recall : 0.85]} 
		- {ENN_12 : [Logistic Regression | acc : 0.71, recall : 0.71]}
		- {ENN_13 : [Logistic Regression | acc : 0.66, recall : 0.85]} 

![combined](https://user-images.githubusercontent.com/42338386/107125886-c49c9280-68ef-11eb-8cf5-163b6756db94.png)




### 3-3.__0 ~ 1억까지 범위가 너무 큰 S15, 어떻게 커스텀해야 학습시킬 수 있을까?__  

  아래 boxplot과 jointplot의 단위는 1000만원입니다.  
  
  범위는 0~1억이지만, s14와 s15의 jointplot을 보았을 때 대부분의 데이터 셋이 100만원 이하에 몰려있는 것을 알 수 있습니다.  
  
  보통 큰 사고는 일어나지않으며, 특히 fraud일 때는 더욱 더 큰 사고를 계획하지 않는다고 해석하였습니다.  
  
  따라서 데이터의 노이즈를 줄이고 의미를 명확히 하기 위해 연속형 변수들을 3개(경미한 사고 =0,보통 사고 <=125만,대형 사고>125만)구간으로 나눠 명목형 변수로 custom하였습니다.  
  
  같은 데이터 셋 기준, 명목형 custom 여부에 대한 모델 결과 지표를 보았을 때, 성능이 확연히 좋아졌다는 것을 확인할 수 있습니다.


		- {original : [DecisionTreeClassifier | acc : 0.95, recall : 0.0]}
		- {명목형으로 custom : [DecisionTreeClassifier | acc : 0.77, recall : 0.85]} 


![repair_insure](https://user-images.githubusercontent.com/42338386/105408068-d801f980-5c71-11eb-8c8e-e70159ed5b88.png)


   
### 3-4.__train-fraud 데이터 34개 중 아웃라이어 1개, 어떻게 학습시켜야할까 ? : SCUT__  

  train-fraud 데이터의 아웃라이어를 어떻게 다뤄야할까 고민하였습니다.  

  34개 데이터 중에 3~5개는 보통의 fraud 데이터와 떨어진 아웃라이어였습니다.  
  
<img src='https://user-images.githubusercontent.com/42338386/107127291-8b1c5500-68f8-11eb-99d4-e797375c1707.png' width='50%'></img>


  이를 고려하지않고 샘플링한다면, 소수 클래스의 '소수'인 아웃라이어는 샘플링 모델이 학습하지 못합니다.  
  SMOTE를 통해 fraud 데이터 셋을 34개에서 12845개로 오버샘플링할 때, train-fraud의 아웃라이어인 's3'=5인 데이터는 샘플링 이후에도 1개인 결과가 이를 말해줍니다.  
    
  fraud data set을 다루는 사례 논문들을 조사한 결과 SCUT 알고리즘을 적용해보기로 하였습니다. 
  
  	* SCUT Algorithm  
  	- 비지도 학습인 군집 모델링을 바탕으로 샘플링 하는 방법입니다.
	- Multi-Class Imbalanced Data Classification using SMOTE and Cluster-based Undersampling Technic
  

![scut](https://user-images.githubusercontent.com/42338386/107127038-cddd2d80-68f6-11eb-92a6-ce833b14dbe7.png)
	
  
  기존에 fraud, normal 두 클래스로 나눠졌던 라벨 대신 K-means clustering을 통해 학습한 군집 0:fraud, 1:fraud, 2:fraud, 3:normal 네 클래스의 라벨을 사용하여 모델링합니다.  
    
   SMOTE 샘플링 시 소수 클래스 내에서의 소수, 즉 train-fraud 데이터를 학습하지 못하는 문제를 보완합니다.  
   
   K-means clustering을 통해 소수 클래스에서의 다수, 소수 클래스에서의 소수1, 소수 클래스에서의 소수2 이렇게 3가지 클래스로 분리하여 소수 클래스의 '소수'클래스 역시 학습 가능해집니다.  
     
   일례로 단순 SMOTE로 샘플링 시에는 샘플링 이후에도 1개였던 train-fraud 데이터의 아웃라이어 's3'=5 데이터가, SCUT을 적용할 때에는 2869개로 오버 샘플링됩니다. 클래스를 나눠줌으로써 SMOTE 모델이 train-fraud 데이터의 아웃라이어를 학습 가능해졌기 때문입니다.
  
  accuracy가 대폭 상승했다는 점에서 유의미합니다. fraud 데이터의 아웃라이어를 학습함으로써 normal 사고 예측 정확도가 높아진 것으로 추정합니다.
  

		- {단순 SMOTE : [Logistic Regression | acc : 0.57, recall : 0.42]}
		- {SCUT : [DecisionTreeClassifier | acc : 0.81, recall : 0.42]} 




## 4. Result
- 본 팀은 인사이트를 종합하여 최고의 모델을 선정하였습니다.

- result

|model name| train accuracy | train precision | train recall | test accuracy | test precision | test recall | 
|:----------:|:-------------:|:---------------:|:---------------:|:---------------:|:---------------:|:---------------:|
|DecisionTreeClassifier|0.825259|0.0006611|0.515151|0.77399|0.00930|0.81541

-전처리?






## 5. 보완할 점 & 배운 점

#### * 보완할 점  

  __test set에 대한 자세__  
  	프로젝트 진행 시 데이터의 노이즈를 제거하여 학습 성능을 높이기 위해 데이터 셋의 아웃라이어를 제거하는 임의적인 Under-Sampling을 진행하였습니다.  
	이 과정에서 test set을 유실하는 일이 발생하였습니다.  
	프로젝트를 마무리하며 최종발표를 할 때 test set은 무슨 일이 있어도 데이터를 유실하거나 과하게 변형되는 일이 발생하면 안된다는 피드백을 받았습니다.  
	이를 통해 test set에 대한 자세를 배웠습니다.  
	추후 프로젝트 보완을 통하여 test set의 유실된 데이터를 찾아 모델에 적용하였습니다.  
	앞으로 test set의 데이터는 유실되는 일이 없도록 더욱 더  계기가 되었습니다.
	
	
	
  	
#### * 배운 점   
  
  __1. Imbalanced data에 대한 다양한 Sampling 기법__    
    
   * imblearn 패키지 :  
	- imblearn 패키지의 Over-Sampling(SMOTE, BorderlineSMOTE, ADASYN, Random-OverSampling) , Under-Sampling(ENN, CNN, 		Nearmiss, RandomUnderSampling, Tomeklinks), 그리고 pipeline을 통한 Combined-Sampling까지 데이터 셋에 최적화된 샘플링 모델을 찾기위해 각 모델들을 공부하고 적용하였습니다.     
	
  * 데이터 내 노이즈, Outlier 제거를 통한 Under-Sampling    
    
  * SCUT : 소수 클래스를 K-means Clustering 방법을 통해 multi class로 분리하고, 샘플링하는 방법
        
	
   __2. Feature Selection 기준에 대한 다양한 접근법__  
   	  
  * EDA 및 도메인 조사 내용을 기반으로 한 Feature Seletion  
	  
  * Feature 랜덤 drop 실험을 통한 Feature 간 상관관계 분석
	  
	  
   __3. 다양한 모델링 기법에 대한 경험적 지식__  
   		  
  * Decision Tree가 성능이 좋은데 RandomForest는 왜 성능이 안좋을까?  
			  
	- 의사결정나무를 시각화해본 결과, 두 모델은 너무도 다른 feature를 선정하여 클래스를 분류하였습니다.  
	- Imbalanced data set의 모델링 과정에서 발생한 현상이니, 비정상적인 현상은 아니라고 할 수 있습니다.  
	- 이후, DecisionTreeClassifier에서 사용한 feature들만 RandomForest에 넣었을 때는 DecisionTreeClassifier와 유사한 성능을 보였습니다.  
		  
  * Multi-Classes 분류일 때, Support Vector Machine Classifier 커널 무한 로딩 문제  
			  
	- 보통 SVC가 연산량이 많아 시간 소요가 되는 모델이라는 점을 감안해도, 과하게 오래 걸리는 문제가 발생했습니다.  
	- 연산량을 줄어주기 위해 데이터 셋 스케일링을 적용하였더니, 비교적 짧은 시간에 모델 성능을 확인할 수 있었습니다.  
	- 더불어, SVC보다 연산량이 적은 SVR 모델을 적용하였더니, 이 역시 비교적 짧은 시간에 모델 성능을 확인할 수 있었습니다.  
		  
  * Support Vector Machine : randomseed에 따라 성능이 많이 달라지는 현상     
  
  	- 데이터 셋, 모델마다 randomseed에 따라 성능이 많이 달라지는 현상이 존재할 수 있다는 점을 학습하였습니다.  
	
__4. Imbalanced Data Set 모델 성능 지표 해석 및 평가__    
  
  * Imbalanced Data Set의 경우 일반적인 데이터 셋과 달리 Recall을 확보해야합니다.    
    
       - 다수 클래스로 몰린 예측을 해도, 데이터 셋의 대부분이 다수 클래스이기 때문에 accuracy가 높기 때문입니다.    
         따라서 Imbalanced Data set의 경우, 높은 recall과 동시에 높은 accuracy를 확보하는 것이 중요합니다.
  
  
__5. 비대면 업무 시 의사소통 방법__  
  
  * discord를 통해 매일 오전에 작업 계획을 공유하고, 밤 10시 경에 zoom 회의를 통해 작업 결과 공유
  * github을 활용하여 파일 공유 및 결과 업데이
  
  
  
 
## 6. 참고 문헌

- Jalal Ahammad, Nazia Hossain, January 2020, Credit Card Fraud Detection using Data Pre-processing on Imbalanced Data - both Oversampling and Undersampling(ICCA 2020: Proceedings of the International Conference on Computing Advancements, Article No.: 68, pp 1–4)
- 정한나, 이정화, 전치혁,March 2010, 불균형 이분 데이터 분류분석을 위한 데이터마이닝 절차 (포항공과대학교 산업경영공학과, Journal of the Korean Institute of Industrial Engineers Vol. 36, No. 1, pp. 13-21)
- Astha Agrawal , Herna L. Viktor and Eric Paquet ,2015, SCUT: Multi-Class Imbalanced Data Classification using SMOTE and Cluster-based Undersampling (In Proceedings of the 7th International Joint Conference on Knowledge Discovery, Knowledge Engineering and Knowledge Management (IC3K 2015) - Volume 1: KDIR, pages 226-234 ISBN: 978-989-758-158-8)
  
      



