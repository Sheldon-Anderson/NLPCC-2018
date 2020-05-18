# NLPCC-2018
Here are the codes for NLPCC 2018 paper: Abstractive Summarization Improved by WordNet-Based Extractive Sentences

# 联邦学习

基于 [谷歌联邦学习论文](https://arxiv.org/abs/1602.05629)，我们实现了一套联邦学习框架（当前不包括加密算法和分布式计算），
与此同时，我们基于乳腺癌组织病理学数据集 [BreaKHis](http://open.baai.ac.cn/data-set-detail/221/20) 进行了图像分类的实验。  

## 环境配置
* python==3.7.6  
* Keras==2.3.1  
* scikit-learn==0.22.1  
* tensorflow-gpu==2.2.0  
* timm==0.1.26  
* torch==1.5.0+cu92  
* torchsummary==1.5.1  
* torchvision==0.6.0+cu92

## 项目结构
```
.  
├── LICENSE  
├── README.md  
├── client.py  
├── figs  
├── logs  
├── main.py  
├── models  
│   └── nets.py  
├── requirements.txt  
├── run_centroid.sh  
├── run_direct.sh  
├── run_fed.sh  
├── save  
├── server.py  
├── utils  
│   ├── __init__.py  
│   ├── data.py  
│   ├── options.py  
│   └── utils.py  
```

## 运行方式
联邦学习模型训练和预测：
> bash run_fed.sh

集中学习模型训练和预测：
> bash run_centroid.sh

各个节点独立模型训练和预测：
> bash run_direct.sh

## 实验说明

### 参数配置
参数配置参考 [options.py](utils/options.py)  
* server_ep: int类型，默认值为8，服务器端监控训练轮次（SE）  
* user_num: int类型，默认值为6，客户端用户数量（K）  
* client_ratio：str类型, 默认值为'536,500,500,500,500,500,500,500,500,500,500'，不同客户端用户的数据比例)  
* client_ep: int类型，默认值为5，客户端训练轮次（CE）  
* client_bs: int类型，默认值为32，客户端批大小（CB）  
* bs: int类型，默认值为32，模型测试批大小  
* lr: float类型，默认值为0.001，学习率  
* model: str类型，默认值为'densenet201'，模型名称（M）  
* mode: str类型，默认值为'fed'，模型训练和测试的模式（fed / centroid / direct）  
* dataset: str类型，默认值为'breakhis'，数据集名称（D）  
* data_folder: str类型，默认值为'/.../AllData'，数据集保存目录  
* train_prop: float类型，默认值为0.7，训练集占数据集的比例  
* image_size: int类型，默认值为224，图像矩阵的宽和高  
* num_classes: int类型，默认值为2，分类类别数量  
* num_channels: int类型，默认值为3，图像通道数量  
* gpu: int类型，默认值为0，GPU ID，-1代表CPU  
* verbose: action='store_true'，当读取参数中出现--verbose时触发  
* log_interval: int类型，默认值为50，训练间隔多少个批大小打印出日志信息  
* seed: int类型，默认值为1，随机种子  

### 实验指标
实验指标参考 [utils.py](utils/utils.py)  
* accuracy: 准确率（accuracy，ACC），ACC = (TP + TN) / (TP + TN + FP + FN)  
* patient_level_accuracy: 基于病人的平均正确率  
* se: 敏感性（sensitivity, SE）= TP / (TP + FN)，即召回率，所有真实正例中预测出了多少真实正例  
* sp: 特异性（specificity，SP）= TN / (TN + FP)，所有真实负例中预测出了多少真实负例  
* ppv: 正确率（precision，P）= TP / (TP + FP)，即阳性预测值（positive predictive value，PPV）= TP / (TP + FP)，所有预测出来的正例中有多少是真的正例  
* npv: 假性预测值（negative predictive value，NPV）= TN / (TN + FN)，所有预测出来的负例中有多少是真的负例  
* dor: 诊断比值比（Diagnostic Odds Ratio，DOR）= (TP * TN) / (FP * FN)，即诊断优势比，是PLR+与NLR-的比值，即阳性似然比（TP / FP）与阴性似然比（FN / TN）的比值，反映诊断试验的结果与疾病的联系程度  
  * 取值大于1时，其值越大说明该诊断试验的判别效果较好  
  * 取值小于1时，正常人比患者更有可能被诊断试验判为阳性  
  * 取值等于1时，表示该诊断试验无法判别正常人与患者  
* f1: F1值（F1 Score）= 2 * ppv * se / (ppv + se)，精确率和召回率的调和均值  
* kappa: 一致性检验kappa系数 = (p0 - pe) / (1 - pe)，kappa计算结果为 [-1, 1]，但通常kappa是落在 [0, 1]，可分为五组来表示不同级别的一致性`  
  * [0.0, 0.20]：极低的一致性(slight)  
  * [0.21, 0.40]：一般的一致性(fair)  
  * [0.41, 0.60]：中等的一致性(moderate)  
  * [0.61, 0.80]：高度的一致性(substantial)  
  * [0.81, 1]：几乎完全一致(almost perfect)  

### 数据说明

### 模型说明


## 实验结果
<table>
    <tr>
        <th rowspan="2">
            <tr>accuracy</tr>
            <tr>40x, 100x, 200x, 400x</tr>
        </th>
        <th rowspan="2">
            <tr>patient level accuracy</tr>
            <tr>40x, 100x, 200x, 400x</tr>
        </th>
    </tr>
</table>

放大系数40x时：

|        Model        |   accuracy  | patient level accuracy |  se  |  sp  |  ppv  |  npv  |  dor  |  f1  |  kappa  |
| ------------------- | ----------- | ---------------------- | ---- | ---- | ----- | ----- | ----- | ---- | ------- |
| densenet201         |  %          |   %                    |      |      |       |       |       |      |         |
| resnet152           |  %          |   %                    |      |      |       |       |       |      |         |
| mobilenetv2_100     |  %          |                        |      |      |       |       |       |      |         |
| inception_resnet_v2 |  %          |                        |      |      |       |       |       |      |         |
| nasnetalarge        |  %          |                        |      |      |       |       |       |      |         |
| tf_efficientnet_b7  |  %          |                        |      |      |       |       |       |      |         |

放大系数100x时：

|        Model        |   accuracy  | patient level accuracy |  se  |  sp  |  ppv  |  npv  |  dor  |  f1  |  kappa  |
| ------------------- | ----------- | ---------------------- | ---- | ---- | ----- | ----- | ----- | ---- | ------- |
| densenet201         |  %          |   %                    |      |      |       |       |       |      |         |
| resnet152           |  %          |   %                    |      |      |       |       |       |      |         |
| mobilenetv2_100     |  %          |                        |      |      |       |       |       |      |         |
| inception_resnet_v2 |  %          |                        |      |      |       |       |       |      |         |
| nasnetalarge        |  %          |                        |      |      |       |       |       |      |         |
| tf_efficientnet_b7  |  %          |                        |      |      |       |       |       |      |         |

放大系数200x时：

|        Model        |   accuracy  | patient level accuracy |  se  |  sp  |  ppv  |  npv  |  dor  |  f1  |  kappa  |
| ------------------- | ----------- | ---------------------- | ---- | ---- | ----- | ----- | ----- | ---- | ------- |
| densenet201         |  %          |   %                    |      |      |       |       |       |      |         |
| resnet152           |  %          |   %                    |      |      |       |       |       |      |         |
| mobilenetv2_100     |  %          |                        |      |      |       |       |       |      |         |
| inception_resnet_v2 |  %          |                        |      |      |       |       |       |      |         |
| nasnetalarge        |  %          |                        |      |      |       |       |       |      |         |
| tf_efficientnet_b7  |  %          |                        |      |      |       |       |       |      |         |

放大系数400x时：

|        Model        |   accuracy  | patient level accuracy |  se  |  sp  |  ppv  |  npv  |  dor  |  f1  |  kappa  |
| ------------------- | ----------- | ---------------------- | ---- | ---- | ----- | ----- | ----- | ---- | ------- |
| densenet201         |  %          |   %                    |      |      |       |       |       |      |         |
| resnet152           |  %          |   %                    |      |      |       |       |       |      |         |
| mobilenetv2_100     |  %          |                        |      |      |       |       |       |      |         |
| inception_resnet_v2 |  %          |                        |      |      |       |       |       |      |         |
| nasnetalarge        |  %          |                        |      |      |       |       |       |      |         |
| tf_efficientnet_b7  |  %          |                        |      |      |       |       |       |      |         |

基于总体数据集时：

|        Model        |   accuracy  | patient level accuracy |  se  |  sp  |  ppv  |  npv  |  dor  |  f1  |  kappa  |
| ------------------- | ----------- | ---------------------- | ---- | ---- | ----- | ----- | ----- | ---- | ------- |
| densenet201         |  %          |   %                    |      |      |       |       |       |      |         |
| resnet152           |  %          |   %                    |      |      |       |       |       |      |         |
| mobilenetv2_100     |  %          |                        |      |      |       |       |       |      |         |
| inception_resnet_v2 |  %          |                        |      |      |       |       |       |      |         |
| nasnetalarge        |  %          |                        |      |      |       |       |       |      |         |
| tf_efficientnet_b7  |  %          |                        |      |      |       |       |       |      |         |
