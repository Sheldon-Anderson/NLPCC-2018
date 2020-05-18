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
为了跟其它论文的实验结果有可比性，我们把原始数据集 [BreaKHis](http://open.baai.ac.cn/data-set-detail/221/20) 随机打乱后按照7 ：3进行了训练集（5536条）、测试集（2373条）切分；在独立训练和联邦训练时，将训练集分为11个集合，分配给11个对应的客户端，其中第一个客户端得到536条数据，其余每个客户端得到500条数据。

### 模型说明
我们基于'densenet201'、'xception'、'resnet152'、'mobilenetv2_100'、'inception_resnet_v2'、'nasnetalarge'、'tf_efficientnet_b7'模型分别进行了实验。

## 实验结果
集中训练（Centroid Train）在测试集上的结果：
<div class="table">
<table border="1" cellspacing="0" cellpadding="10" width="100%">
<thead>
<tr class="firstHead">
    <th colspan="1" rowspan="2">Model</th> <th colspan="5">accuracy</th> <th colspan="5">patient level accuracy</th> <th colspan="5">se</th> <th colspan="5">sp</th> <th colspan="5">ppv</th> <th colspan="5">npv</th> <th colspan="5">dor</th> <th colspan="5">f1</th> <th colspan="5">kappa</th> 
</tr>
<tr class="twoHead">
    <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th><th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th>
</tr>
</thead>
<tbody>
<tr>
<td>densenet201</td>
<td>0.8804</td> <td>0.9180</td> <td>0.9068</td> <td>0.8960</td> <td>0.9001</td> 
<td>0.8834</td> <td>0.9128</td> <td>0.8998</td> <td>0.8973</td> <td>0.8990</td> 
<td>0.7586</td> <td>0.8556</td> <td>0.8140</td> <td>0.8155</td> <td>0.8112</td> 
<td>0.9255</td> <td>0.9442</td> <td>0.9450</td> <td>0.9335</td> <td>0.9369</td> 
<td>0.7904</td> <td>0.8652</td> <td>0.8589</td> <td>0.8509</td> <td>0.8416</td> 
<td>0.9119</td> <td>0.9398</td> <td>0.9251</td> <td>0.9158</td> <td>0.9231</td> 
<td>39.0612</td> <td>100.1987</td> <td>75.1359</td> <td>62.0551</td> <td>63.7764</td> 
<td>0.7742</td> <td>0.8603</td> <td>0.8358</td> <td>0.8328</td> <td>0.8261</td> 
<td>0.6929</td> <td>0.8023</td> <td>0.7708</td> <td>0.7574</td> <td>0.7561</td> 
</tr>
<tr>
<td>xception</td>
<td>0.8748</td> <td>0.8756</td> <td>0.8702</td> <td>0.8414</td> <td>0.8660</td> 
<td>0.8735</td> <td>0.8810</td> <td>0.8606</td> <td>0.8473</td> <td>0.8687</td> 
<td>0.6118</td> <td>0.6617</td> <td>0.6743</td> <td>0.6704</td> <td>0.6552</td> 
<td>0.9852</td> <td>0.9694</td> <td>0.9553</td> <td>0.9215</td> <td>0.9587</td> 
<td>0.9455</td> <td>0.9048</td> <td>0.8676</td> <td>0.8000</td> <td>0.8748</td> 
<td>0.8581</td> <td>0.8672</td> <td>0.8710</td> <td>0.8564</td> <td>0.8634</td> 
<td>104.7879</td> <td>62.0294</td> <td>44.2788</td> <td>23.8644</td> <td>44.1471</td> 
<td>0.7429</td> <td>0.7644</td> <td>0.7588</td> <td>0.7295</td> <td>0.7492</td> 
<td>0.6650</td> <td>0.6826</td> <td>0.6720</td> <td>0.6185</td> <td>0.6603</td> 
</tr>
<tr>
<td>resnet152</td>
<td>0.9047</td> <td>0.9091</td> <td>0.9413</td> <td>0.9132</td> <td>0.9170</td> 
<td>0.9086</td> <td>0.8832</td> <td>0.9333</td> <td>0.9171</td> <td>0.9104</td> 
<td>0.8564</td> <td>0.9139</td> <td>0.9448</td> <td>0.9060</td> <td>0.9042</td> 
<td>0.9281</td> <td>0.9066</td> <td>0.9398</td> <td>0.9158</td> <td>0.9228</td> 
<td>0.8522</td> <td>0.8377</td> <td>0.8724</td> <td>0.7988</td> <td>0.8417</td> 
<td>0.9303</td> <td>0.9523</td> <td>0.9750</td> <td>0.9635</td> <td>0.9550</td> 
<td>76.9552</td> <td>102.9565</td> <td>266.7600</td> <td>104.9370</td> <td>112.7901</td> 
<td>0.8543</td> <td>0.8741</td> <td>0.9072</td> <td>0.8491</td> <td>0.8718</td> 
<td>0.7835</td> <td>0.8032</td> <td>0.8643</td> <td>0.7885</td> <td>0.8106</td> 
</tr>
<tr>
<td>mobilenetv2_100</td>
<td>0.8931</td> <td>0.9047</td> <td>0.8865</td> <td>0.8769</td> <td>0.8909</td> 
<td>0.8998</td> <td>0.8901</td> <td>0.8914</td> <td>0.8663</td> <td>0.8866</td> 
<td>0.7638</td> <td>0.8241</td> <td>0.8375</td> <td>0.7888</td> <td>0.8025</td> 
<td>0.9533</td> <td>0.9429</td> <td>0.9043</td> <td>0.9155</td> <td>0.9293</td> 
<td>0.8837</td> <td>0.8723</td> <td>0.7614</td> <td>0.8038</td> <td>0.8314</td> 
<td>0.8967</td> <td>0.9188</td> <td>0.9385</td> <td>0.9081</td> <td>0.9154</td> 
<td>65.9745</td> <td>77.3143</td> <td>48.7161</td> <td>40.4858</td> <td>53.3796</td> 
<td>0.8194</td> <td>0.8475</td> <td>0.7976</td> <td>0.7962</td> <td>0.8167</td> 
<td>0.7441</td> <td>0.7783</td> <td>0.7190</td> <td>0.7081</td> <td>0.7390</td> 
</tr>
<tr>
<td>inception_resnet_v2</td>
<td>0.8546</td> <td>0.8300</td> <td>0.8405</td> <td>0.8354</td> <td>0.8403</td> 
<td>0.8765</td> <td>0.8507</td> <td>0.8609</td> <td>0.8479</td> <td>0.8533</td> 
<td>0.5870</td> <td>0.5340</td> <td>0.5410</td> <td>0.5906</td> <td>0.5624</td> 
<td>0.9696</td> <td>0.9663</td> <td>0.9714</td> <td>0.9450</td> <td>0.9635</td> 
<td>0.8926</td> <td>0.8793</td> <td>0.8919</td> <td>0.8279</td> <td>0.8723</td> 
<td>0.8452</td> <td>0.8184</td> <td>0.8289</td> <td>0.8376</td> <td>0.8324</td> 
<td>45.3644</td> <td>32.8266</td> <td>39.9732</td> <td>24.8034</td> <td>33.9310</td> 
<td>0.7082</td> <td>0.6645</td> <td>0.6735</td> <td>0.6894</td> <td>0.6839</td> 
<td>0.6168</td> <td>0.5596</td> <td>0.5762</td> <td>0.5817</td> <td>0.5836</td> 
</tr>
<tr>
<td>nasnetalarge</td>
<td>0.7883</td> <td>0.8196</td> <td>0.8281</td> <td>0.7931</td> <td>0.8074</td> 
<td>0.7957</td> <td>0.8375</td> <td>0.8395</td> <td>0.8086</td> <td>0.8168</td> 
<td>0.4697</td> <td>0.5474</td> <td>0.5729</td> <td>0.4754</td> <td>0.5164</td> 
<td>0.9399</td> <td>0.9367</td> <td>0.9557</td> <td>0.9511</td> <td>0.9453</td> 
<td>0.7881</td> <td>0.7879</td> <td>0.8661</td> <td>0.8286</td> <td>0.8174</td> 
<td>0.7883</td> <td>0.8280</td> <td>0.8174</td> <td>0.7848</td> <td>0.8049</td> 
<td>13.8526</td> <td>17.8804</td> <td>28.9598</td> <td>17.6215</td> <td>18.4672</td> 
<td>0.5886</td> <td>0.6460</td> <td>0.6897</td> <td>0.6042</td> <td>0.6329</td> 
<td>0.4581</td> <td>0.5302</td> <td>0.5775</td> <td>0.4777</td> <td>0.5113</td> 
</tr>
<tr>
<td>tf_efficientnet_b7</td>
<td>0.8106</td> <td>0.8590</td> <td>0.8621</td> <td>0.8183</td> <td>0.8378</td> 
<td>0.8133</td> <td>0.8347</td> <td>0.8749</td> <td>0.8247</td> <td>0.8332</td> 
<td>0.5851</td> <td>0.7135</td> <td>0.7143</td> <td>0.5812</td> <td>0.6481</td> 
<td>0.9130</td> <td>0.9226</td> <td>0.9309</td> <td>0.9388</td> <td>0.9259</td> 
<td>0.7534</td> <td>0.8012</td> <td>0.8280</td> <td>0.8284</td> <td>0.8026</td> 
<td>0.8289</td> <td>0.8804</td> <td>0.8750</td> <td>0.8152</td> <td>0.8499</td> 
<td>14.8077</td> <td>29.6711</td> <td>33.7037</td> <td>21.2951</td> <td>23.0189</td> 
<td>0.6587</td> <td>0.7548</td> <td>0.7670</td> <td>0.6831</td> <td>0.7171</td> 
<td>0.5305</td> <td>0.6563</td> <td>0.6698</td> <td>0.5612</td> <td>0.6052</td> 
</tr>
</tbody>
</table>
</div>

独立训练（Direct Train）在测试集上的结果：
<div class="table">
<table border="1" cellspacing="0" cellpadding="10" width="100%">
<thead>
<tr class="firstHead">
    <th colspan="1" rowspan="2">Model</th> <th colspan="5">accuracy</th> <th colspan="5">patient level accuracy</th> <th colspan="5">se</th> <th colspan="5">sp</th> <th colspan="5">ppv</th> <th colspan="5">npv</th> <th colspan="5">dor</th> <th colspan="5">f1</th> <th colspan="5">kappa</th> 
</tr>
<tr class="twoHead">
    <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th><th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th>
</tr>
</thead>
<tbody>
<tr>
<td>densenet201</td>
<td>0.8545</td> <td>0.8458</td> <td>0.8677</td> <td>0.8399</td> <td>0.8508</td> 
<td>0.8589</td> <td>0.8527</td> <td>0.8525</td> <td>0.8458</td> <td>0.8474</td> 
<td>0.7602</td> <td>0.7900</td> <td>0.7754</td> <td>0.6089</td> <td>0.8084</td> 
<td>0.9013</td> <td>0.8718</td> <td>0.9098</td> <td>0.9496</td> <td>0.8709</td> 
<td>0.7926</td> <td>0.7418</td> <td>0.7967</td> <td>0.8516</td> <td>0.7476</td> 
<td>0.8834</td> <td>0.8990</td> <td>0.8988</td> <td>0.8364</td> <td>0.9057</td> 
<td>28.9384</td> <td>25.5810</td> <td>34.8037</td> <td>29.3398</td> <td>28.4592</td> 
<td>0.7760</td> <td>0.7651</td> <td>0.7859</td> <td>0.7101</td> <td>0.7768</td> 
<td>0.6683</td> <td>0.6505</td> <td>0.6902</td> <td>0.6037</td> <td>0.6650</td> 
</tr>
<tr>
<td>xception</td>
<td>0.7525</td> <td>0.8387</td> <td>0.8154</td> <td>0.8228</td> <td>0.8251</td> 
<td>0.7844</td> <td>0.8372</td> <td>0.7999</td> <td>0.8090</td> <td>0.8273</td> 
<td>0.3333</td> <td>0.7062</td> <td>0.6277</td> <td>0.5756</td> <td>0.6680</td> 
<td>0.9481</td> <td>0.8916</td> <td>0.8966</td> <td>0.9396</td> <td>0.8944</td> 
<td>0.7500</td> <td>0.7225</td> <td>0.7239</td> <td>0.8182</td> <td>0.7360</td> 
<td>0.7529</td> <td>0.8837</td> <td>0.8478</td> <td>0.8241</td> <td>0.8594</td> 
<td>9.1429</td> <td>19.7817</td> <td>14.6095</td> <td>21.0822</td> <td>17.0364</td> 
<td>0.4615</td> <td>0.7143</td> <td>0.6724</td> <td>0.6758</td> <td>0.7004</td> 
<td>0.3304</td> <td>0.6019</td> <td>0.5448</td> <td>0.5588</td> <td>0.5773</td> 
</tr>
<tr>
<td>resnet152</td>
<td>0.8647</td> <td>0.6847</td> <td>0.9067</td> <td>0.6581</td> <td>0.8774</td> 
<td>0.8593</td> <td>0.7037</td> <td>0.8948</td> <td>0.7037</td> <td>0.8777</td> 
<td>0.8587</td> <td>0.0000</td> <td>0.8421</td> <td>0.0000</td> <td>0.7291</td> 
<td>0.8673</td> <td>1.0000</td> <td>0.9359</td> <td>1.0000</td> <td>0.9463</td> 
<td>0.7383</td> <td>0.0000</td> <td>0.8556</td> <td>0.0000</td> <td>0.8632</td> 
<td>0.9337</td> <td>0.6847</td> <td>0.9292</td> <td>0.6581</td> <td>0.8826</td> 
<td>39.7170</td> <td>0.0000</td> <td>77.8272</td> <td>0.0000</td> <td>47.4204</td> 
<td>0.7940</td> <td>0.0000</td> <td>0.8488</td> <td>0.0000</td> <td>0.7905</td> 
<td>0.6941</td> <td>0.0000</td> <td>0.7814</td> <td>0.0000</td> <td>0.7047</td> 
</tr>
<tr>
<td>mobilenetv2_100</td>
<td>0.8124</td> <td>0.8393</td> <td>0.8520</td> <td>0.8530</td> <td>0.8437</td> 
<td>0.7874</td> <td>0.8163</td> <td>0.8729</td> <td>0.8583</td> <td>0.8319</td> 
<td>0.5838</td> <td>0.6580</td> <td>0.6907</td> <td>0.6647</td> <td>0.6562</td> 
<td>0.9057</td> <td>0.9233</td> <td>0.9275</td> <td>0.9377</td> <td>0.9274</td> 
<td>0.7163</td> <td>0.7987</td> <td>0.8171</td> <td>0.8273</td> <td>0.8017</td> 
<td>0.8421</td> <td>0.8537</td> <td>0.8649</td> <td>0.8616</td> <td>0.8579</td> 
<td>13.4667</td> <td>23.1510</td> <td>28.5867</td> <td>29.8240</td> <td>24.3965</td> 
<td>0.6433</td> <td>0.7216</td> <td>0.7486</td> <td>0.7372</td> <td>0.7217</td> 
<td>0.5178</td> <td>0.6102</td> <td>0.6447</td> <td>0.6369</td> <td>0.6145</td> 
</tr>
<tr>
<td>inception_resnet_v2</td>
<td>0.7612</td> <td>0.8143</td> <td>0.7697</td> <td>0.8120</td> <td>0.7897</td> 
<td>0.7798</td> <td>0.8213</td> <td>0.7861</td> <td>0.8102</td> <td>0.8049</td> 
<td>0.3655</td> <td>0.5922</td> <td>0.4631</td> <td>0.6836</td> <td>0.5344</td> 
<td>0.9532</td> <td>0.9024</td> <td>0.9235</td> <td>0.8761</td> <td>0.9091</td> 
<td>0.7912</td> <td>0.7067</td> <td>0.7520</td> <td>0.7333</td> <td>0.7332</td> 
<td>0.7559</td> <td>0.8479</td> <td>0.7743</td> <td>0.8474</td> <td>0.8068</td> 
<td>11.7322</td> <td>13.4315</td> <td>10.4043</td> <td>15.2723</td> <td>11.4773</td> 
<td>0.5000</td> <td>0.6444</td> <td>0.5732</td> <td>0.7076</td> <td>0.6182</td> 
<td>0.3699</td> <td>0.5200</td> <td>0.4275</td> <td>0.5693</td> <td>0.4780</td> 
</tr>
<tr>
<td>nasnetalarge</td>
<td>0.7166</td> <td>0.7073</td> <td>0.7778</td> <td>0.7477</td> <td>0.7400</td> 
<td>0.7196</td> <td>0.7257</td> <td>0.7981</td> <td>0.7555</td> <td>0.7410</td> 
<td>0.4899</td> <td>0.5526</td> <td>0.4375</td> <td>0.4262</td> <td>0.6029</td> 
<td>0.8245</td> <td>0.7738</td> <td>0.9479</td> <td>0.9076</td> <td>0.8050</td> 
<td>0.5706</td> <td>0.5122</td> <td>0.8077</td> <td>0.6964</td> <td>0.5943</td> 
<td>0.7725</td> <td>0.8009</td> <td>0.7712</td> <td>0.7608</td> <td>0.8105</td> 
<td>4.5125</td> <td>4.2247</td> <td>14.1556</td> <td>7.2975</td> <td>6.2660</td> 
<td>0.5272</td> <td>0.5316</td> <td>0.5676</td> <td>0.5288</td> <td>0.5986</td> 
<td>0.3265</td> <td>0.3192</td> <td>0.4353</td> <td>0.3699</td> <td>0.4063</td> 
</tr>
<tr>
<td>tf_efficientnet_b7</td>
<td>0.7940</td> <td>0.8017</td> <td>0.8210</td> <td>0.7739</td> <td>0.8024</td> 
<td>0.7894</td> <td>0.8052</td> <td>0.7998</td> <td>0.7972</td> <td>0.8009</td> 
<td>0.5838</td> <td>0.6402</td> <td>0.6875</td> <td>0.5057</td> <td>0.6061</td> 
<td>0.8884</td> <td>0.8768</td> <td>0.8808</td> <td>0.9104</td> <td>0.8938</td> 
<td>0.7012</td> <td>0.7076</td> <td>0.7213</td> <td>0.7417</td> <td>0.7266</td> 
<td>0.8263</td> <td>0.8396</td> <td>0.8627</td> <td>0.7836</td> <td>0.8297</td> 
<td>11.1623</td> <td>12.6694</td> <td>16.2627</td> <td>10.3949</td> <td>12.9449</td> 
<td>0.6371</td> <td>0.6722</td> <td>0.7040</td> <td>0.6014</td> <td>0.6609</td> 
<td>0.4950</td> <td>0.5306</td> <td>0.5758</td> <td>0.4514</td> <td>0.5230</td> 
</tr>
</tbody>
</table>
</div>

联邦训练（Federated Train）在测试集上的结果：
<div class="table">
<table border="1" cellspacing="0" cellpadding="10" width="100%">
<thead>
<tr class="firstHead">
    <th colspan="1" rowspan="2">Model</th> <th colspan="5">accuracy</th> <th colspan="5">patient level accuracy</th> <th colspan="5">se</th> <th colspan="5">sp</th> <th colspan="5">ppv</th> <th colspan="5">npv</th> <th colspan="5">dor</th> <th colspan="5">f1</th> <th colspan="5">kappa</th> 
</tr>
<tr class="twoHead">
    <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th><th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th> <th>40x</th> <th>100x</th> <th>200x</th> <th>400x</th> <th>all</th>
</tr>
</thead>
<tbody>
<tr>
<td>densenet201</td>
<td>0.9019</td> <td>0.8967</td> <td>0.8978</td> <td>0.8687</td> <td>0.8917</td> 
<td>0.8998</td> <td>0.9048</td> <td>0.8936</td> <td>0.8722</td> <td>0.8961</td> 
<td>0.7551</td> <td>0.7350</td> <td>0.7861</td> <td>0.7486</td> <td>0.7559</td> 
<td>0.9747</td> <td>0.9720</td> <td>0.9488</td> <td>0.9257</td> <td>0.9559</td> 
<td>0.9367</td> <td>0.9245</td> <td>0.8750</td> <td>0.8272</td> <td>0.8903</td> 
<td>0.8891</td> <td>0.8872</td> <td>0.9068</td> <td>0.8858</td> <td>0.8922</td> 
<td>118.7083</td> <td>96.3821</td> <td>68.0750</td> <td>37.1159</td> <td>67.1695</td> 
<td>0.8362</td> <td>0.8189</td> <td>0.8282</td> <td>0.7859</td> <td>0.8176</td> 
<td>0.7673</td> <td>0.7480</td> <td>0.7558</td> <td>0.6916</td> <td>0.7413</td> 
</tr>
<tr>
<td>xception</td>
<td>0.7946</td> <td>0.8742</td> <td>0.8475</td> <td>0.8377</td> <td>0.8390</td> 
<td>0.8073</td> <td>0.8695</td> <td>0.8329</td> <td>0.8339</td> <td>0.8393</td> 
<td>0.4815</td> <td>0.6384</td> <td>0.6330</td> <td>0.6570</td> <td>0.6006</td> 
<td>0.9407</td> <td>0.9684</td> <td>0.9402</td> <td>0.9231</td> <td>0.9441</td> 
<td>0.7913</td> <td>0.8898</td> <td>0.8207</td> <td>0.8014</td> <td>0.8258</td> 
<td>0.7954</td> <td>0.8702</td> <td>0.8556</td> <td>0.8506</td> <td>0.8428</td> 
<td>14.7411</td> <td>54.1038</td> <td>27.1299</td> <td>22.9831</td> <td>25.4115</td> 
<td>0.5987</td> <td>0.7434</td> <td>0.7147</td> <td>0.7220</td> <td>0.6954</td> 
<td>0.4714</td> <td>0.6630</td> <td>0.6130</td> <td>0.6090</td> <td>0.5897</td> 
</tr>
<tr>
<td>resnet152</td>
<td>0.9076</td> <td>0.9097</td> <td>0.9133</td> <td>0.8848</td> <td>0.9043</td> 
<td>0.9013</td> <td>0.9174</td> <td>0.9004</td> <td>0.8859</td> <td>0.9067</td> 
<td>0.7609</td> <td>0.7969</td> <td>0.7947</td> <td>0.7807</td> <td>0.7835</td> 
<td>0.9716</td> <td>0.9616</td> <td>0.9667</td> <td>0.9389</td> <td>0.9605</td> 
<td>0.9211</td> <td>0.9053</td> <td>0.9152</td> <td>0.8690</td> <td>0.9021</td> 
<td>0.9031</td> <td>0.9114</td> <td>0.9126</td> <td>0.8918</td> <td>0.9052</td> 
<td>108.7121</td> <td>98.3221</td> <td>112.5586</td> <td>54.7095</td> <td>88.0023</td> 
<td>0.8333</td> <td>0.8476</td> <td>0.8507</td> <td>0.8225</td> <td>0.8387</td> 
<td>0.7702</td> <td>0.7838</td> <td>0.7900</td> <td>0.7376</td> <td>0.7712</td> 
</tr>
<tr>
<td>mobilenetv2_100</td>
<td>0.8626</td> <td>0.9000</td> <td>0.8717</td> <td>0.8566</td> <td>0.8732</td> 
<td>0.8525</td> <td>0.8925</td> <td>0.8919</td> <td>0.8583</td> <td>0.8765</td> 
<td>0.7110</td> <td>0.8135</td> <td>0.7216</td> <td>0.6763</td> <td>0.7326</td> 
<td>0.9245</td> <td>0.9400</td> <td>0.9420</td> <td>0.9377</td> <td>0.9360</td> 
<td>0.7935</td> <td>0.8626</td> <td>0.8537</td> <td>0.8298</td> <td>0.8364</td> 
<td>0.8869</td> <td>0.9159</td> <td>0.8784</td> <td>0.8657</td> <td>0.8868</td> 
<td>30.1350</td> <td>68.3822</td> <td>42.1296</td> <td>31.4263</td> <td>40.0532</td> 
<td>0.7500</td> <td>0.8373</td> <td>0.7821</td> <td>0.7452</td> <td>0.7811</td> 
<td>0.6557</td> <td>0.7652</td> <td>0.6921</td> <td>0.6469</td> <td>0.6923</td> 
</tr>
<tr>
<td>inception_resnet_v2</td>
<td>0.7828</td> <td>0.8460</td> <td>0.8207</td> <td>0.8477</td> <td>0.8239</td> 
<td>0.7946</td> <td>0.8522</td> <td>0.8592</td> <td>0.8452</td> <td>0.8366</td> 
<td>0.5330</td> <td>0.6313</td> <td>0.5567</td> <td>0.6497</td> <td>0.5899</td> 
<td>0.9039</td> <td>0.9313</td> <td>0.9531</td> <td>0.9465</td> <td>0.9332</td> 
<td>0.7292</td> <td>0.7847</td> <td>0.8561</td> <td>0.8582</td> <td>0.8051</td> 
<td>0.7996</td> <td>0.8642</td> <td>0.8109</td> <td>0.8442</td> <td>0.8296</td> 
<td>10.7400</td> <td>23.1965</td> <td>25.5076</td> <td>32.8014</td> <td>20.1020</td> 
<td>0.6158</td> <td>0.6997</td> <td>0.6746</td> <td>0.7395</td> <td>0.6809</td> 
<td>0.4694</td> <td>0.5978</td> <td>0.5584</td> <td>0.6349</td> <td>0.5632</td> 
</tr>
<tr>
<td>nasnetalarge</td>
<td>0.7752</td> <td>0.7975</td> <td>0.8299</td> <td>0.8240</td> <td>0.8057</td> 
<td>0.7705</td> <td>0.8158</td> <td>0.8290</td> <td>0.8298</td> <td>0.8118</td> 
<td>0.5101</td> <td>0.5684</td> <td>0.6354</td> <td>0.5956</td> <td>0.5767</td> 
<td>0.9014</td> <td>0.8959</td> <td>0.9271</td> <td>0.9375</td> <td>0.9143</td> 
<td>0.7113</td> <td>0.7013</td> <td>0.8133</td> <td>0.8258</td> <td>0.7612</td> 
<td>0.7945</td> <td>0.8285</td> <td>0.8357</td> <td>0.8234</td> <td>0.8201</td> 
<td>9.5235</td> <td>11.3383</td> <td>22.1592</td> <td>22.0946</td> <td>14.5304</td> 
<td>0.5941</td> <td>0.6279</td> <td>0.7135</td> <td>0.6921</td> <td>0.6562</td> 
<td>0.4445</td> <td>0.4909</td> <td>0.5950</td> <td>0.5733</td> <td>0.5244</td> 
</tr>
<tr>
<td>tf_efficientnet_b7</td>
<td>0.8428</td> <td>0.8370</td> <td>0.8710</td> <td>0.7989</td> <td>0.8390</td> 
<td>0.8359</td> <td>0.8428</td> <td>0.8612</td> <td>0.8145</td> <td>0.8372</td> 
<td>0.6701</td> <td>0.6561</td> <td>0.7031</td> <td>0.5398</td> <td>0.6446</td> 
<td>0.9203</td> <td>0.9212</td> <td>0.9463</td> <td>0.9306</td> <td>0.9296</td> 
<td>0.7904</td> <td>0.7949</td> <td>0.8544</td> <td>0.7983</td> <td>0.8100</td> 
<td>0.8614</td> <td>0.8519</td> <td>0.8766</td> <td>0.7990</td> <td>0.8488</td> 
<td>23.4409</td> <td>22.2962</td> <td>41.7048</td> <td>15.7356</td> <td>23.9405</td> 
<td>0.7253</td> <td>0.7188</td> <td>0.7714</td> <td>0.6441</td> <td>0.7179</td> 
<td>0.6162</td> <td>0.6055</td> <td>0.6827</td> <td>0.5111</td> <td>0.6073</td> 
</tr>
</tbody>
</table>
</div>
