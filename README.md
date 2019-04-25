# compare-aggregate
## 复现《A Compare-Aggregate Model for Matching Text Sequences》论文
[**原作代码**](https://github.com/shuohangwang/SeqMatchSeq/blob/master/wikiqa/compAggWikiqa.lua)
### 引言
Compare-Aggregate Model源于2017年的一篇paper，这个模型是一个强力的baseline，在sentence maching有着非凡的意义，找了好几个tensorflow版本的，结果复现不尽人意，这个repo记录实验的一些记录，如果有同学成功复现了论文的结果的，烦请告知，感谢！

### 环境配置
    
    Python版本为3.6  
    tensorflow版本为1.13  
  
### 结果总览
实验结果以表格的形式呈现  
pt --> pointwise  
lt --> listwise  
ps --> paperSet  
ms --> mySet  

|Model|set|Parameters|Epoch|DATE|Datatype|备注|MAP|MRR|  
|-|-|-|-|-|-|-|-|-|
|Demo1|ms||100|2019.4.23|pt|无|0.6413|0.6417|
|Demo2-1|ms1|2073302|100|2019.4.23|pt|验证集达到0.77/0.78|0.6977|0.7080|
|Demo2-2|ms1|2073302|100|2019.4.24|pt|验证集达到0.75/0.76|0.7014|0.7155|
|Demo2-3|ps|2073302|100|2019.4.24|pt|验证集达到0.75/0.76|0.7106|0.7204|
|Demo3|ms2|10308301|100|2019.4.24|lt|验证集达到0.72/0.73|0.7036|0.7210|
|Demo3|ps|10308301|100|2019.4.24|lt|验证集达到0.684/0.694|0.6864|0.7065|
|Demo4|ms3||200|2019.4.25|lt|验证集达到0.680/0.680|0.6060|0.6060|
|**Best**|||||||**0.7106**|**0.7204**|



### 实验
- **参数设置**  
**paperSet**  

|batchSize|qLen|aLen|lr|hiddenDim|listwise|dropout|备注|  
|-|-|-|-|-|-|-|-|
|**25**|25|100|**0.002**|**100**|15|0.5|句子长度，listwise，dropout自己设定|

**mySet1**  

|batchSize|qLen|aLen|lr|hiddenDim|listwise|dropout|备注|  
|-|-|-|-|-|-|-|-|
|5|25|100|0.001|300|/|0.5|/|

**mySet2**  

|batchSize|qLen|aLen|lr|hiddenDim|listwise|dropout|备注|  
|-|-|-|-|-|-|-|-|
|5|25|100|0.001|300|15|0.5|/|

**mySet3**  

|batchSize|qLen|aLen|lr|hiddenDim|listwise|dropout|备注|  
|-|-|-|-|-|-|-|-|
|64|25|100|0.001|300|15|0.5|/|

- **Demo1**  
reference: [seq_match_seq](https://github.com/WenRichard/Question_Answering_Models/tree/master/cQA/seq_match_seq)

- **Demo2**  
分为model和model2，区别在于sim matrix的不同实现,Demo2-1,Demo2-3是model的实现，Demo2-2是model2的实现  

- **Demo3**  
reference: [compareAggregate](https://github.com/UKPLab/aaai2019-coala-cqa-answer-selection/blob/c4fcf77373cd86a9a043c38ae9ab0dc1882a6b17/experiment/qa_pairwise/model/__init__.py)  
Demo3算是比较细节的一个模型了，模型的结构很清楚，我也加了学习率动态变化的效果，确实，加了之后，模型更加鲁棒了  

- **Demo4**  
reference: [PMGA](https://github.com/laox1ao/PMGA)  
该repo是将dynamic-clip-attention实现的keras版本改下成tensorflow版本的，基本上还原了原作的模型结构。  

