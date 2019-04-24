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

|Model|set|Parameters|Epoch|MAP|MRR|DATE|Datatype|备注|  
|-|-|-|-|-|-|-|-|-|
|Demo1|mySet|/|100|0.6413|0.6417|2019.4.23|pt|无|
|Demo2-1|mySet|2073302|100|0.6977|0.7080|2019.4.23|pt|验证集达到0.77/0.78|
|Demo2-1|mySet|2073302|100|0.6948|0.7063|2019.4.23|pt|验证集达到0.75/0.77|
|Demo2-2|mySet|2073302|100|0.7014|0.7155|2019.4.23|pt|验证集达到0.75/0.76|



### 实验
**参数设置**  
paperSet  

|batchSize|qLen|aLen|lr|hiddenDim|listwise|dropout|备注|  
|-|-|-|-|-|-|-|-|
|25|25|100|0.002|100|15|0.5|句子长度，listwise，dropout自己设定|

- Demo1  


