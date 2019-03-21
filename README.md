# 垃圾邮件过滤

写在前面：中文邮件用Jieba分词，如果在spamFiltering里调用不了spamEmailBayes，是因为没有将两个py放在一个路径下。

结果显示：
![Image text](https://github.com/cksywang/spam/blob/master/bayesspam%E7%BB%93%E6%9E%9C.png)
![Image text](https://github.com/cksywang/spam/blob/master/bayesspam%E7%BB%93%E6%9E%9C2.png)
![Image text](https://github.com/cksywang/spam/blob/master/bayesspam%E7%BB%93%E6%9E%9C3.png)

1.训练过程

将文本切分后，需要分别计算单词在垃圾邮件和正常邮件中出现的概率。
一封邮件只有两种分类，一类是正常邮件，一类则是垃圾邮件。因为训练集中已经分类完毕，只需将读取数据集，对每封邮件去掉停用词和重复出现的词语，再对剩下所有的文本使用正则表达式过滤掉邮件中的非中文字符。分别保存正常邮件与垃圾邮件中出现的词有多少邮件出现该词，得到两个词典。比如，我们假定"愚蠢"这个词，在4000封垃圾邮件中，有200封包含这个词，那么它的出现频率就是5%；而在4000封正常邮件中，只有2封包含这个词，那么出现频率就是0.05%。

这里有四种情况需要讨论：

（1）垃圾邮件词典和正常邮件词典中同时出现该单词的情况
分别计算垃圾邮件中出现该单词的概率P(W|S)，在正常邮件中出现该单词的概率P(W|N)，并计算出联合概率
P(S│W)=(P(W|S))/(P(W│S)+P(W|N))

（2）垃圾邮件词典中出现该单词，而正常邮件中没有出现的情况
计算垃圾邮件中出现该单词的概率P(W|S)，并设正常邮件中出现该单词的概率P(W│N)=0.01，并计算出联合概率
P(S│W)=(P(W|S))/(P(W│S)+P(W|N))
如果在此处将设置概率为0，则会导致后面的连乘计算全部为0，这会使贝叶斯分类器失去意义。

（3）正常邮件词典中出现该单词，而垃圾邮件中没有出现的情况
计算正常邮件中出现该单词的概率P(W|N)，并设在垃圾邮件中出现该单词的概率P(W│S)=0.01，并计算出联合概率
P(S│W)=(P(W|S))/(P(W│S)+P(W|N))

（4）在正常邮件词典和垃圾词典邮件中都没有出现的情况
这个情况通常不在训练过程中考虑，此处所出现的单词往往由测试集文本所引入，但在训练集中没有出现。因为垃圾邮件用的往往都是某些固定的词语，如果你从来没见过某个词，它多半是一个正常的词。于是我们人为地设定在垃圾邮件中出现该单词的概率P(S│W)=0.4

2.测试过程

现在对于已经训练好的垃圾邮件过滤器来说，测试集中的邮件都是未知分类的。它需要通过提取新邮件的特征，来判断邮件的类型。而在未经统计分析先，我们将假定每一封新邮件是垃圾邮件的概率P(S)=0.5，同样的，新邮件是正常邮件的概率P(N)=0.5。
过滤器将新邮件的文本切分成若干个单词，此时我们需要训练集中的单词在邮件分类中的概率来帮助我们判断。根据贝叶斯公式可知，该邮件是垃圾邮件，而某单词在此垃圾邮件中的的概率为
P(S│W)=(P(W|S)P(S))/(P(W│S)P(S)+P(W|N)P(N))

但是，考虑到每个用户收到的垃圾邮件的实际比例，想要先验概率P(S)，P(N)准确，你必须按小时去进行统计它的概率。因为比例变化很大，取决于一天的时间总体先验概率作为预测因素是无用的。同样，因为贝叶斯分类器可以看成对于输入的电子邮件没有偏见。于是将上式简化一般的公式
P(S│W)=(P(W|S))/(P(W│S)+P(W|N))

可是分类器仅凭一个垃圾单词就给这个邮件做判断，未免会有一些武断。像是单词“狗”，它可能出现在垃圾邮件中，但这并不一定表示含有“狗”的邮件就一定是垃圾邮件。对于邮箱用户来说，如果算法错误的判断使得过滤器将正常邮件打成垃圾邮件，比过滤器将垃圾邮件放过的后果更让人头痛。
几乎所有的正常邮件都有许多在垃圾邮件中也会存在的词汇。而这些单词对于我们的分类器来说，最多也是噪音。如何合理地判断是否垃圾邮件——在这个邮件中出现足够多的垃圾单词。但是分类器需要尽可能多地查看单词，因为正常邮件最多可能包含两到三个具有高垃圾邮件概率的单词。如果我们只使用前5个单词，那么分类器始出现误报的可能性会大大增加。因为朴素贝叶斯假设样本的各个特征之间是相互独立的，所以
P(S│W)=P(s│w_1 )P(s│w_2 )⋯P(s│w_n )
P(N│S)=(1-P(s│w_2 ))(1-P(s│w_2 ))⋯(1-P(s│w_n ))

则联合概率为
P=P(S│W)/(P(S│W)+P(N│S))
当P大于阈值时，我们将其判断为垃圾邮件；反之，它为正常邮件。在该垃圾邮件过滤器中，我将0.9设定为分类器的阈值。
