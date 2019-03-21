import jieba;
import os;
class spamEmailBayes:

    # 获得停用词表
    def getStopWords(self):
        # 初始化
        stopList=[]
        for line in open("../data/中文停用词表.txt"):
            # 更新停用词表
            stopList.append(line[:len(line)-1])
        return stopList;

    # 获得词典，对训练集用结巴分词，使用停用表进行过滤
    # 停用词过滤，是文本分析中一个预处理方法，它的功能是过滤分词结果中的噪声（例如：的、是、啊等）
    def get_word_list(self,content,wordsList,stopList):
        # 将结巴分词的结果放入res_list
        res_list = list(jieba.cut(content))
        for i in res_list:
            if i not in stopList and i.strip()!='' and i!=None:
                if i not in wordsList:
                    wordsList.append(i)
                    
    # 若列表中的词已在词典中，则计数加1；如果不在词典中，则添加进去
    def addToDict(self,wordsList,wordsDict):
        """
                  Args：		wordsList：单词列表
                                wordsDict：单词词典
                  """
        for item in wordsList:
            if item in wordsDict.keys():
                wordsDict[item]+=1
            else:
                wordsDict.setdefault(item,1)
                            
    def get_File_List(self,filePath):
        # 返回指定的文件夹包含的文件或文件夹的名字的列表
        filenames=os.listdir(filePath)
        # 返回一个包含所有文档中出现的词的列表
        return filenames


    # 通过计算每个文件中p(s|w)来得到对分类影响最大的15个词
    def getTestWords(self,testDict,spamDict,normDict,normFilelen,spamFilelen):
        """
            Args：		testDict：测试词典
                        spamDict：垃圾邮件词典
                        normDict：正常邮件词典
                        normFilelen：正常邮件文件个数
                        spamFilelen：垃圾邮件文档个数
            """
        # 初始化
        wordProbList={}
        for word,num  in testDict.items():
            # 单词在垃圾邮件词典中和正常邮件词典中同时出现的情况
            if word in spamDict.keys() and word in normDict.keys():
                # 该文件中包含词个数
                # 是垃圾邮件的概率
                pw_s=spamDict[word]/spamFilelen
                # 是正常邮件的概率
                pw_n=normDict[word]/normFilelen
                ps_w=pw_s/(pw_s+pw_n)
                # setdefault() 函数和get() 方法类似, 如果相应的文本word不存在于字典中，将会添加键并将值设为默认值（这里是是ps_w）
                wordProbList.setdefault(word,ps_w)

            # 单词在垃圾邮件词典中,不在正常邮件词典中的情况
            if word in spamDict.keys() and word not in normDict.keys():
                pw_s=spamDict[word]/spamFilelen
                # 设是正常邮件的概率为0.01，如果这里是0，会导致后面概率相乘为0
                pw_n=0.01
                ps_w=pw_s/(pw_s+pw_n) 
                wordProbList.setdefault(word,ps_w)

            # 单词不在垃圾邮件词典中,在正常邮件词典中的情况
            if word not in spamDict.keys() and word in normDict.keys():
                pw_s=0.01
                pw_n=normDict[word]/normFilelen
                ps_w=pw_s/(pw_s+pw_n) 
                wordProbList.setdefault(word,ps_w)

            # 单词不在垃圾邮件词典中,在正常邮件词典中的情况
            if word not in spamDict.keys() and word not in normDict.keys():
                # 假定p(s|w) = 0.4
                wordProbList.setdefault(word,0.4)
        #对wordProbList排序
        sorted(wordProbList.items(),key=lambda d:d[1],reverse=True)[0:15]
        return (wordProbList)
    
    # 计算贝叶斯概率
    def calBayes(self,wordList,spamdict,normdict):
        # 初始化概率
        ps_w=1
        ps_n=1
         
        for word,prob in wordList.items() :
            # 输出的是单词 + 拥有该单词的邮件是垃圾邮件的概率
            print(word+"/"+str(prob))
            ps_w*=(prob)
            ps_n*=(1-prob)
        # 联合概率表达式
        p=ps_w/(ps_w+ps_n)
#         print(str(ps_w)+"////"+str(ps_n))
        return p        

    # 计算预测结果正确率
    def calAccuracy(self,testResult):
        rightCount=0
        errorCount=0
        for name ,catagory in testResult.items():
            # 测试集中文件名低于1000的为正常邮件，大于1000的是垃圾邮件
            if (int(name)<1000 and catagory==0) or(int(name)>1000 and catagory==1):
                rightCount+=1
            else:
                errorCount+=1
        return rightCount/(rightCount+errorCount)
