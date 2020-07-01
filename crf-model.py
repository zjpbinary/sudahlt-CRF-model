import numpy as np
from scipy.special import logsumexp
import math
#预处理文本类
class Text_preprocess:
    def __init__(self, file):
        self.sentences = []
        self.taglists = []
        self.tag = []
        self.tag2index = dict()
        self.index2tag = dict()
        self.preprocess(file)
    def preprocess(self, filename):
        with open(filename, 'r', encoding='utf-8') as f1:
            lines = [line.strip().split('\t') for line in f1.readlines()]
        sent = []
        taglist = []
        for line in lines:
            if line==['']:
                self.sentences.append(sent)
                self.taglists.append(taglist)
                sent = []
                taglist = []
            else:
                sent.append(line[1])
                taglist.append(line[3])
                if line[3] not in self.tag:
                    self.tag.append(line[3])
        for i, t in enumerate(self.tag):
            self.tag2index[t] = i
            self.index2tag[i] = t

class CRFmodel:
    #初始化特征集合，索引，权重矩阵
    def __init__(self, text: Text_preprocess):
        self.featureset = set()
        self.feature2index = dict()
        self.preoperator(text)
        self.weight = np.zeros([len(text.tag), len(self.featureset)])
        self.tag2index = text.tag2index
        self.index2tag = text.index2tag
        self.tagset = text.tag
    #建立特征集与特征索引
    def preoperator(self, text:Text_preprocess):
        for i in range(len(text.sentences)):
            for pos in range(len(text.sentences[i])):
                if pos == 0:
                    feature = self.exa_feature1(text.sentences[i], pos)
                    feature.append(self.exa_feature2('*'))
                else:
                    feature = self.exa_feature1(text.sentences[i], pos)
                    feature.append(self.exa_feature2(text.taglists[i][pos-1]))
                for f in feature:
                    self.featureset.add(f)
        for n, f in enumerate(self.featureset):
            self.feature2index[f] = n

    #特征提取函数
    def exa_feature2(self, pretag):
        return '01:' + pretag
    def exa_feature1(self, sent, pos):
        f = []
        word = sent[pos]
        if pos == len(sent)-1:
            nextword = '$$'
        else:
            nextword = sent[pos+1]
        if pos == 0:
            pre = '**'
        else:
            pre = sent[pos-1]
        f.append('02:'+word)
        f.append('03:'+pre)
        f.append('04:'+nextword)
        f.append('05:'+word+pre[-1])
        f.append('06:'+word+nextword[0])
        f.append('07:'+word[0])
        f.append('08:'+word[-1])
        for i in range(1, len(word)-1):
            f.append('09:'+word[i])
            f.append('10:'+word[0]+word[i])
            f.append('11:'+word[-1]+word[i])
        if len(word)==1:
            f.append('12:'+word+pre[-1]+nextword[0])
        for i in range(0, len(word)-2):
            if word[i]==word[i+1]:
                f.append('13:'+word[i]+'consecutive')
        if len(word)>=4:
            for k in range(4):
                f.append('14:'+word[:k+1])
                f.append('15:'+word[-k-1::])
        return f

    #计算得分
    def computer_score(self, feature, tag):
        score = 0
        for f in feature:
            if f in self.featureset:
                score += self.weight[self.tag2index[tag]][self.feature2index[f]]
        return score

    #利用动态规划进行预测,维特比算法
    def predict(self, sent):
        senlen = len(sent)
        taglen = len(self.tagset)
        path = []
        max_score = np.zeros((senlen, taglen))
        fea1 = self.exa_feature1(sent, 0)
        fea1.append(self.exa_feature2('*'))

        for k in range(taglen):
            max_score[0][k] = self.computer_score(fea1, self.tagset[k])

        for i in range(1, senlen):
            fea1 = self.exa_feature1(sent, i)
            subpath = []
            for k in range(taglen):
                scores = np.array([max_score[i-1][t]+self.computer_score(fea1, self.tagset[k])+self.computer_score([self.exa_feature2(self.tagset[t])], self.tagset[k]) for t in range(taglen)])
                max_score[i][k] = np.max(scores)
                subpath.append(np.argmax(scores))
            path.append(subpath)

        real_path = []
        real_path.insert(0, np.argmax(max_score[-1]))
        while len(path)>0:
            pos = real_path[0]
            sub = path.pop()
            real_path.insert(0, sub[pos])
        goal = [self.index2tag[i] for i in real_path]
        return goal

    # 定义forward
    def forward(self, sent):
        sentlen = len(sent)
        taglen = len(self.tagset)
        am = np.zeros((sentlen, taglen))

        fea = self.exa_feature1(sent, 0)
        fea.append(self.exa_feature2('*'))
        for t in range(taglen):
            am[0][t] = self.computer_score(fea, self.tagset[t])
        temp = [[self.computer_score([self.exa_feature2(preti)], curti) for preti in self.tagset] for curti in self.tagset]

        for i in range(1, sentlen):
            fea = self.exa_feature1(sent, i)
            scores = np.array([self.computer_score(fea, ti) for ti in self.tagset]).reshape(-1, 1)
            scores = temp + scores
            am[i] = logsumexp(scores+am[i-1], axis=1) #按行计算logsumexp

        return am
    #backward
    def backward(self, sent):
        taglen = len(self.tagset)
        senlen = len(sent)
        am = np.zeros((senlen, taglen))
        temp = [[self.computer_score([self.exa_feature2(preti)], curti) for curti in self.tagset] for preti in self.tagset]
        for i in reversed(range(senlen-1)):
            fea = self.exa_feature1(sent, i+1)
            scores = np.array([self.computer_score(fea, ti) for ti in self.tagset])
            scores = scores + temp
            am[i] = logsumexp(am[i+1]+scores, axis=1)

        return am
    #定义权重更新
    def updateweight(self, sent, target, etc=0.1):
        #输入为句子，目标预测，学习率
        senlen = len(sent)
        gradients = dict()

        for i in range(senlen):
            fea = self.exa_feature1(sent, i)
            if i == 0:
                fea.append(self.exa_feature2('*'))
            else:
                fea.append(self.exa_feature2(target[i-1]))
            for f in fea:
                if f in self.featureset:
                    gradients[(self.tag2index[target[i]],self.feature2index[f])] = gradients.get((self.tag2index[target[i]],self.feature2index[f]),0) + 1

        # 在这里计算前向与反向,z
        forma = self.forward(sent)
        backma = self.backward(sent)
        logZ = logsumexp(forma[-1])

        fea = self.exa_feature1(sent, 0)
        for ti in self.tagset:
            score = self.computer_score(fea, ti)
            fea2 = [self.exa_feature2('*')]
            sc = score + self.computer_score(fea2, ti)
            p = math.e ** (sc + backma[0][self.tag2index[ti]] - logZ)
            for f in fea:
                if f in self.featureset:
                    gradients[(self.tag2index[ti], self.feature2index[f])] = gradients.get((self.tag2index[ti], self.feature2index[f]), 0) - p
                    #self.weight[self.tag2index[ti]][self.feature2index[f]] -= p
            for f in fea2:
                if f in self.featureset:
                    #self.weight[self.tag2index[ti]][self.feature2index[f]] -= p
                    gradients[(self.tag2index[ti], self.feature2index[f])] = gradients.get((self.tag2index[ti], self.feature2index[f]), 0) - p
        for i in range(1, senlen):
            fea = self.exa_feature1(sent, i)
            for ti in self.tagset:
                score = self.computer_score(fea, ti)
                for preti in self.tagset:
                    fea2 = [self.exa_feature2(preti)]
                    sc = score + self.computer_score(fea2, ti)
                    p = math.e**(forma[i-1][self.tag2index[preti]]+sc+backma[i][self.tag2index[ti]]-logZ)
                    for f in fea:
                        if f in self.featureset:
                            #self.weight[self.tag2index[ti]][self.feature2index[f]] -= p
                            gradients[(self.tag2index[ti], self.feature2index[f])] = gradients.get((self.tag2index[ti], self.feature2index[f]), 0) - p
                    for f in fea2:
                        if f in self.featureset:
                            #self.weight[self.tag2index[ti]][self.feature2index[f]] -= p
                            gradients[(self.tag2index[ti], self.feature2index[f])] = gradients.get((self.tag2index[ti], self.feature2index[f]), 0) - p

        for k, v in gradients.items():
            self.weight[k[0]][k[1]] += v*etc

    def SGD(self, text: Text_preprocess, num):
        for iterator in range(num):
            for i in range(len(text.sentences)):
                self.updateweight(text.sentences[i], text.taglists[i])

            precision = self.evaluate(text)
            print('第%d次迭代的精度：'%iterator, precision)
    def evaluate(self, text: Text_preprocess):
        count = 0
        right = 0
        for i in range(len(text.sentences)):
            eti = self.predict(text.sentences[i])
            for j in range(len(eti)):
                count+=1
                if eti[j] == text.taglists[i][j]:
                    right+=1
        return right/count


if __name__ == '__main__':
    train = Text_preprocess('data/train.conll')
    dev = Text_preprocess('data/dev.conll')

    model = CRFmodel(train)
    model.SGD(train, 5)
    print('测试集上的精度为：', model.evaluate(dev))

