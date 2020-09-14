from functools import reduce
class Perceptron(object):
    def __init__(self,input_num,activator):#初始化感知器，设置输入参数个数和激活函数
        self.activator=activator
        self.weights=[0.0 for _ in range(input_num)]#将权重向量和偏置项初始化为0
        self.bias=0.0
    def __str__(self):#打印学习到的权重和偏置项
        return 'weights\t:%s\nbias\t:%f\n'%(self.weights,self.bias)
    def predict(self,input_vec):#输入向量，输出感知器的计算结果
        return self.activator(reduce(lambda a,b:a+b,[x*w for x,w in zip(input_vec,self.weights)],0.0)+self.bias)
    def train(self,input_vecs,labels,iteration,rate):#输入训练数据、对应的标签、训练轮数、学习率
        for i in range(iteration):
            self.__one__iteration(input_vecs,labels,rate)
    def __one__iteration(self,input_vecs,labels,rate):#一次迭代，把所有的训练数据过一遍
        samples=zip(input_vecs,labels)#把输入和输出打包在一起，成为样本的列表[(input_vec, label), ...]
    #对每个样本，按照感知器规则更新权重
        for (input_vec,label) in samples:
            output=self.predict(input_vec)#计算感知器在当前权重下的输出
            self._update_weights(input_vec,output,label,rate)#更新权重
    def _update_weights(self,input_vec,output,label,rate):#按照感知器规则更新权重
        delta=label-output
        self.weights=[w+rate*delta*x for x,w in zip(input_vec,self.weights)]#感知器训练规则
        self.bias+=rate*delta#更新bias
#利用感知器去实现and函数
def f(x):#定义激活函数f
    return 1 if x>0 else 0
def get_training_dataset():#基于and真值构建训练数据
    input_vecs=[[1,1],[0,0],[1,0],[0,1]]
    labels=[1,0,0,0]
    return input_vecs,labels
def train_and_perceptron():
    p=Perceptron(2,f)#创建感知器，输入参数个数为2（因为and是二元函数），激活函数为f
    input_vecs,labels=get_training_dataset()
    p.train(input_vecs,labels,10,0.1)#迭代10轮，学习速率0.1
    return p
if __name__=='__main__':#作为程序的入口，“__name__”是Python的内置变量，用于指代当前模块，提高代码的规范性。
    and_perceptron=train_and_perceptron()#训练and感知器
    print(and_perceptron)#打印训练获得的权重
    print("1 and 1 ={:>3}".format(and_perceptron.predict([1,1])))#测试
    print("0 and 0 ={:>3}".format(and_perceptron.predict([0,0])))
    print("1 and 0 ={:>3}".format(and_perceptron.predict([1,0])))
    print("0 and 1 ={:>3}".format(and_perceptron.predict([0,1])))