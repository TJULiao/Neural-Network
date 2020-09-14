#简单的3层神经网络案例，用于识别手写数字
import numpy
import scipy.special#用于使用激活函数Sigmoid
import matplotlib.pyplot#用于绘制数组
import pylab#解决matplotlib图片不出现
class neutralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate): #初始化神经网络
        self.inodes=inputnodes#为各层设置神经元数量
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)) #设置各层的链接权重
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate#设置学习率
        self.activation_fuction=lambda x:scipy.special.expit(x)#激活函数为Sigmoid函数
        pass
    def train(self,inputs_list,targets_list):#训练神经网络
        inputs=numpy.array(inputs_list,ndmin=2).T#转换输入为二维数组,指定最小维度为2
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)#计算隐藏层输入
        hidden_outputs=self.activation_fuction(hidden_inputs)#计算隐藏层输出
        final_inputs=numpy.dot(self.who,hidden_outputs)#计算输出层输入
        final_outputs=self.activation_fuction(final_inputs)#计算输出层输出
        output_errors=targets-final_outputs#计算节点误差
        hidden_errors=numpy.dot(self.who.T,output_errors)#反向传播误差
        self.who+=self.lr*numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))#更新隐藏层和输出层链接权重
        self.wih+=self.lr*numpy.dot((hidden_errors*hidden_outputs*(1.0-hidden_outputs)),numpy.transpose(inputs))#更新输入层和隐藏层链接权重
        pass
    def query(self,inputs_list):#查询神经网络，接受输入，返回输出
        inputs=numpy.array(inputs_list,ndmin=2).T#转换输入为二维数组,指定最小维度为2
        hidden_inputs=numpy.dot(self.wih,inputs)#计算隐藏层输入
        hidden_outputs=self.activation_fuction(hidden_inputs)#计算隐藏层输出
        final_inputs=numpy.dot(self.who,hidden_outputs)#计算输出层输入
        final_outputs=self.activation_fuction(final_inputs)#计算输出层输出
        return final_outputs
#设定学习率以及输入层、隐藏层、输出层的神经元数量
input_nodes=784#因为输入为28*28个数值
hidden_nodes=100#可根据实验进行调整
output_nodes=10#因为标签对应10个数字
learning_rate=0.3#设定学习率
#创建神经网络实例，将对象实例化
n=neutralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
#导入训练数据集
training_data_file=open('E:/PythonPath/素材/mnist_train.csv','r')
training_data_list=[]
for line in training_data_file:
    training_data_list.append(line)
training_data_file.close()
#训练神经网络
epochs=2#设置世代的值，过高会过度拟合
for e in range(epochs):
    for record in training_data_list:
        all_values=record.split(',')#源文件数据值间以,分隔
        inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01#将0-255范围的原始输入变成0.01-1.00
        targets = numpy.zeros(output_nodes)+0.01#构建目标矩阵
        targets[int(all_values[0])]=0.99
        n.train(inputs,targets)
        pass
#导入测试数据集
test_data_file=open('E:/PythonPath/素材/mnist_test.csv','r')
test_data_list=[]
for line in test_data_file:
    test_data_list.append(line)
test_data_file.close()
#演示测试一个案例
all_values=test_data_list[0].split(',')
print(all_values[0])#输出正确结果
image_array=numpy.asfarray(all_values[1:]).reshape((28,28))#将列表ls中的文本字符串转换为实数，并创建这些数字的数组，最后形成m*n的矩阵
matplotlib.pyplot.imshow(image_array,cmap='Greys',interpolation='None')#使用imshow()函数绘出image_array
pylab.show()
print(n.query((numpy.asfarray(all_values[1:])/255.0*0.99)+0.01))#查询神经网络，返回输出
#测试神经网络
score_card=[]#创建记分卡
for record in test_data_list:
    all_values=record.split(',')
    correct_label=int(all_values[0])#输出正确标签
    print(correct_label,"correct label")
    inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01#调整输入范围
    outputs=n.query(inputs)#查询神经网络
    label=numpy.argmax(outputs)#返回数组最大值的索引，即神经网络给出的标签
    print(label,"network's answer")
    if(label==correct_label):
        score_card.append(1)
    else:
        score_card.append(0)
        pass
    pass
print(score_card)
scorecard_array=numpy.asarray(score_card)
print("performance={:.2%}".format(scorecard_array.sum()/scorecard_array.size))