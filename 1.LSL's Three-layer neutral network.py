#简单的3层神经网络案例，用于识别手写数字
import numpy
import scipy.special#用于使用激活函数Sigmoid
import time
import scipy.ndimage
class neutralNetwork:
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate): #初始化神经网络
        self.inodes=inputnodes#为各层设置神经元数量
        self.hnodes=hiddennodes
        self.onodes=outputnodes
        self.wih=numpy.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes)) #设置各层的链接权重
        self.who=numpy.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        self.lr=learningrate#设置学习率
        self.activation_fuction=lambda x:scipy.special.expit(x)#激活函数为Sigmoid函数
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
        return final_outputs,output_errors
    def query(self,inputs_list,targets_list):#查询神经网络，接受输入，返回输出
        inputs=numpy.array(inputs_list,ndmin=2).T#转换输入为二维数组,指定最小维度为2
        targets=numpy.array(targets_list,ndmin=2).T
        hidden_inputs=numpy.dot(self.wih,inputs)#计算隐藏层输入
        hidden_outputs=self.activation_fuction(hidden_inputs)#计算隐藏层输出
        final_inputs=numpy.dot(self.who,hidden_outputs)#计算输出层输入
        final_outputs=self.activation_fuction(final_inputs)#计算输出层输出
        output_errors=targets-final_outputs#计算节点误差
        return final_outputs,output_errors
#设定学习率以及输入层、隐藏层、输出层的神经元数量
input_nodes=784#因为输入为28*28个数值
hidden_nodes=100#可根据实验进行调整
output_nodes=10#因为标签对应10个数字
print("输入层节点{}个，隐藏层节点{}个，输出层节点{}个。".format(input_nodes,hidden_nodes,output_nodes))
#导入训练数据集
training_data_file=open('E:/PythonPath/素材/mnist_train.csv','r')
training_data_list=[]
for line in training_data_file:
    training_data_list.append(line)
training_data_file.close()
#导入测试数据集
test_data_file=open('E:/PythonPath/素材/mnist_test.csv','r')
test_data_list=[]
for line in test_data_file:
    test_data_list.append(line)
test_data_file.close()
for learning_rate in numpy.arange(0.01,0.06,0.02):
    #创建神经网络实例，将对象实例化
    n=neutralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
    #训练神经网络
    epochs=20#设置世代的值，过高会过度拟合
    for e in range(epochs):
        start=time.perf_counter()
        train_card=[]
        trainerror=0.0
        for record in training_data_list:
            all_values=record.split(',')#源文件数据值间以,分隔
            correct_label=int(all_values[0])
            inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01#将0-255范围的原始输入变成0.01-1.00
            targets=numpy.zeros(output_nodes)+0.01#构建目标矩阵
            targets[int(all_values[0])]=0.99
            [outputs,train_error]=n.train(inputs,targets)
            trainerror+=numpy.sum(train_error**2)#计算节点误差
            label=numpy.argmax(outputs)
            train_card.append(1) if (label==correct_label) else train_card.append(0)
            #数据增强
            inputs_plus10_img=scipy.ndimage.interpolation.rotate(inputs.reshape(28,28),10,cval=0.01,order=1,reshape=False)
            [outputs,train_error]=n.train(inputs_plus10_img.reshape(784),targets)
            trainerror+=numpy.sum(train_error**2)#计算节点误差
            label=numpy.argmax(outputs)
            train_card.append(1) if (label==correct_label) else train_card.append(0)
            inputs_minus10_img=scipy.ndimage.interpolation.rotate(inputs.reshape(28,28),-10,cval=0.01,order=1,reshape=False)
            [outputs,train_error]=n.train(inputs_minus10_img.reshape(784),targets)
            trainerror+=numpy.sum(train_error**2)#计算节点误差
            label=numpy.argmax(outputs)
            train_card.append(1) if (label==correct_label) else train_card.append(0)
        trainerror=trainerror/len((training_data_list)*3)
        traincard_array=numpy.asarray(train_card)
        train_accuracy=traincard_array.sum()/traincard_array.size#计算正确率
        #测试神经网络
        test_card=[]#创建记分卡
        testerror=0.0
        for record in test_data_list:
            all_values=record.split(',')
            correct_label=int(all_values[0])#输出正确标签
            inputs=(numpy.asfarray(all_values[1:])/255.0*0.99)+0.01#调整输入范围
            targets=numpy.zeros(output_nodes)+0.01#构建目标矩阵
            targets[int(all_values[0])]=0.99
            [outputs,test_error]=n.query(inputs,targets)#查询神经网络
            testerror+=numpy.sum(test_error**2)
            label=numpy.argmax(outputs)#返回数组最大值的索引，即神经网络给出的标签
            test_card.append(1) if (label==correct_label) else test_card.append(0)
        testerror=testerror/len(test_data_list)
        testcard_array=numpy.asarray(test_card)
        dur=time.perf_counter()-start
        test_accuracy=testcard_array.sum()/testcard_array.size
        print("学习率={0:.2f}，世代={1:0>2}，训练正确率={2:.2%}，训练误差={3:.3f}，测试正确率={4:.2%}，测试误差={5:.3f}，时间={6:.2f}s"\
              .format(learning_rate,e+1,train_accuracy,trainerror,test_accuracy,testerror,dur))