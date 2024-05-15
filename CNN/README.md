# 手写数字分类

这个 lab 是利用卷积神经网络完成手写数字分类。

我的代码写了逐行的注释， 这里简要展示主体部分：

### CNN

这是我依照题目任务书， 利用 `Pytorch` 框架定义出的 `CNN`：

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 特征提取层
        self.features = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 5, stride = 1, padding = 0),
            # 首先是一个卷积层， 用来提取图像的特征
            # 输入是 28 * 28 * 1 的图像，即长宽都是28，单通道
            # 这里用 5 * 5 的 16 个卷积核， 所以长宽都会减少 4， 变成 24 * 24， 然后每个
            # 卷积核产生一个特征图，输出是 16 个 24 * 24 的特征图
            # 所以输出的张量是 24 * 24 * 16

            nn.ReLU(inplace = True),
            # 之后立马 ReLU 激活函数， 用来增加网络的非线性
            # inplace = True 表示直接在原张量上进行操作，节省内存
            # 不改变输出的张量大小， 仍然是 24 * 24 * 16

            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 然后 MaxPool2d 池化层， 用来减少特征图的大小， 这里用 2 * 2 的滑动窗口， 每次滑动 2 个像素
            # 所以经过这一步， 图像的大小变为 1/4， 即 24 * 24 -> 12 * 12
            # 所以输出是 12 * 12 * 16

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 5, stride = 1, padding = 0),  
            # 再来一个卷积层， 用来进一步提取特征
            # 输入是 12 * 12 * 16 的特征图， 由于 kernel 是 5 * 5 * 32
            # 所以输出是 8 * 8 * 32

            nn.ReLU(inplace = True),
            # 继续 ReLU

            nn.MaxPool2d(kernel_size = 2, stride = 2),
            # 继续 MaxPool2d， 这里是 2 * 2 的滑动窗口， 每次滑动 2 个像素
            # 所以 8 * 8 -> 4 * 4
            # 所以输出是 4 * 4 * 32

            nn.Conv2d(in_channels = 32, out_channels = 100, kernel_size = 4, stride = 1, padding = 0),
            # 这里是用卷积操作实现了一个 FC 层， 用来将特征图展平成一维向量
            # 输入是 4 * 4 * 32 的特征图， 由于 kernel 是 4 * 4 * 100
            # 所以输出是 1 * 1 * 100
        )

        # 接下来是分类层
        self.classifier = nn.Sequential(
            nn.ReLU(inplace = True),
            # ReLU 

            nn.Linear(100, 100),
            # FC, 输出 100 维向量

            nn.ReLU(inplace = True),
            # ReLU

            nn.Linear(100, 10),
            # FC, 输出 10 维向量
        )

    # forward 函数    
    def forward(self, x):
        x = self.features(x)
        # 将输入的图片经过特征提取层

        x = x.view(x.size(0), -1)
        # 将特征图展平成一维向量， 这一步很关键， 不写就出大 bug 了

        x = self.classifier(x)
        # 经过分类层
        return x

```

接着， 我写了一个类 `CNN_on_MNIST` 封装本次任务:

```python
# 封装一个类， 包含本次任务所需的成员和函数
class CNN_on_MNIST:
    def __init__(self):
        self.custom_epochs = 5  
        # 总共训练迭代的次数

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 如果有 GPU 就用 GPU， 否则用 CPU

        self.learning_rate = 0.0008  
        # 设定初始的学习率

        self.CNNcustom_model = CNN().to(self.device)
        # 实例化模型，如果有 GPU 可以用， 那就把模型迁移到 GPU 上

        self.training_loss_list = []
        self.testing_loss_list = []
        # 收集损失函数， 用于绘图

    # 训练函数
    def train(self,  train_set_loader, custom_model, device):
        loss_prototype = nn.CrossEntropyLoss()
        # 损失函数用的是 CrossEntropyLoss， 这个函数是默认加上 softmax

        myopt_Adam = optim.Adam(self.CNNcustom_model.parameters(), lr = self.learning_rate)
        # 用 Adam 作为优化器

        custom_model.train() 

        for epoch in range(self.custom_epochs):
            print("This is epoch ", epoch)
            # 一共 custom_epochs 轮
            for num, (samples, labels) in enumerate(train_set_loader):
                # 每轮循环先取出训练集的图片和标签
                
                output = custom_model(samples.reshape(-1, 1, 28, 28))
                # 把一组数据转化为一个张量，然后输入到模型中

                loss = loss_prototype(output, labels)
                # 计算损失值

                myopt_Adam.zero_grad()
                # 优化器内部参数梯度必须变为0

                loss.backward()
                # 这里是 back propagation

                myopt_Adam.step()
                # 更新参数

                if (num + 1) % 50 == 0:
                    print("Loss becomes {:.4f}".format(loss.item()))
                    # 隔 50 个样本打印一点信息

                if epoch == 0 and (num + 1) % 2 == 0:
                    self.training_loss_list.append(loss.item())
                # 便于绘图
            
            print("\n")


    # 测试函数
    def test(self, test_set_loader, custom_model, device):
        loss_prototype = nn.CrossEntropyLoss()
        custom_model.eval() 
        Accuracy = 0
        length = len(test_set_loader.dataset)
    
        i = 0
        # 在测试集上每个样本点进行测试
        for data, target in test_set_loader:
            input = data.reshape(-1, 1, 28, 28)
            # 把数据转化为张量

            output = self.CNNcustom_model(input)
            result = output.data.max(1, keepdim = True)[1]
            # 得到模型在测试数据上的结果

            loss_one = loss_prototype(output, target).item() 
            if (i + 1) % 10 == 0:
                self.testing_loss_list.append(loss_one)
            # 计算损失值并加进 list 里面, 便于绘图

            Accuracy += result.eq(target.data.view_as(result)).sum() 
            # 更新表示模型准确度的 Accuracy， 
            i += 1
    
        print('\n', 'Accuracy is {:.2f}%'.format(100. * Accuracy / length), "\n")

        return 100. * Accuracy / length
```


然后是我的数据读取与预处理部分:

```python
# 获取原始的 MNIST 训练集
    full_train_set = datasets.MNIST(
                        root = 'data', # 指定文件路径
                        download = True, # 如果还没有下载则从网络上进行下载
                        train = True, # 指定是训练集
                        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, ), std=(0.5, )) 
                        ]) # 将数据转换为张量，并进行归一化, 这一步会做任务书上的 step2
    )

    # 接下来每个类别选取 300 个样本点
    indices_per_class = [np.random.choice(np.where(full_train_set.targets == i)[0], 300, replace = False) for i in range(10)]
    # np.where 找到某一类别的 300 个样本， 然后将它们作为一个整体存入 list

    indices = np.concatenate(indices_per_class)
    # 将选出的所有样本点合并为一个整体

    subset_train_dataset = Subset(full_train_set, indices)
    # 之后我的训练集都用这个子集

    Batch_Size = 2  
    # 指定批量大小, 由于训练集很小， 这里我直接选成了 2， 这样效果比较好。


    # 加载训练集
    train_loader = Data.DataLoader(
        dataset = subset_train_dataset, # 就用这个子集
        batch_size = Batch_Size, 
        shuffle = True
    )  # 每次迭代前打乱数据集
    
    print("size of train set: ", len(train_loader.dataset))

    # 对于 test set, 直接用全部的
    test_dataset = datasets.MNIST(
        root = 'data', 
        train = False, 
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = (0.5, ), std = (0.5, ))
        ])
    )

    # 加载测试集
    test_loader = Data.DataLoader(
        test_dataset,
        batch_size = Batch_Size, 
        shuffle=True
    )

```

下面是程序主体：

```python
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("This is the ", i, "-th turn")

# 实例化一个 CNN_on_MNIST
My_CNN_on_MNIST = CNN_on_MNIST()

# 进行训练和测试
print("---------------Training Start----------")
My_CNN_on_MNIST.train(train_loader, My_CNN_on_MNIST.CNNcustom_model, My_CNN_on_MNIST.device)
print("---------------Traning End-------------")

print("Testing our model on the TRANING SET: ")
My_CNN_on_MNIST.test(train_loader, My_CNN_on_MNIST.CNNcustom_model, My_CNN_on_MNIST.device)

print("Testing our model on the TESTING SET: ")
accuracy = My_CNN_on_MNIST.test(test_loader, My_CNN_on_MNIST.CNNcustom_model, My_CNN_on_MNIST.device)

if i == 0:
    drawing_graph(My_CNN_on_MNIST.testing_loss_list, My_CNN_on_MNIST.training_loss_list)

accuracy_list.append(accuracy)
```

剩下还有一些输出组件， 这里就不多赘述了。


### 损失函数值曲线

测试集的损失函数值绘图：

![](https://notes.sjtu.edu.cn/uploads/upload_f2c922ad2fe621e0f658cbda698d5509.png)


训练集的损失函数值绘图：

![](https://notes.sjtu.edu.cn/uploads/upload_9e07053efddfaeace012f5bfea8ece09.png)

可以看出， 测试集的损失函数普遍较小， 说明模型预测效果较好。 另外， 训练集的损失函数是收敛的趋势， 说明模型训练过程较为成功。

### 测试准确度

我进行了 10 次独立重复试验， 测试效果如下：

```
In the  0 -th turn, we have Accuracy 95.69%
In the  1 -th turn, we have Accuracy 94.89%
In the  2 -th turn, we have Accuracy 97.04%
In the  3 -th turn, we have Accuracy 97.25%
In the  4 -th turn, we have Accuracy 96.74%
In the  5 -th turn, we have Accuracy 95.59%
In the  6 -th turn, we have Accuracy 96.84%
In the  7 -th turn, we have Accuracy 95.63%
In the  8 -th turn, we have Accuracy 95.40%
In the  9 -th turn, we have Accuracy 96.95%
And the average accuracy is 96.20%
```

可见， 模型预测效果较好， 且较为稳定。

## 总结

通过本次作业， 我真正实现了自己的神经网络， 不由得感叹神经网络的强大威力和乐趣。 