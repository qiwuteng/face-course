# face-course
deeplearning class

## 存放数据集
在目录中创建 data 文件夹，将解压好的 FS2K 数据集放入 data 文件夹中。目录在 config.py 文件中，可以自行修改

```
face-course
├── data
│       ├── FS2K
│              ├── photo
│                      ├── photo1
│                      ├── photo2
│                      └── photo3
│              ├── sketch
│                      ├── sketch1
│                      ├── sketch2
│                      └── sketch3
│              ├── anno_test.json
│              ├── anno_train.json
│              └── README.pdf
├── FS2K.py
├── README.md
├── config.py
├── model.py
├── test.py
└── train.py

```


## 环境配置
pip install sklearn
pip install matplotlib
pip install pandas
conda install pytorch cudatoolkit torchvision -c pytorch

## 训练代码
修改 config.py 文件中的 model 选项，可选为 Alexnet， VGG16. VGG19, ResNet18， ResNet50。如果有需要可以修改其他选项，在终端输入 python train.py 就可以开始训练

## 测试代码
训练结束后，相应模型的权重就保存在 checkpoint 文件夹下，在终端输入 python test.py 即可开始训练。同样需要测试哪个模型，就修改 config.py 中的 model 选项
