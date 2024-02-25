import torch.optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import time
import os
from PIL import Image, ImageChops

# 加载gpu
device = torch.device("mps")

# 图片处理
normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
trans_alter = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop(224),
    torchvision.transforms.RandomRotation(90, expand=True),
    torchvision.transforms.Resize(224),
    torchvision.transforms.ToTensor(),
    normalize
])


# 构建数据集
class MyDataset(Dataset):

    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = str(os.path.join(self.root_dir, self.label_dir))
        self.img_path = os.listdir(self.path)

    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        img = ImageChops.invert(img)
        img = trans_alter(img)
        label = self.img_path[idx].split(".")[0]
        if label == "dog":
            label = torch.tensor(1)
        else:
            label = torch.tensor(0)
        return img, label

    def __len__(self):
        return len(self.img_path)


root_dir = "woc_imgs"
train_label_dir = "train"
train_data = MyDataset(root_dir, train_label_dir)
train_loader = DataLoader(dataset=train_data, batch_size=16, shuffle=True, num_workers=0, drop_last=True)

# length 长度
train_data_size = len(train_data)

writer = SummaryWriter("logs")

# 检查图片变换效果
# step = 0
#
# for i in train_data:
#     img, label = i
#     writer.add_image("train", img, step)
#     step += 1

writer.close()
# 创建网络模型+修改最后的输出
resnet50_model = torchvision.models.resnet50(weights='IMAGENET1K_V2')
resnet50_model.fc = torch.nn.Linear(2048, 2)
resnet50_model.add_module('softmax', nn.Softmax(dim=1))
resnet50_model.to(device)

# 构建损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01
optimzer = torch.optim.SGD(resnet50_model.parameters(), learning_rate)

# 设置训练网络参数
total_train_step = 0        # 训练次数
total_test_step = 0         # 测试次数
epoch = 10                  # 训练轮数
start_time = time.time()    # 开始时间

for i in range(epoch):
    print("第 {} 轮".format(i+1))

    total_test_loss = 0
    total_acc = 0
    for data in train_loader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = resnet50_model(imgs)
        outputs = nn.Softmax(dim=1)(outputs)
        Loss = loss_fn(outputs, targets)
        total_test_loss += Loss.item()

        # 优化器优化模型
        optimzer.zero_grad()
        Loss.backward()
        optimzer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print("训练次数：{}，Loss:{:.4}".format(total_train_step, Loss))
            print("训练花费总时间为：{:.4}s".format(end_time - start_time))
            writer.add_scalar("train_loss", Loss.item(), total_train_step)
        acc = (outputs.argmax(1) == targets).sum()
        total_acc += acc

    print("整体测试集上的Loss: {}".format(total_test_loss))
    print("整体测试集上的正确率: {}".format(total_acc / train_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_acc / train_data_size, total_test_step)
    total_test_step = total_test_step + 1

    torch.save(resnet50_model, "resnet50_model_new_{}.pth".format(i))
    print("模型已保存")

writer.close()
