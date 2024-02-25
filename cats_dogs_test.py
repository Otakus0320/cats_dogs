import torch
import csv
import os
import torchvision
from PIL import Image
from torch import nn

# GPU设置
device = torch.device("mps")

# 模型加载
test_model = torch.load("resnet50_model_new_9.pth")
test_model.eval()
test_model.to(device)

trans_alter = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor()
])

root_dir = "woc_imgs/test"

with open('output.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['ID', 'TARGET'])

    pic_id = 1
    for i in range(9375):
        m = str(i+1) + ".jpg"
        img_path = os.path.join(root_dir, m)
        data = trans_alter(Image.open(img_path))
        data = data.unsqueeze(0)
        data = data.to(device)
        prediction = test_model(data)
        prediction = torch.nn.functional.softmax(prediction, dim=1)
        output = prediction[0][1].item()
        row = [pic_id, output]
        writer.writerow(row)
        pic_id += 1
        if pic_id % 100 == 0:
            print(pic_id)

print("结束")
