import os
import numpy as np

from PIL import Image
from torchvision import transforms
from Gcutils import GradCAM, show_cam_on_image, center_crop_img
import torch
from matplotlib import pyplot as plt
from torch import nn
from torchvision.transforms import transforms

from model import MssNet_base
import os


from torchstat import stat

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def main():

    model = MssNet_base(num_classes=2)
    net = model

    device = torch.device("cpu")
    net.load_state_dict(
        torch.load(r"best_model.pth",
                   map_location=device))


    target_layers = [net.stages[-1]]
    print(target_layers)
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, 0.5, 0.5], std=[.5, 0.5, 0.5])])

    img_path = r"MCIc_052_S_0952_slice_Z41.jpg"  # 这里是导入你需要测试图片
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path).convert('RGB')
    img = img.resize((224, 224),Image.ANTIALIAS)

    img_tensor = data_transform(img)
    input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 增加一个batch维度
    cam = GradCAM(model=net, target_layers=target_layers, use_cuda=False)
    grayscale_cam = cam(input_tensor=input_tensor)

    grayscale_cam = grayscale_cam[0, :]
    img = np.array(img, dtype=np.float32)
    visualization = show_cam_on_image(img / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    print(visualization.shape)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig(r'save.jpg',bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
