import torch
from torchvision import models,transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageFilter,ImageChops

CUDA_ENDABLED = True

LAYER_ID = 28
NUM_ITERATIONS = 5
LR = 1e-7

NUM_DOWNSCAES = 20
BLEDN_ALPHA = 0.5


class DeepDream:
    def __init__(self, image):
        self.image = image
        self.model = models.vgg16(pretrained=True)

        if CUDA_ENDABLED:
            self.model = self.model.cuda()
        self.modules = list(self.model.features.modules())

        imageSize = 224
        self.transformMean = [0.485,0.456,0.406]
        self.transformStd = [0.229,0.224,0.225]
        self.transformNormalise =transforms.Normalize(
            mean=self.transformMean,
            std=self.transformStd
        )

        self.transfromPreprocess = transforms.Compose([
            transforms.Resize((imageSize,imageSize)),
            transforms.ToTensor(),
            self.transformNormalise])

        # self.tensorMean = torch.Tensor(self.transformMean)
        # if(CUDA_ENDABLED):
        #     self.tensorMean = self.transformMean.cuda()
        #
        # self.tensorStd = torch.Tensor(self.transformStd)
        # if CUDA_ENDABLED:
        #     self.tensorStd = self.tensorStd.cuda()


    def toImage(self,input):
        return input*self.tensorStd + self.tensorMean


class DeepDream(DeepDream):
    def deepDream(self,image,layer,iterations,lr):
        transformed = self.transfromPreprocess(image).unsqueeze(0)

        if CUDA_ENDABLED:
            transformed = transformed.cuda()

        input = torch.autograd.Variable(transformed,requires_grad=True)
        self.model.zero_grad()
        optimizer = optim.Adam([input.requires_grad()],lr=LR)
        for _ in range(iterations):
            optimizer.zero_grad()
            out = input
            for layerId in range(layer):
                out = self.modules[layerId + 1](out)
            loss = -out.norm()  # 让负的变小, 正的变大
            loss.backward()
            optimizer.step()
            # input.data = input.data + lr * input.grad.data
        input = input.data.squeeze()
        input.transpose_(0, 1)
        input.transpose_(1, 2)
        input = self.toImage(input)
        if CUDA_ENDABLED:
            input = input.cpu()
        input = np.clip(input, 0, 1)
        return Image.fromarray(np.uint8(input * 255))


class DeepDream(DeepDream):
    def deepDreamRecursive(self, image, layer, iterations, lr, num_downscales):
        if num_downscales > 0:
            # scale down the image
            image_small = image.filter(ImageFilter.GaussianBlur(2)) # 高斯模糊
            small_size = (int(image.size[0]/2), int(image.size[1]/2))
            if (small_size[0] == 0 or small_size[1] == 0):
                small_size = image.size
            image_small = image_small.resize(small_size, Image.ANTIALIAS)
            # run deepDreamRecursive on the scaled down image
            image_small = self.deepDreamRecursive(image_small, layer, iterations, lr, num_downscales-1)
            print('Num Downscales : {}'.format(num_downscales))
            print('====Small Image=====')
            plt.imshow(image_small)
            plt.show()
            # Scale up the result image to the original size
            image_large = image_small.resize(image.size, Image.ANTIALIAS)
            print('====Large Image=====')
            plt.imshow(image_large)
            plt.show()
            # Blend the two image
            image = ImageChops.blend(image, image_large, BLEDN_ALPHA)
            print('====Blend Image=====')
            plt.imshow(image)
            plt.show()
        img_result = self.deepDream(image, layer, iterations, lr)
        print(img_result.size)
        img_result = img_result.resize(image.size)
        print(img_result.size)
        return img_result
    def deepDreamProcess(self):
        return self.deepDreamRecursive(self.image, LAYER_ID, NUM_ITERATIONS, LR,NUM_DOWNSCAES)

IMAGE_PATH = 'face.jpg'
img = Image.open(IMAGE_PATH)
plt.imshow(img)
plt.title("Image loaded from " + IMAGE_PATH)

img_deep_dream = DeepDream(img).deepDreamProcess()
plt.imshow(img_deep_dream)
plt.title("Deep dream image")