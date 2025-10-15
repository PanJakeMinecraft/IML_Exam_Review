import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

tensor_data = transforms.Compose([transforms.ToTensor()])
train = datasets.FashionMNIST(
    root = "data",
    train = True,
    transform= tensor_data,
    download=False
)

fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(8,8))
plt.suptitle("Fashion MNIST Dataset")

for i, ax in enumerate(axes.flat):
    img, lbl = train[i]
    ax.imshow(img.squeeze(), cmap= "bone")
    ax.set_title(f"Label: {lbl}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.show()