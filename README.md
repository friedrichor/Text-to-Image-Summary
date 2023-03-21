# Text-to-Image-Summary
Summary of personal code for the Text-to-Image task.


## **Demo**

[demo](https://github.com/friedrichor/Text-to-Image-Summary/tree/main/demo) 中是使用预训练好的模型直接进行 Text-to-Image 的代码。  

## **Fine-tuning**

Fine-tuning 代码主要使用 [PyTorch](https://pytorch.org/) 和 [Hugging Face](https://huggingface.co/docs) 中的各个库，思路主要来源于 [Hugging Face Diffusers](https://github.com/huggingface/diffusers)。  

### **Requirement**

```
transformers>=4.27.0
diffusers>=0.15.0
```

如果您的数据集中没有现成的图像描述，想要使用 Image Captioning 模型来自动生成图像描述，需要通过以下指令安装最新版(开发版)的 transformers(>=4.27.0)，用于调用 BLIP-2 模型。
```
pip install git+https://github.com/huggingface/transformers
```
您可以通过以下指令安装最新版(开发版)的 diffusers(>=0.15.0)。
```
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install .
```

### **text2image**

[fine-tune/text2image](https://github.com/friedrichor/Text-to-Image-Summary/tree/main/fine-tune/text2image) 是正常的 fine-tune 代码，您可以通过查看其中的 README 来了解如何对 Stable Diffusion 进行 fine-tune。


### **DreamBooth**

&emsp;&emsp;DreamBooth 是一种个性化 text2image 模型的方法，例如给出一个物体/人物等的几张（3~5张）图像就能够 fine-tune Stable Diffusion，使模型能够“学会”这个物体，从而在后续的生成时能够更加准确地生成这个物体/人物。
