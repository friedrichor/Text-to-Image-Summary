# Text-to-Image-Summary
Summary of personal code for the Text-to-Image task.


## **Demo**

[demo](https://github.com/friedrichor/Text-to-Image-Summary/tree/main/demo) 中是使用预训练好的模型直接进行 Text-to-Image 的代码。  

## **Fine-tuning**

Fine-tuning 代码主要使用 [PyTorch](https://pytorch.org/) 和 [Hugging Face](https://huggingface.co/docs)，思路主要来源于 [Hugging Face Diffusers](https://github.com/huggingface/diffusers)。  

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

&emsp;&emsp;[fine-tune/text2image](https://github.com/friedrichor/Text-to-Image-Summary/tree/main/fine-tune/text2image) 是正常的 fine-tune 代码，您可以通过查看其中的 README 来了解如何对 Stable Diffusion 进行 fine-tune。


### **DreamBooth**

&emsp;&emsp;DreamBooth 是一种个性化 text2image 模型的方法，例如给出一个物体/人物等的几张（3~5张）图像就能够 fine-tune Stable Diffusion，使模型能够“学会”这个物体，从而在后续的生成时能够更加准确地生成这个物体/人物。

## Blogs related to Stable Diffusion

- [From DALL·E to Stable Diffusion: how do text-to-image generation models work?](https://tryolabs.com/blog/2022/08/31/from-dalle-to-stable-diffusion)  
作者讲解了 Diffusion Model 如何生成图像，解释了 DALL·E 2、Stable Diffusion 和 Imagen 这三种模型之间的差异所在，并从长远分析这些模型可能帮助公司和个人完成哪些实际任务。
- [Training Stable Diffusion with Dreambooth using 🧨 Diffusers](https://huggingface.co/blog/dreambooth)  
介绍了如何使用 Hugging Face Diffusers 提供的 Dreambooth 训练代码来 fine-tune Stabel Diffusion。作者进行了大量的实验来分析 Dreambooth 中不同参数设置的效果。这篇文章介绍了作者发现的一些技巧，可以在使用 Dreambooth fine-tune Stabel Diffusion 时改善结果。其中作者也讲解了如何使模型能够学会比较难的人脸。
- [The guide to fine-tuning Stable Diffusion with your own images](https://tryolabs.com/blog/2022/10/25/the-guide-to-fine-tuning-stable-diffusion-with-your-own-images)  
介绍了如何使用自己的数据集进行 fine-tune，讲的比较详细。
- [How does negative prompt work?](https://stable-diffusion-art.com/how-negative-prompt-work/)  
- [How to use negative prompts?](https://stable-diffusion-art.com/how-to-use-negative-prompts/)
- [Beginner’s guide to inpainting (step-by-step examples)](https://stable-diffusion-art.com/inpainting_basics/)  
如何修复图像（如人的脸部不自然，手臂缺失等等）
- [How to use VAE to improve eyes and faces](https://stable-diffusion-art.com/how-to-use-vae/)
