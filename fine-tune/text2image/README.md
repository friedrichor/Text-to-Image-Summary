# Stable Diffusion text-to-image fine-tuning

# **Prepare the dataset**

1. 将训练集图像放到 `images` 文件夹下 (当前文件夹中有几张图片作为样例)
2. 准备图像描述，格式请参考 `image_captions.json`  
若没有现成的图像描述，可以使用 Image-to-Text 模型来生成 Image Caption，如 BLIP-2 (应该是目前最好的Image Captioning模型)、BLIP、ViT-GPT2 (目前 HuggingFace 下载量最高的 Image Captioning 模型，且显存占用较少)  
- BLIP-2: `generate_captions.py` 中给出了如何使用 BLIP-2 将 `images` 文件夹下的图片生成图像描述并保存到 `image_captions.json`  
&emsp;&emsp;如果要运行这个程序，请先在终端执行以下命令安装开发版的 transformers 库 (因为 BLIP-2 只在4.27版本后才有，目前稳定版的库还没更新到4.27)  
```
pip install git+https://github.com/huggingface/transformers
``` 
- BLIP: 可参考 https://huggingface.co/Salesforce/blip-image-captioning-base 或 https://huggingface.co/Salesforce/blip-image-captioning-large
- ViT-GPT2: 可参考 https://huggingface.co/nlpconnect/vit-gpt2-image-captioning
3. 运行 `create_dataset.py` 创建数据集  
- 默认情况下是将创建的数据集保存到本地磁盘 (直接运行这个代码即可)，然后从本地调用数据集，如果不太了解怎么将数据集上传到HuggingFace的推荐这种方式。  
- 同时也提供了将数据集上传到 Huggingface 的代码, 设置 `--save_to_disk=True` 等，这个代码参考于 https://github.com/YaYaB/finetune-diffusion，不过目前我还没能把数据集上传到 HuggingFace，后续可能对这部分代码进行改进。

**注意: images 所提供的图片并不足以 fine-tune Stable Diffusion，仅仅是这几张图片训练也可能使模型效果变差，这里放几张图片只是作为示例，您也可以用这几张图片来测试代码是否能跑。实际效果需以您的数据集为准。**

# **Fine-tuning Stable Diffusion**

以下运行示例均按从本地磁盘读取数据集的情况，如有问题可参考 [Hugging Face Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

## **Train with LORA**

Low-Rank Adaption of Large Language Models was first introduced by Microsoft in [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) by *Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen*.

In a nutshell, LoRA allows adapting pretrained models by adding pairs of rank-decomposition matrices to existing weights and **only** training those newly added weights. This has a couple of advantages:

- Previous pretrained weights are kept frozen so that model is not prone to [catastrophic forgetting](https://www.pnas.org/doi/10.1073/pnas.1611835114).
- Rank-decomposition matrices have significantly fewer parameters than original model, which means that trained LoRA weights are easily portable.
- LoRA attention layers allow to control to which extent the model is adapted toward new training images via a `scale` parameter.

[cloneofsimo](https://github.com/cloneofsimo) was the first to try out LoRA training for Stable Diffusion in the popular [lora](https://github.com/cloneofsimo/lora) GitHub repository.

With LoRA, it's possible to fine-tune Stable Diffusion on a custom image-caption pair dataset on consumer GPUs like Tesla T4, Tesla V100.

&emsp;&emsp;关于 LORA 的其他介绍，可以参考 HuggingFace 文档 https://huggingface.co/docs/diffusers/v0.14.0/en/training/lora  
&emsp;&emsp;使用 LORA 训练时显存占用大大减小 (正常的 fine-tune 运行时大概需要 26G显存左右，使用 LORA 后大概只需要 8G 显存)，并且最终保存的模型也很小 (只有几M，比原先的模型大小小了几个数量级)，并且训练速度更快。


### **Running**

具体参数设置示例可参考 `train_text_to_image_lora.sh`, 在终端输入以下命令即可运行训练代码: 
```
sh train_text_to_image_lora.sh
``` 
如果您需要修改或添加参数等直接在 `train_text_to_image_lora.sh` 中修改即可。

**注意: 如果您使用的是 Stable Diffusion-2 768x768 模型，则将 `resolution` 更改为768**

1. 由于 fine-tune 过程中并不是训练的 epoch 或 step 越多，模型效果越好，模型会在中途某一阶段效果显著下降，因此在训练过程中每 `validation_epochs` 个 epoch 就会用训练过程中的模型来进行 text-to-image generation，每个 prompt 生成 `num_validation_images` 张图像，以检查训练过程中模型的效果，以便于您保存最佳的模型。您可以根据自己需要更改 `validation_prompts.txt` 中的内容，每行为一个 prompt。
2. `use_8bit_adam` 表示是否使用 8-bit Adam，使用 8-bit Adam 会显著降低模型训练时所需的显存大小，并且模型效果基本不会受到影响。  
paper: [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/abs/2110.02861#)  

<center><img src="https://github.com/friedrichor/Text-to-Image-Summary/blob/main/photos_for_readme/8-bit_Adam.png" width="50%"></center>  

3. `report_to` 默认设置为 "tensorboard"，您可以通过在终端输入  
```
tensorboard --logdir=text2image/txt2img-finetune-lora/logs
```
来查看训练过程中生成的图像，其中 logdir=后面接的是您将 logs 保存的路径。   
4. 如果您需要更改 [accelerate](https://huggingface.co/docs/accelerate/v0.16.0/en/index) 相关的设置，如更改程序运行所在的 GPU (默认device为"cuda:0")，可以在终端输入以下指令来调整:
```
accelerate config
```

### **Inference**

&emsp;&emsp;您可以通过 `inference.ipynb` 来进行 inference，训练好的模型默认保存在 `txt2img-finetune-lora/pytorch_lora_weights.bin`。  
&emsp;&emsp;当然您也可以从某个 checkpoint 进行 inference，只需更改 `model_path = "txt2img-finetune-lora/checkpoint-500/pytorch_model.bin"` 即可。   
&emsp;&emsp;生成的图片保存在 `results` 文件夹下。   


# Reference
1. [Hugging Face Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)
2. [YaYaB/finetune-diffusion](https://github.com/YaYaB/finetune-diffusion)
