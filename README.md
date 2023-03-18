# Text-to-Image-Summary
Summary of personal code for the Text-to-Image task.

demo 中是使用预训练好的模型直接进行 Text-to-Image 的代码。  
未来将补充 fine-tune 的代码。  

## fine-tune

### DreamBooth

&emsp;&emsp;DreamBooth 是一种个性化 text2image 模型的方法，例如给出一个物体/人物等的几张（3~5张）图像就能够 fine-tune Stable Diffusion，使模型能够“学会”这个物体，从而在后续的生成时能够更加准确地生成这个物体/人物。
