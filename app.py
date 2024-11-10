import torch
from torchvision import transforms
import gradio as gr

from Model import TTR
from dataset.charMapper import CharMapper

# arguments
model_path = "./experiments/real_train/model_best.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def getTransforms():
    return transforms.Compose([
        transforms.Resize((32, 128), transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])


mapper = CharMapper()
model = TTR({"img_size": [32, 128],
             "patch_size": [4, 4],
             "embed_dim": 512,
             "num_heads": 8,
             "position_attention_hidden": 64,
             "mask_ratio": 0.0
             })
model.load_state_dict(torch.load(model_path)["model"])
model.eval()
model = model.to("cuda:0")

preprocess = getTransforms()


def inference(raw_image):
    batch = preprocess(raw_image).unsqueeze(0).to(device)
    outputs = model(batch)
    preds = torch.max(outputs, dim=2)[1]
    pred_text = mapper.reverseMapper(preds[0])
    return pred_text


inputs = [gr.Image(type='pil', interactive=True, )]
outputs = gr.components.Textbox(label="Caption")
title = "MViT-TR"
paper_link = "https://www.sciencedirect.com/science/article/pii/S2215098624002672"
github_link = "https://github.com/serdaryildiz/MViT-TR"
description = f"<p style='text-align: center'><a href='{github_link}' target='_blank'>MViT-TR</a> : Masked Vision Transformer for Text Recognition"
examples = [
    ["fig/0.jpg"],
    ["fig/145.jpg"],
    ["fig/195.jpg"],
    ["fig/270.jpg"],
]
article = f"<p style='text-align: center'><a href='{paper_link}' target='_blank'>Paper</a> | <a href='{github_link}' target='_blank'>Github Repo</a></p>"
css = ".output-image, .input-image, .image-preview {height: 600px !important}"

iface = gr.Interface(fn=inference,
                     inputs=inputs,
                     outputs=outputs,
                     title=title,
                     description=description,
                     examples=examples,
                     article=article,
                     css=css)
iface.launch()
