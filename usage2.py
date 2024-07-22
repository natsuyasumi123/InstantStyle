from diffusers import StableDiffusionXLPipeline
from diffusers.image_processor import IPAdapterMaskProcessor
from transformers import CLIPVisionModelWithProjection
from PIL import Image
import torch

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "./", subfolder="models/image_encoder", torch_dtype=torch.float16
).to("cuda")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "RunDiffusion/Juggernaut-XL-v9", torch_dtype=torch.float16, image_encoder=image_encoder, variant="fp16"
).to("cuda")

pipe.load_ip_adapter(
    ["ostris/ip-composition-adapter", "./"],
    subfolder=["", "sdxl_models"],
    weight_name=[
        "ip_plus_composition_sdxl.safetensors",
        "ip-adapter_sdxl_vit-h.safetensors",
    ],
    image_encoder_folder=None,
)

scale_1 = {
    "down": [[0.0, 0.0, 1.0]],
    "mid": [[0.0, 0.0, 1.0]],
    "up": {"block_0": [[0.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 0.0, 1.0]], "block_1": [[0.0, 0.0, 1.0]]},
}
# activate the first IP-Adapter in everywhere in the model,
# configure the second one for precise style control to each masked input.
pipe.set_ip_adapter_scale([1.0, scale_1])

processor = IPAdapterMaskProcessor()
female_mask = Image.open("./assets/female_mask.png")
male_mask = Image.open("./assets/male_mask.png")
background_mask = Image.open("./assets/background_mask.png")
composition_mask = Image.open("./assets/composition_mask.png")
mask1 = processor.preprocess([composition_mask], height=1024, width=1024)
mask2 = processor.preprocess([female_mask, male_mask, background_mask], height=1024, width=1024)
mask2 = mask2.reshape(1, mask2.shape[0], mask2.shape[2], mask2.shape[3])   # output -> (1, 3, 1024, 1024)

ip_female_style = Image.open("./assets/ip_female_style.png")
ip_male_style = Image.open("./assets/ip_male_style.png")
ip_background = Image.open("./assets/ip_background.png")
ip_composition_image = Image.open("./assets/ip_composition_image.png")

image = pipe(
    prompt="high quality, cinematic photo, cinemascope, 35mm, film grain, highly detailed",
    negative_prompt="",
    ip_adapter_image=[ip_composition_image, [ip_female_style, ip_male_style, ip_background]],
    cross_attention_kwargs={"ip_adapter_masks": [mask1, mask2]},
    guidance_scale=6.5,
    num_inference_steps=25,
).images[0]
image
