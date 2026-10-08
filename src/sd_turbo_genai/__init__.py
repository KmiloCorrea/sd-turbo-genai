import torch #Ejecutar operaciones matematicas del modelo de IA
from diffusers import AutoPipelineForText2Image #Diffusers es una libreria especializada en modelos generativos

print("Cargando el modelo ...")

modelo = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo",
    torch_dtype=torch.float32 #float o float32
    #torch.floatXX, indica la precisión numerica
    # con que se cargan los pesos (decimales, la precisión
    # para generar la imagen)
)

modelo = modelo.to("cpu")

prompt = input("Escribe el prompt de la imagen que quieres crear: ")
negative_prompt = "low quality, worst quality, low resolution, blurry, out of focus, deformed, badly drawn, bad anatomy, bad hands, extra fingers, mutated limbs, text, watermark, signature, logo, compression artifacts, ugly"

print("Generando imagen... ")

imagen = modelo(
    prompt=prompt, #Instrucciones o detalle para crear la imagen
    negative_prompt=negative_prompt,
    num_inference_steps=10, #Las veces que va a tratar de crear o refinar la imagen
    guidance_scale=2.0, #Que tan fiel debe ser al prompt (creatividad, temperatura)
    height=1024,
    width=1024
).images[0]

imagen.save("imagen.png")

print("Imagen guardada.")
