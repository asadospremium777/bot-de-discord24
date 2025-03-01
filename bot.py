#esto es para instalarse en google colab
#!pip install discord
#!pip install tensorflow==2.12.0 keras==2.12.0
#!unzip converted_keras.zip
import discord
from discord.ext import commands
import random
import os
import asyncio

# La variable intents almacena los privilegios del bot
intents = discord.Intents.default()
# Activar el privilegio de lectura de mensajes
intents.message_content = True
# Crear un bot en la variable cliente y transferirle los privilegios
bot = commands.Bot(command_prefix = "$",intents=intents)

@bot.event
async def on_ready():
    print(f'Hemos iniciado sesión como {bot.user}')

@bot.command()
async def hello(ctx):
    await ctx.send("HI!")

@bot.command()
async def bye(ctx):
    await ctx.send("😞")

@bot.command()
async def repeat(ctx, times: int, content='repeating...'):
    """Repeats a message multiple times."""
    for i in range(times):
        await ctx.send(content)

@bot.command()
async def mem_aleatorio(ctx):
    mem_alet = random.choice(os.listdir("images"))
    with open(f'images/{mem_alet}', 'rb') as f:
        # ¡Vamos a almacenar el archivo de la biblioteca Discord convertido en esta variable!
        picture = discord.File(f)
    # A continuación, podemos enviar este archivo como parámetro.
    await ctx.send(file=picture)


@bot.command()
async def roll(ctx, dice: str):
    """Rolls a dice in NdN format."""
    try:
        rolls, limit = map(int, dice.split('d'))
    except Exception:
        await ctx.send('Format has to be in NdN!')
        return

    result = ', '.join(str(random.randint(1, limit)) for r in range(rolls))
    await ctx.send(result)

@bot.command()
async def upload_image(ctx):
    if len(ctx.message.attachments) == 0:
        await ctx.send("No se ha encontrado ninguna imagen adjunta.")
    else:
        # Iterar sobre los archivos adjuntos
        for attachment in ctx.message.attachments:
            if attachment.filename.endswith(('jpg', 'jpeg', 'png')):
                # Guardar la imagen en el sistema de archivos local
                filepath = f"images/{attachment.filename}"
                file_name = attachment.filename
                await attachment.save(filepath)
                # Enviar la URL de la imagen de vuelta al usuario
                await ctx.send(get_class(model_path="/content/keras_model.h5",labels_path="/content/labels.txt", image_path=f"images/{file_name}"))

            else:
                await ctx.send(f"El archivo {attachment.filename} no es una imagen válida.")

#Instead of calling bot.run() directly, create a function that calls it
def start_bot():
    asyncio.get_event_loop().run_until_complete(bot.start("token"))
#And run it in a loop using nest_asyncio to avoid conflict with jupyter's asyncio loop
import nest_asyncio
nest_asyncio.apply()

start_bot()
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np


def get_class(model_path="Path to model", labels_path="Path to labels", image_path="Path to image"):
    # Disable scientific notation for clarity
    np.set_printoptions(suppress=True)

    # Load the model
    model = load_model(model_path, compile=False)

    # Load the labels
    class_names = open(labels_path, "r").readlines()

    # Create the array of the right shape to feed into the keras model
    # The 'length' or number of images you can put into the array is
    # determined by the first position in the shape tuple, in this case 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    # Replace this with the path to your image
    image = Image.open(image_path).convert("RGB")

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    data[0] = normalized_image_array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print("Class:", class_name[2:], end="")
    print("Confidence Score:", confidence_score)
