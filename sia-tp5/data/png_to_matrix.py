from PIL import Image
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import glob


def resize_and_convert_to_grayscale(image_path, target_size=(24, 24), resampling=Image.NEAREST):
    # Open the PNG image
    img = Image.open(image_path)

    # Resize the image to the target size
    img_resized = img.resize(target_size, resampling)

    # Convert the resized image to grayscale
    gray_img = img_resized.convert('L')

    # Convert the PIL image to a NumPy array
    gray_array = np.array(gray_img)

    gray_array = gray_array / 255
    gray_array = 1 - gray_array

    return gray_array


if __name__ == '__main__':
    image_paths = glob.glob(f"./emojis/*.png")

    colorscale = [[0, 'white'], [1, 'black']]
    N = len(image_paths)
    fig = make_subplots(rows=1+N//3, cols=3)
    for i, image in enumerate(image_paths):
        resized_grayscale_matrix = resize_and_convert_to_grayscale(image)
        print(image)
        print("(", end="")
        for row in resized_grayscale_matrix:
            for col in row:
                print(f"{col}, ", end="")
        print(")")
        print("-------------")
        fig.add_trace(go.Heatmap(z=np.flipud(resized_grayscale_matrix), colorscale=colorscale),
                      row=1+i//3, col=1+i%3)
    fig.show()
