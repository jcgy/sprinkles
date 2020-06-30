import cv2
import numpy as np
import random

def sprinkles(img, size, perc, style='black', channels=3):
    """Produces 'sprinkles' image augmentation on input
    see: https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a
    
    Parameters
    ----------
    x = np.array of input image
    size = int specifying sprinkle width and height in pixels
    perc = approximate (sprinkles can overlap) percentage of image to occlude
    style = string, option of ['black', 'frosted', 'mean'] for style of sprinkle
    channels = int, number of image channels, defaults to 3
    """
    x = img.copy()
    number_of_pixels_to_change = perc * np.ceil((x.shape[1] * x.shape[2]))
    number_of_sprinkles = int(np.ceil(number_of_pixels_to_change / (size * size)))
    for sprinkle in range(0, number_of_sprinkles):
        # set boundaries to prevent out of index errors
        row_options = range((size), (x.shape[1] - size))
        col_options = range((size), (x.shape[2] - size))
        # get random index position
        row = np.random.choice(row_options, replace=False)
        col = np.random.choice(col_options, replace=False)
        # change initial pixel value for all channels based on style
        if style == "black":
            replacement = 0
            for c in range(0, channels):
                x[c, row, col] = replacement
        elif style == "mean":
            channel_means = cv2.mean(x)
            for c in range(0, channels):
                x[c, row, col] = channel_means[c]
        elif style == "frosted":
            for c in range(0, channels):
                x[c, row, col] = np.random.randint(0, 1)
        # randomly determine fill direction
        horizontal_fill_direction = np.random.choice(["left", "right"])
        vertical_fill_direction = np.random.choice(["up", "down"])
        # replace pixel values
        if (horizontal_fill_direction == "left") & (vertical_fill_direction == "up"):
            for i in (range(0, (size - 1))):
                for j in (range(0, (size - 1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            x[(row - j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = channel_means[c]
                        else:
                            x[(row - j), (col - i)] = 0
        elif (horizontal_fill_direction == "left") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            x[(row - j), (col + i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = channel_means[c]
                        else:
                            x[(row - j), (col + i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "up"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            x[(row + j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = channel_means[c]
                        else:
                            x[(row + j), (col - i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            x[(row - j), (col - i)][c] = np.random.randint(0, 255)
                        elif style == 'mean':
                            x[(row - j), (col - i)][c] = channel_means[c]
                        else:
                            x[(row - j), (col - i)] = 0
    return np.array(x)