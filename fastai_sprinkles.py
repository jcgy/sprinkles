import random

def _sprinkles(x, size: int, perc: float, style='black', channels=3):
    """Produces 'sprinkles' image augmentation on input
    see: https://medium.com/@lessw/progressive-sprinkles-a-new-data-augmentation-for-cnns-and-helps-achieve-new-98-nih-malaria-6056965f671a
    
    Parameters
    ----------
    x = pytorch tensor, input image
    size = int, specifying sprinkle width and height in pixels
    perc = float, approximate (sprinkles can overlap) percentage of image to occlude
    style = string, option of ['black', 'frosted', 'mean'] for style of sprinkle
    channels = int, number of image channels, defaults to 3
    """
    img = x.detach().clone()
    number_of_pixels_to_frost = perc * np.ceil((img.shape[1] * img.shape[2]))
    number_of_sprinkles = int(np.ceil(number_of_pixels_to_frost / (size * size)))
    for sprinkle in range(0, number_of_sprinkles):
        # set boundaries to prevent out of index errors
        row_options = range((size), (img.shape[1] - size))
        col_options = range((size), (img.shape[2] - size))
        # get random index position
        row = np.random.choice(row_options, replace=False)
        col = np.random.choice(col_options, replace=False)
        # change initial pixel value for all channels based on style
        if style == "black":
            replacement = 0
            for c in range(0, channels):
                img[c, row, col] = replacement
        elif style == "mean":
            channel_means = []
            for c in range(0, channels): 
                mean = torch.mean(img[c, :, :])
                channel_means.append(mean)
            for c in range(0, channels):
                img[c, row, col] = channel_means[c]
        elif style == "frosted":
            for c in range(0, channels):
                img[c, row, col] = np.random.randint(0, 1)
        # randomly determine fill direction
        horizontal_fill_direction = np.random.choice(["left", "right"])
        vertical_fill_direction = np.random.choice(["up", "down"])
        # replace pixel values for each sprinkle
        if (horizontal_fill_direction == "left") & (vertical_fill_direction == "up"):
            for i in (range(0, (size - 1))):
                for j in (range(0, (size - 1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            img[c, (row - j), (col - i)] = np.random.uniform(0, 1)
                        elif style == 'mean':
                            img[c, (row - j), (col - i)] = channel_means[c]
                        else:
                            img[c, (row - j), (col - i)] = 0
        elif (horizontal_fill_direction == "left") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            img[c, (row - j), (col + i)] = np.random.uniform(0, 1)
                        elif style == 'mean':
                            img[c, (row - j), (col - i)] = channel_means[c]
                        else:
                            img[c, (row - j), (col + i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "up"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            img[c, (row + j), (col - i)] = np.random.uniform(0, 1)
                        elif style == 'mean':
                            img[c, (row - j), (col - i)] = channel_means[c]
                        else:
                            img[c, (row + j), (col - i)] = 0
        elif (horizontal_fill_direction == "right") & (vertical_fill_direction == "down"):
            for i in (range(0, (size-1))):
                for j in (range(0, (size-1))):
                    for c in range(0, channels):
                        if style == 'frosted':
                            img[c, (row - j), (col - i)] = np.random.uniform(0, 1)
                        elif style == 'mean':
                            img[c, (row - j), (col - i)] = channel_means[c]
                        else:
                            img[c, (row - j), (col - i)] = 0
    return img
sprinkles = TfmPixel(_sprinkles)