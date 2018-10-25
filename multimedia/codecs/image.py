import numpy as np


def RLE_Encode(image):
    cnt = 0
    curr_el = image[0,0]

    encoded = []

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i,j] == curr_el:
                cnt += 1
            else:
                encoded.append('{0}!{1}'.format(curr_el, cnt))
                curr_el = image[i, j]
                cnt = 1        

    encoded.append('{0}!{1}'.format(curr_el, cnt))
    return encoded


def RLE_Decode(encodedImage, dims):
    original_dimensions = dims
    decoded = []

    for i in encodedImage:
        symbol, count = i.split('!')
        decoded.extend([float(symbol)]*int(count))   

    decoded = np.array(decoded, dtype=np.float64).reshape(original_dimensions)
    return np.array(decoded)