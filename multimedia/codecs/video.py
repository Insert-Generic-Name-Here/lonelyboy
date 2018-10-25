import numpy as np
import sys, os

sys.path.append(os.path.join('.','..'))
import codecs.image as imgCodecs
from image import QuantizeImage
from video import QuantizeVideo, CalculateFrameDifference, ReconstructVideo



def __EncodeVideo(frames):
    enc_list= []
    for frame in frames:
        enc_list.append(imgCodecs.RLE_Encode(frame))
    return enc_list


def RLE_Encode(bw_frames):
    '''
    RLE Encoding of a **GrayScale** Video''s frames 
    '''
    # calculate the frame difference matrix
    diffs = CalculateFrameDifference(bw_frames)
    # quantize the matrix
    q_diffs = QuantizeVideo(diffs, Q=100)
    # compress quantized matrix and 1st frame
    q_diffs_comp = __EncodeVideo(q_diffs)

    img1_q = QuantizeImage(bw_frames[0,:,:])
    first_img_comp = imgCodecs.RLE_Encode(img1_q)
    
    return first_img_comp, q_diffs_comp, bw_frames[0,:,:].shape


def RLE_Decode(frames, dims):
    '''
    RLE Decoding of a **GrayScale** Video''s frames 
    '''
    dec_list = []
    for frame in frames:    
        original_dimensions = dims
        decoded = []

        for i in frame:
            symbol, count = i.split('!')
            decoded.extend([float(symbol)]*int(count))   

        dec = np.array(decoded, dtype=np.float64).reshape(original_dimensions)
        dec_list.append(dec)
        dec_vid = np.array(dec_list)
        rec_vid = ReconstructVideo(dec_vid[0], dec_vid[1:])
    return rec_vid


    