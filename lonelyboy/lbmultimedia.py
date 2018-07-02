from io import StringIO
from PIL import Image
import imageio
import numpy as np
import pylab
import matplotlib.pyplot as plt
import os


def QuantizeVid(frames, Q=10):
    frames_q = []
    for frame in frames:
        frames_q.append(np.around(np.around(frame/Q, 2)*Q, 2))
    return frames_q

def QuantizeImage(img, Q=10):
    return np.around(np.around(img/Q, 2)*Q, 2)

def VidConvertGrayScale(rgbVid):
    '''Converts numpy array vid to BW vid'''
    frames_bw = []
    for frame in rgbVid:
        frames_bw.append(np.dot(frame[:,:,:3], [0.2989, 0.587, 0.114]))
    return np.array(frames_bw)

def parse_vid(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    num = 0
    frames = []
    while 1:
        try:
            image = vid.get_data(num)
            frames.append(image)
            num+=1
        except IndexError:
            break

    return np.array(frames), fps

def calc_diff(frames):
    dif_lst = []
    for i in range(1, frames.shape[0]):
        dif_lst.append(frames[i-1,:,:]-frames[i,:,:])
    return np.array(dif_lst)

def comp_vid(frames):
    enc_list= []
    for frame in frames:
        enc_list.append(comp_image(frame))
    return enc_list

def comp_image(frame):
    cnt = 0
    curr_el = frame[0,0]

    encoded = []

    for i in range(frame.shape[0]):
        for j in range(frame.shape[1]):
            if frame[i,j] == curr_el:
                cnt += 1
            else:
                encoded.append('{0}!{1}'.format(curr_el, cnt))
                curr_el = frame[i, j]
                cnt = 1        

    encoded.append('{0}!{1}'.format(curr_el, cnt))
    return encoded

def vid_encode(filename):
    # parse RGB video
    frames, fps = parse_vid(filename)
    # convert to BW
    bw_frames = VidConvertGrayScale(frames)
    # calculate the frame difference matrix
    diffs = calc_diff(bw_frames)
    # quantize the matrix
    q_diffs = QuantizeVid(diffs, Q=100)
    # compress quantized matrix and 1st frame
    q_diffs_comp = comp_vid(q_diffs)
    img1_q = QuantizeImage(bw_frames[0,:,:])
    first_img_comp = comp_image(img1_q)
    
    return first_img_comp, q_diffs_comp, bw_frames[0,:,:].shape, fps


def decode_vid(frames, dims):
    dec_list = []
    for frame in frames:    
        original_dimensions = dims
        decoded = []

        for i in frame:
            symbol, count = i.split('!')
            decoded.extend([float(symbol)]*int(count))   

        dec = np.array(decoded, dtype=np.float64).reshape(original_dimensions)
        dec_list.append(dec)
    return np.array(dec_list)

def decode_image(frame, dims):
    original_dimensions = dims
    decoded = []

    for i in frame:
        symbol, count = i.split('!')
        decoded.extend([float(symbol)]*int(count))   

    decoded = np.array(decoded, dtype=np.float64).reshape(original_dimensions)
    return np.array(decoded)

def reconstruct_vid(img1, diffs):
    final_vid  = [img1]
    cnt = 1
    for frame in diffs:
        final_vid.append(final_vid[-1]-frame)
        
    
    return np.array(final_vid)

def save_vid(frames, path, name, fps):
    writer = imageio.get_writer(os.path.join(path,name), fps=fps,mode="I")

    for frame in frames:
        writer.append_data(frame)
    writer.close()


def imageEntropy(image, base=2):
    _, counts = np.unique(image, return_counts=True)
    return entropy(counts, base=base)

def readVideoFrames(filename):
    vid = imageio.get_reader(filename,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    num = 0
    frames = []
    while 1:
        try:
            image = vid.get_data(num)
            frames.append(image)
            num+=1
        except IndexError:
            break

    return np.array(frames), fps

def makeGrayScale(rgbVid):
    '''Converts numpy array vid to grayscale vid'''
    frames_bw = []
    for frame in rgbVid:
        frames_bw.append(np.around(np.dot(frame[:,:,:3], [0.2989, 0.587, 0.114])))
    return np.array(frames_bw)

def calculateDifferenceFrames(frames):
    dif_lst = []
    for i in range(1, frames.shape[0]):
        dif_lst.append(frames[i,:,:]-frames[i-1,:,:])
    return np.array(dif_lst)

def divideToMacroblocks(frame, macroblock_size=16):
    macroblocks = []
    m, n = frame.shape
    for i in range(0, m, macroblock_size):
        for j in range(0, n, macroblock_size):
            macroblock = frame[i:i+macroblock_size,j:j+macroblock_size]
    #         print (macroblock.shape)
            if macroblock.shape == (macroblock_size, macroblock_size):
    #             print ('works')
                macroblocks.append(macroblock)
            else:
    #             print ('something\'s goin\' on')
                try:
                    macroblock = np.vstack((macroblock, np.zeros(macroblock.shape[0], macroblock_size-macroblock.shape[1])))
                except TypeError:
                    pass
                try:
                    macroblock = np.hstack((macroblock, np.zeros(macroblock_size-macroblock.shape[0], macroblock.shape[1])))
                except TypeError:
                    pass
                macroblocks.append(macroblock)
    return np.array(macroblocks).reshape((int(m/macroblock_size), int(n/macroblock_size), macroblock_size, macroblock_size))

def createNeighborhood(referenceFrame, indexOfMacroblock, macroblock_size=16, k=16):
    neighborhood = []
#     print (indexOfMacroblock)
    for i in range(indexOfMacroblock[0]-k, indexOfMacroblock[0]+k+1, k):
        for j in range(indexOfMacroblock[1]-k, indexOfMacroblock[1]+k+1, k):
            if (i >= 0 and j >= 0 and i+macroblock_size < referenceFrame.shape[0] and j+macroblock_size < referenceFrame.shape[1]):
#                 print (i,j)
                neighborhood.append(referenceFrame[i:i+macroblock_size, j:j+macroblock_size])
            else:
                neighborhood += [None]
    return neighborhood

def SAD(referenceMacroblock, targetMacroblock):
#     print (targetMacroblock.shape, referenceMacroblock.shape)
    return np.sum(np.abs(targetMacroblock - referenceMacroblock))

def calculateSAD(targetMacroblock, referenceFrame_neighbor_macroblocks):
    SADvals = []
        
    for macroblock in referenceFrame_neighbor_macroblocks:
        if macroblock is not None:
            SADvals.append(SAD(macroblock, targetMacroblock))
        else:
            SADvals.append(np.Inf)
    
    return np.array(SADvals).reshape((3,3))

def logarithmicSearch(referenceFrame, targetMacroblock, indexOfMacroblock, macroblock_size=16, k=16):
    if (k == 0):
        return indexOfMacroblock, referenceFrame[indexOfMacroblock[0]:indexOfMacroblock[0]+macroblock_size, indexOfMacroblock[1]:indexOfMacroblock[1]+macroblock_size] # motionVector END (To_WIDTH, To_HEIGHT), return Predicted Frame
    
    referenceFrame_neighbor_macroblocks = createNeighborhood(referenceFrame, indexOfMacroblock, macroblock_size, k)

    SAD_values = calculateSAD(targetMacroblock, referenceFrame_neighbor_macroblocks)
#     print (SAD_values)
    indexofMinimumSAD = divmod(SAD_values.argmin(), SAD_values.shape[1])
    newIndexOfMacroblock = list(indexOfMacroblock)
    
    if (indexofMinimumSAD[0] == 0):
        newIndexOfMacroblock[0] = indexOfMacroblock[0] - k
    elif (indexofMinimumSAD[0] == 2):
        newIndexOfMacroblock[0] = indexOfMacroblock[0] + k
    
    if (indexofMinimumSAD[1] == 0):
        newIndexOfMacroblock[1] = indexOfMacroblock[1] - k
    elif (indexofMinimumSAD[1] == 2):
        newIndexOfMacroblock[1] = indexOfMacroblock[1] + k

    if (indexofMinimumSAD[0] == 1 and indexofMinimumSAD[1] == 1):
        newK = k//2
    else:
        newK = k       
#     print (indexofMinimumSAD)
#     print (newIndexOfMacroblock)
    return logarithmicSearch(referenceFrame, targetMacroblock, tuple(newIndexOfMacroblock), macroblock_size, newK)

def motionCompensation(referenceFrame, targetFrame, macroblock_size=16):
    predictedBlocks = []
    motionVectors = []
    
    targetMacroblocks = divideToMacroblocks(targetFrame, macroblock_size)
    for i in range(targetMacroblocks.shape[0]):
        for j in range(targetMacroblocks.shape[1]):
            motionVectorSTART = (i*macroblock_size, j*macroblock_size)
            indexofBlock = (i*macroblock_size, j*macroblock_size)
            motionVectorEND, prediction = logarithmicSearch(referenceFrame, targetMacroblocks[i,j,:,:], indexofBlock)
            predictedBlocks.append(prediction)
            motionVectors.append(motionVectorSTART+motionVectorEND)

#     print (len(motionVectors))
    predictedBlocks = np.array(predictedBlocks).reshape(targetMacroblocks.shape)
    motionVectors = np.array(motionVectors, dtype=(int,4)).reshape((targetMacroblocks.shape[0], targetMacroblocks.shape[1], 4))
    return predictedBlocks, motionVectors

def imageReconstructFromBlocks(blocks):
    lines = []
    for i in range(blocks.shape[0]):
        line = []
        for j in range(blocks.shape[1]):
            line.append(blocks[i,j,:,:])
        line = np.hstack(line)
        lines.append(line)
    return np.vstack(lines)

def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result

def make_var_list(mbs):
    ''' Make a list (a,x,y)
        a = max value of variance in each macroblock
        x,y = macroblock's coordinates
        
        input: macroblocks of differnce image'''
    lst = []
    for y, macby in enumerate(mbs):
        for x, macbx in enumerate(macby):
            lst.append((max(pd.DataFrame(macbx).var()), y, x))
            
    return lst

def swap_object(coords, frame, back):
    ''' Swap frame's macroblocks given as coords with the same ones from back '''
    tmp = frame
    for tup in coords:
        y,x = tup
        tmp[y][x] = back[y][x]
        
    return tmp

def reconstruct_vid(img1, diffs):
    ''' Reconstruct a video from the difference frames and the 1st frame'''
    final_vid  = [img1]
    cnt = 1
    for frame in diffs:
        final_vid.append(final_vid[-1]-frame)
        
    
    return np.array(final_vid)

def has_n_neighbor(point, lst, num=1):
    ''' Return True if point has atleast #num of neighbors in list lst (other than itself)'''
    neighs = []
    for cand in lst:
        if abs(cand[0]-point[0])<=1 and abs(cand[1]-point[1])<=1 and point != cand:
            neighs.append(cand)
            
    if len(neighs)>=num:
        return True
    else:
        return False
        
        

def return_range(lst, min_neighs=1):
    '''Return the full area that coords of lst cover, creating a rectangle around said coords. 
        Used for clustering macroblocks for better frame swapping'''
    neighs=  []
    final_range = []
    for point in lst:
        if has_n_neighbor(point, lst, num=min_neighs):
            neighs.append(point)
    
    xs, ys = [val[0] for val in neighs], [val[1] for val in neighs]
    
    if len(xs)==0 or len(ys)==0:
        return []
    for i in range(min(xs), max(xs)+1):
        for j in range(min(ys), max(ys)+1):
            final_range.append((i,j))
            
    return final_range

def return_mem(lst):
    ''' Returns the set list that contains the macroblocks to be swap from all the lists inside lst'''
    tmp = []
    for el in lst:
        tmp += el
        
    return list(set(tmp))

def find_moving_object_v2(lst):
    ''' Returns the coordinates of the macroblocks that have a higher max variance value than the upper inner fence'''
    vals = np.array([val[0] for val in lst])
    Q75 = np.percentile(vals, 75)
    iqr = Q75 - np.percentile(vals, 25)
    
    outs = []
    for tup in lst:
        if tup[0]>Q75+(1.5*iqr):
            outs.append((tup[1], tup[2]))
    
    return outs

def save_vid(frames, resources_path='.', name='sample'):
    ''' Saves video '''
    writer = imageio.get_writer(os.path.join(resources_path,name+'.mp4'), fps=fps,mode="I")

    for frame in frames:
        writer.append_data(frame)
    writer.close()