from image import DivideToMacroblocks
from metrics import SAD
import numpy as np
import imageio
import os



def ConvertToGrayscale(frames):
    '''
    Converts a Video''s Frames from RGB to GrayScale
        frames: the NxM numpy array of the Video''s frames
    '''
    frames_bw = []
    for frame in frames:
        frames_bw.append(np.dot(frame[:,:,:3], [0.2989, 0.587, 0.114]))
    return np.array(frames_bw)


def QuantizeVideo(frames, Q=10):
    '''
    Quantize a Video''s Frames.
        frames: the NxM numpy array of the Video''s frames
        Q: Quantization Coefficient
    '''
    frames_q = []
    for frame in frames:
        frames_q.append(np.around(np.around(frame/Q, 2)*Q, 2))
    return frames_q


def ParseVideo(videoPath):
    vid = imageio.get_reader(videoPath,  'ffmpeg')
    fps = vid.get_meta_data()['fps']
    num = 0
    frames = []
    while 1:
        try:
            image = vid.get_data(num)
            frames.append(image)
            num += 1
        except IndexError:
            break
    return np.array(frames), fps


def CalculateFrameDifference(frames):
    dif_lst = []
    for i in range(1, frames.shape[0]):
        dif_lst.append(frames[i-1,:,:]-frames[i,:,:])
    return np.array(dif_lst)


def ReconstructVideo(img1, diffs):
    ''' Reconstruct a Video from the **Difference Frames** and the **1st Frame**'''
    final_vid  = [img1]
    cnt = 1
    for frame in diffs:
        final_vid.append(final_vid[-1]-frame)

    return np.array(final_vid)


def CreateNeighborhood(referenceFrame, indexOfMacroblock, macroblock_size=16, k=16):
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


def CalculateNeigborhoodSAD(targetMacroblock, referenceFrame_neighbor_macroblocks):
    SADvals = []
        
    for macroblock in referenceFrame_neighbor_macroblocks:
        if macroblock is not None:
            SADvals.append(SAD(macroblock, targetMacroblock))
        else:
            SADvals.append(np.Inf)
    
    return np.array(SADvals).reshape((3,3))


def LogarithmicSearch(referenceFrame, targetMacroblock, indexOfMacroblock, macroblock_size=16, k=16):
    if (k == 0):
        return indexOfMacroblock, referenceFrame[indexOfMacroblock[0]:indexOfMacroblock[0]+macroblock_size, indexOfMacroblock[1]:indexOfMacroblock[1]+macroblock_size] # motionVector END (To_WIDTH, To_HEIGHT), return Predicted Frame
    
    referenceFrame_neighbor_macroblocks = CreateNeighborhood(referenceFrame, indexOfMacroblock, macroblock_size, k)

    SAD_values = CalculateNeigborhoodSAD(targetMacroblock, referenceFrame_neighbor_macroblocks)
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
    return LogarithmicSearch(referenceFrame, targetMacroblock, tuple(newIndexOfMacroblock), macroblock_size, newK)


def MotionCompensation(referenceFrame, targetFrame, macroblock_size=16):
    predictedBlocks = []
    motionVectors = []
    
    targetMacroblocks = DivideToMacroblocks(targetFrame, macroblock_size)
    for i in range(targetMacroblocks.shape[0]):
        for j in range(targetMacroblocks.shape[1]):
            motionVectorSTART = (i*macroblock_size, j*macroblock_size)
            indexofBlock = (i*macroblock_size, j*macroblock_size)
            motionVectorEND, prediction = LogarithmicSearch(referenceFrame, targetMacroblocks[i,j,:,:], indexofBlock)
            predictedBlocks.append(prediction)
            motionVectors.append(motionVectorSTART+motionVectorEND)

#     print (len(motionVectors))
    predictedBlocks = np.array(predictedBlocks).reshape(targetMacroblocks.shape)
    motionVectors = np.array(motionVectors, dtype=(int,4)).reshape((targetMacroblocks.shape[0], targetMacroblocks.shape[1], 4))
    return predictedBlocks, motionVectors


def SaveVideo(frames, fps, savePath='.', name='sample'):
    ''' Saves video '''
    writer = imageio.get_writer(os.path.join(savePath,name+'.mp4'), fps=fps, mode="I")

    for frame in frames:
        writer.append_data(frame)
    writer.close()


def GroupConsecutives(vals, step=1):
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


def MakeVarList(mbs):
    ''' Make a list (a,x,y)
        a = max value of variance in each macroblock
        x,y = macroblock's coordinates
        
        input: macroblocks of differnce image'''
    lst = []
    for y, macby in enumerate(mbs):
        for x, macbx in enumerate(macby):
            lst.append((max(pd.DataFrame(macbx).var()), y, x))
            
    return lst


def SwapObject(coords, frame, back):
    ''' Swap frame's macroblocks given as coords with the same ones from back '''
    tmp = frame
    for tup in coords:
        y,x = tup
        tmp[y][x] = back[y][x]
        
    return tmp


def HasNeighbors(point, lst, num=1):
    ''' Return True if point has atleast #num of neighbors in list lst (other than itself)'''
    neighs = []
    for cand in lst:
        if abs(cand[0]-point[0])<=1 and abs(cand[1]-point[1])<=1 and point != cand:
            neighs.append(cand)
            
    if len(neighs)>=num:
        return True
    else:
        return False      
        

def ReturnRange(lst, min_neighs=1):
    '''Return the full area that coords of lst cover, creating a rectangle around said coords. 
        Used for clustering macroblocks for better frame swapping'''
    neighs=  []
    final_range = []
    for point in lst:
        if HasNeighbors(point, lst, num=min_neighs):
            neighs.append(point)
    
    xs, ys = [val[0] for val in neighs], [val[1] for val in neighs]
    
    if len(xs)==0 or len(ys)==0:
        return []
    for i in range(min(xs), max(xs)+1):
        for j in range(min(ys), max(ys)+1):
            final_range.append((i,j))
            
    return final_range


def ReturnMem(lst):
    ''' Returns the set list that contains the macroblocks to be swap from all the lists inside lst'''
    tmp = []
    for el in lst:
        tmp += el
        
    return list(set(tmp))


def FindMovingObject(lst): #V2
    ''' Returns the coordinates of the macroblocks that have a higher max variance value than the upper inner fence'''
    vals = np.array([val[0] for val in lst])
    Q75 = np.percentile(vals, 75)
    iqr = Q75 - np.percentile(vals, 25)
    
    outs = []
    for tup in lst:
        if tup[0]>Q75+(1.5*iqr):
            outs.append((tup[1], tup[2]))
    
    return outs