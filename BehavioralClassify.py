#!/usr/bin/env python3
'''
Functions for classifying behavioral states

Authors: Jimmy Chen, Shreya Mantripragada, Emma Dionne, Brian R. Mullen
Date: 2019-07-03
'''
import os
import sys
sys.path.append('../pyWholeBrain')
import numpy as np
import colorsys
import cv2
import matplotlib.pyplot as plt

import scipy
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage.morphology import disk, watershed
from skimage.morphology import erosion, dilation, opening, closing

from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter

from hdf5manager import hdf5manager as h5
import wholeBrain as wb

# FUNCTIONS

def getAnglemap(xydim=25):
    #creates angle maps
    print("entered the getAnglemap function")
    o = of(np.random.random((10,10,100)))
    o.angleMapping(xydim=xydim)
    rgb_map = o.angle_map
    #change rgb map back to hsv
    hsv_map = np.zeros((rgb_map.shape[0], rgb_map.shape[1], 3))
    for i in np.arange(hsv_map.shape[0]):
        for j in np.arange(hsv_map.shape[1]):
            hsv_map[i,j,:] = colorsys.rgb_to_hsv(rgb_map[i,j,0], rgb_map[i,j,1], rgb_map[i,j,2])
    #only show angle component(0-360)
    angle_map = hsv_map[:,:,0]*360
    #put 0/360 at the top of the map
    rot_map = angle_map - 90
    rot_map[rot_map < 0] += 360
    return rgb_map, angle_map, rot_map


def localMaxima2d(array_2d):
    # finds local maxima of a given 2d array
    print("entered the localMaxima2d function")
    neighborhood = np.ones((5,5))
    local_max = maximum_filter(array_2d, footprint=neighborhood)==array_2d
    background = (array_2d==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    lmax = local_max ^ eroded_background
    return lmax


def findMeans(array_3d):
    #finds the means of the mice movemment, given a 3d array
    print("entered the findMeans function")
    all_means = []
    x = []
    for i, frame in enumerate(array_3d):
        all_means.append(np.mean(frame))
        x.append(i)
    new_means = np.around(all_means, 3)
    return new_means 


def standardDeviation(array_3d):
    #includes 0s with the standard deviation
    #finds the standard deviation of the mice movement, given a 3d array
    print("entered the standardDeviation function")
    all_deviations = []
    x = []
    for i, frame in enumerate(array_3d):
        mean = np.mean(frame)
        x.append(i)
        deviation = 0
        deviations = []
        for r in frame:
            for c in range(len(r)):
                deviations.append((r[c] - mean) ** 2)
        mean1 = np.mean(deviations)
        current_deviation = mean1 ** (1/2)
        all_deviations.append(current_deviation)
        new_all_deviations = np.around(all_deviations, 3)
    #plt.plot(x, all_deviations, "bo")
    return new_all_deviations


def motionCharacterize(array3d):
    print("entered the motionCharacterize function")
    brain_magnitude = np.zeros(array3d.shape[0])
    for n, frame in enumerate(array3d):
        brain_magnitude[n] = np.mean(frame)

    win_size = 10
    mag_mean = np.convolve(brain_magnitude, np.ones(win_size)/win_size, mode = 'same')

    threshold = np.zeros(mag_mean.shape)
    threshold[mag_mean > 0] = 1 
    frame_ind = np.where(threshold == 1)

    start = []
    end = []
    for i, frame in enumerate(frame_ind[0]):
        if i == 0:
            start.append(frame)
        elif len(frame_ind[0])-1 == i:   
            end.append(frame)
        elif (frame + 1) != (frame_ind[0][i + 1]):
            end.append(frame)
            start.append(frame_ind[0][i + 1])
            
#     Duration of event frames in seconds
    event_frames = (np.array(end) - np.array(start))/30
    
    mag_per_event = np.zeros(array3d.shape[0])
    duration = np.zeros(array3d.shape[0])
    rest = np.zeros_like(duration)

    for i, st in enumerate(start):
        if i == 0:        
            rest[:st] = st
            rest[end[i]:start[i+1]] = start[i+1] - end[i]
        elif i == len(start)-1:
            rest[end[i]:] = rest.shape[0] - end[i]
        else:
            rest[end[i]:start[i+1]] = start[i+1] - end[i]
        mag_per_event[st:end[i]] = np.sum(array3d[st:end[i]])/event_frames[i]
        duration[st:end[i]] = event_frames[i]
    return mag_per_event, duration, rest


def commonOccurences(array_3d):
    print("entered the commonOccurences function")
    dictionary = {}
    for i in array_3d:
        if i in dictionary:
            dictionary[i] += 1
        else:
            dictionary[i] = 1
    return dictionary


def sameSize(list_1, list_2):
   #returns the lists with the same size
   print("entered the sameSize function")
   new_array = []
   if (len(list_1) > len(list_2)):
       divide = len(list_1) / len(list_2)
       if (divide > 2):
           for i in range(len(list_2)):
               new_array.append(list_1[i * int(np.floor(divide))])
       else:
           new_array = list_1[0:len(list_2)]
   elif(len(list_2) > len(list_1)):
       divide = len(list_2) / len(list_1)
       if (divide > 2):
           for i in range(len(list_1)):
               new_array.append(list_2[i* int(np.floor(divide))])
       else:
           new_array = list_2[0:len(list_1)]
   return new_array


def findMode(array_3d):
    print("entered the findMode function")
    means = findMeans(array_3d)
    mode = commonOccurences(means)
    m = []
    for i in means:
        m.append(mode.get(i))
    return m


def findEvent(array3d):
    print("entered the findEvent function")
    means = findMeans(array3d)
    is_event = []
    for i in means:
        if i > 0:
            is_event.append(1)
        else:
            is_event.append(0)
    return is_event


def findRange(array3d):
    print("entered the findRange function")
    return np.around(np.max(array3d, axis = (1,2)), 3)


def sameSizeUp(small, big):
    print("entered the sameSizeUp function")
    scalar = len(big)/len(small)
    scaled = []
    index = 0
    count = 0
    for i in range(len(big)):
        if count >= scalar:
            index += 1
            scaled.append(small[index])
            count = 1
        else:
            scaled.append(small[index])
            count += 1
    return scaled


def findingRangeValues(array2d): 
    print("entered the findingRangeValues function")
    #finding the ranges of snippet of the brain data
    d_switch = [] #the decreasing part of the graph 
    u_switch = [] #the increasing part of the graph

    for i in range(len(array2d) - 1):
        if i == 0 or i == len(array2d):
            continue
        elif array2d[i-1] < array2d[i] and array2d[i] > array2d[i+1]:
            u_switch.append(i)
        elif array2d[i-1] > array2d[i] and array2d[i] < array2d[i+1]:
            d_switch.append(i)
        else:
            continue
            
    return u_switch, d_switch, array2d


def rangeOfSections(u_switch, d_switch, array2d):
    print("entered the rangeOfSections function")
    #u_switch is the list of the indices of all the maxs
    #d_switch is the list of the indices of all the mins
    difference_list = []
    for i in range(len(u_switch)):
        difference_list.append(abs(abs(array2d[u_switch[i]] - array2d[d_switch[i]])))
    return np.around(difference_list, 3)


def maxValueOfEvent(array_3d):
    #rests_events = []
    print("entered the maxValueOfEvent function")
    max_values = [] #final list
    events_max = [] #takes all the maximums
    means_mov = findMeans(array_3d)
    events = findEvent(array_3d)
    i = 0
    while i < len(means_mov):
        if events[i] == 1:
            temp_max = float('-Inf')
            index = i
            while(index < len(means_mov) and events[index] == 1):
                if (means_mov[index] > temp_max):
                    temp_max = means_mov[index]
                index += 1
            i = index + 1
            events_max.append(temp_max)
        else:
            i += 1
    event_index = 0
    index = 0
    while (index < len(events)):
        if events[index] == 0:
            max_values.append(0)
            index += 1
        else:
            while(index < len(events) and events[index] != 0):
                max_values.append(events_max[event_index])
                index += 1
            event_index += 1
    return max_values


def listOfTotalMagnitude(array3d):
    print("entered the listOfTotalMagnitude function")
    total = []
    for i in array3d:
        total.append(np.sum(i))
    return np.around(total, 3)


def surfaceArea(array3d):
    print("entered the surfaceArea function")
    area_count = []
    for i in array3d:
        count = 0
        for r in i:
            for c in r:
                if c > 0:
                    count += 1          
        area_count.append(count)
    return area_count


def totalMagnitude(array3d):
    print("entered the totalMagnitude function")
    total = np.sum(array3d)
    
    return total


def motionPercentage(array_3d):
    print("entered the motionPercentage function")
    total = totalMagnitude(array_3d)
    means = findMeans(array_3d)
    percentages = []
    final_percentages = []
    for i in means:
        percentages.append((i/total)*100)

    for i in percentages:
        final_percentages.append(i*1000)
    final_percentages = np.around(final_percentages, 3)
        
    return final_percentages


def timeContinuity(timecourse, forward_or_backward='forward'):
    print("entered the timeContinuity function")
    continuous_limit = []
    difference_continuous = 0
    
    if forward_or_backward == 'forward':
        print("Calculating forward continuance")
    elif forward_or_backward == 'backward':
        print("Calculating backward continuance")
        timecourse = timecourse[::-1]
    
    for i in timecourse:
        if i == 0:
            difference_continuous += 1
            continuous_limit.append(difference_continuous)
        else:
            difference_continuous = 0
            continuous_limit.append(difference_continuous)
            
    if forward_or_backward == 'backward':
        print("Calculating backward continuance")
        continuous_limit = continuous_limit[::-1]
    
    return continuous_limit


def findingFirstDerivativePoints(array_2d): 
    print("entered the findingFirstDerivativePoints function")
    der = np.zeros_like(array_2d)
    #time = np.arange(der.shape[0])/10
    for i in range(len(array_2d)):
        if i == 0 or i == len(array_2d):
            continue
        else:
            der[i] = array_2d[i] - array_2d[i-1]

    derivative_value = []
    for i in der:
        derivative_value.append(i/0.1)

    new_derivative_value = np.around(derivative_value, 3)
    
    return new_derivative_value


def findingSecondDerivativePoints(array_2d): 
    print("entered the findingSecondDerivativePoints function")
    first_derivative = findingFirstDerivativePoints(array_2d)
    second_derivative = findingFirstDerivativePoints(first_derivative)
    
    return second_derivative


def comparison(list_1, list_2):
    print("entered the comparison function")
    print(len(list_1))
    print(len(list_2))
    new_list = []
    for i in range(len(list_1)):
        difference = list_1[i] - list_2[i]
        new_list.append(difference)
    
    new_new_list = np.around(new_list, 3)
    return new_new_list


def standardDeviationY(array_3d):
    print("entered the standardDeviationY function")
    deviations = []
    for i in range(len(array_3d)):
        sums = []
        for r in array_3d[i]:
            temp = 0
            for c in r:
                if c != 0:
                    temp += c
            sums.append(temp)
        deviations.append(np.std(sums))
    return np.around(deviations, 3)


def findingDistanceBetweenMaxOfEvent(array_3d):
    print("entered the findingDistanceBetweenMaxOfEvent function")
    diff_values = [] #final list
    events_max = [] #takes all the maximums
    final_values = []
    means_mov = findMeans(array_3d)
    events = findEvent(array_3d)
    i = 0
    while i < len(means_mov):
        if events[i] == 1:
            temp_max = float('-Inf')
            index = i
            while(index < len(means_mov) and events[index] == 1):
                if (means_mov[index] > temp_max):
                    temp_max = means_mov[index]
                index += 1
            i = index + 1
            events_max.append(temp_max)
        else:
            i += 1   
            
    for i in range(len(events_max) - 1):
        difference = abs(events_max[i] - events_max[i+1])
        final_values.append(difference)
    
    event_index = 0
    index = 0
    while (index < len(events)):
        if events[index] == 0:
            diff_values.append(0)
            index += 1
        else:
            while(index < len(events) and events[index] != 0):
                if (event_index <= len(final_values)-1):
                    diff_values.append(final_values[event_index])
                index += 1
            event_index += 1

    diff_values.append(0)
    diff_values.append(0)
    return diff_values


def standardDeviationX(array_3d):
    print("entered the standardDeviationX function")
    deviations = []
    for i in range(len(array_3d)):
        sums = []
        for c in array_3d[i].transpose():
            temp = 0
            for r in c:
                if r != 0:
                    temp += r
            sums.append(temp)
        deviations.append(np.std(sums))
    return np.around(deviations, 3)


def percentError(array3d):
    print("entered the percentError function")
    surface_areas = surfaceArea(array3d)
    percent_error_list = []
    for i in range(len(array3d)):
        percent_error_list.append((surface_areas[i]/(105 * 141)) * 100)

    new_percent_error_list = np.around(percent_error_list, 3)

    return new_percent_error_list


def brain_event_or_rest(array_2d):
    is_event = []
    for i in array_2d:
        if i > 0:
            is_event.append(1)
        else:
            is_event.append(0)
    return is_event

    

def reshapeMags(mags, pnts):
    print("enetered the reshapeMags function")

    print('\tReshaping magnitudes\n')

    roimask = None
    n_components = None

    gridx = np.asarray(pnts[0, :, 0, 0])
    gridy = np.asarray(pnts[0, :, 0, 1])

    xshape = np.unique(gridx).shape[0]
    yshape = np.unique(gridy).shape[0]

    if (xshape * yshape) == mags.shape[1]:
        print('x * y is equal to vector length')
        print('\t\txshape: ', xshape)
        print('\t\tyshape: ', yshape)
        print('\t\tmags.shape[1]: ', mags.shape[1])
        shape = (mags.shape[0], yshape, xshape)
        print("Shape of resized matrix: ", shape, type(shape))
    else:
        print('x * y is NOT equal to vector length')
        print('\t\txshape: ', xshape)
        print('\t\tyshape: ', yshape)
        print('\t\tmags.shape[1]: ', mags.shape[1])

    mov = mags.copy().astype('float64')
    mov[np.isnan(mags)] = 0
    mov =  mov.reshape((mov.shape[0], yshape, xshape))

    return mov 


def denoiseMags(mov, percent_noise_cuttoff = 99.99, verbose = True):
    print("entered the denoiseMags functions")

    print('\tDenoising magnitudes\n')

    movtc = np.nanmean(mov, axis=(1,2))
    #find and sort the most to least amount of motion based on the mean across the frame
    movsortedindex = sorted(range(len(movtc)), key=lambda k: movtc[k])[::-1]
    #make a noise matrix from frames that have little to no motion (the bottom 10% of motion of all motion vectors)
    noise = mov[movsortedindex[int(mov.shape[0]*-.1):]]
    #find the nosie cutoff (99.99% quantile of the all motion magnitudes in the lowest 10% of frames)
    cutoff = np.percentile(noise, percent_noise_cuttoff)

    if verbose:
        stdev_check = np.std(mov, 0).reshape((mov.shape[1], mov.shape[2]))
        max_check = np.max(mov, 0).reshape((mov.shape[1], mov.shape[2]))

        print('std', np.min(stdev_check), np.median(stdev_check), np.mean(stdev_check), np.max(stdev_check))
        print('max', np.min(stdev_check), np.median(max_check), np.mean(max_check), np.max(max_check))
        print('\nNoise mean: ', np.nanmean(noise), 'Noise std: ', np.nanstd(noise))
        print('Noise cutoff: ', cutoff)

    #create thresholded mask movie
    binmov = mov.copy().astype('uint8') * 0
    binmov[mov>cutoff] = 1

    sigma = [2, 2] #[y_dim, x_dim]

    #get rid of speckles
    selem = disk(2)
    for i, frame in enumerate(binmov):
        binmov[i] = opening(frame, selem=selem)
    
    #multiply 
    return binmov * mov


def smoothLocalmaxMags(mags_denoise, neighborhood = None):
    print("entered the smoothLocalmaxMags function")

    print('\tSmoothing magnitudes\n')

    if neighborhood == None:
        neighborhood = np.ones((2,2)).astype('uint8')
    
    lmax_mask = np.zeros_like(mags_denoise).astype('uint8')

    sigma = [2, 2] #[y_dim, x_dim]
    mov = mags_denoise.copy()

    for i, frame in enumerate(mags_denoise):
        # mov[i] = convolve(frame, weights, mode='constant')
        mov[i] = gaussian_filter(frame, sigma, mode='constant')
        local_max = maximum_filter(mov[i], footprint=neighborhood)==mov[i]
        background = (mov[i]==0)
        eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
        lmax_mask[i] = local_max ^ eroded_background

    return mov, lmax_mask


# main run
if __name__ == '__main__':

    import argparse
    import time
    import pandas as pd

    # Argument Parsing
    # -----------------------------------------------
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', type = argparse.FileType('r'), 
        nargs = '+', required = False, 
        help = 'path to the processed opticFlow and videodata file(s)')
    ap.add_argument('-f', '--fps', default = 10, required = False,
        help = 'frames per second from recordings')
    ap.add_argument('-t','--train', action='store_true',
        help = 'train the classifier on the newest class_metric dataframe')
    ap.add_argument('-ud', '--updateDF', action='store_true',
        help = 'update full classifier dataframe')
    ap.add_argument('-uc', '--updateClass', action='store_true',
        help = 'update class in experimental dataframe, input ica.hdf5 and ensure metrics.tsv is in the same folder')
    ap.add_argument('-fc', '--force', action='store_true',
        help = 'force re-calculation')
    ap.add_argument('-p', '--plot', action='store_true',
        help= 'Vizualize training outcome')
    args = vars(ap.parse_args())


    #load the raw video
    # raw = wb.loadMovie('170721_07_c1-body_cam.mp4')

    # #mask the angles
    # mask3d = np.zeros_like(mov) * np.nan
    # mask3d[mov>0] = 1
    # angs *= mask3d


    pathlist = [path.name for path in args['input']]
    print('{0} files found:'.format(len(pathlist)))

    for path in pathlist:
        print('\n\n\tWorking on ' + path)

        #find the average movtion vectors
        if path.endswith('videodata.hdf5'):
            print('Input found:')
            print('\t'+path)
            assert os.path.exists(path), 'Videodata file was not found!'
            print('Processing file:', path)
            #
            # assert path.endswith('videodata.hdf5'), "Path did not end in 'videodata.hdf5'"
            print('\nLoading videodata to create classifier metrics\n------------------------------------------------')
            f = h5(path)
            print('\t loading dfof mean')
            dfof = f.load('dfof_mean')

        if path.endswith('OpticFlow.hdf5'):
            assert os.path.exists(path), 'opticFlow file was not found!'
            assert path.endswith('.hdf5'), 'Unknown data type.  Please load .hdf5 only'
            print('Input found:')
            print('\t'+path)

            print('Processing file:', path)
            if path.endswith('.hdf5'):
                # assert path.endswith('opticFlow.hdf5'), "Path did not end in 'opticFlow.hdf5'"
                savepath = path.replace('.hdf5', '_metrics.csv')
                base = os.path.basename(path)

            print('\nLoading optic flow data to create classifier metrics\n------------------------------------------------')
            f = h5(path)
            print('\tloading mags, angs, and pnts')

            mags = f.load('mags')
            angs = f.load('angs')
            pnts = f.load('pnts')

            mov = reshapeMags(mags, pnts)
            mov = denoiseMags(mov, percent_noise_cuttoff = 99.99, verbose = False)
            mov, lmax = smoothLocalmaxMags(mov)

            #reshape the angles movie and rotate the angles 90 degress
            print('\tPreparing angles data')
            shape = mov.shape
            angs = angs.reshape(shape)
            angs -= 90
            angs[angs < 0] += 360
            angs[mov == 0] = np.nan
    


    print('\nCreating dataframe for metrics\n------------------------------------------------')
    #make data frame
    df = pd.DataFrame()

    #load non-looped variables into s
    if 'angs' in globals():
        df['angs.stdev'] = np.nanstd(angs, axis = (1,2))
        df['angs.mean'] = np.nanmean(angs, axis = (1,2))

    if 'mov' in globals():
        df["mov.mean"] = findMeans(mov)
        df["mov.std"] = standardDeviation(mov)
        df["mov.mode"] = findMode(mov)
        df["mov.range"] = findRange(mov)
        df["mov.eventrest"] = findEvent(mov)
        df["mov.maxeventval"] = maxValueOfEvent(mov)
        df["mov.surfarea"] = surfaceArea(mov)
        df["mov.totalmag"] = listOfTotalMagnitude(mov)
        df["mov.firstder"] = findingFirstDerivativePoints(findMeans(mov))
        df["mov.secder"] = findingSecondDerivativePoints(findMeans(mov))
        df["mov.stdx"] = standardDeviationX(mov)
        df["mov.stdy"] = standardDeviationY(mov)
        df["mov.diffxystd"] = comparison(standardDeviationX(mov), standardDeviationY(mov))
        df["mov.diffmaxevents"] = findingDistanceBetweenMaxOfEvent(mov)
        df["mov.percent"] = motionPercentage(mov)
        df["mov.percenterror"] = percentError(mov)
        df["mov.timetoevent"] = timeContinuity(findEvent(mov), forward_or_backward='backward')
        df["move.timefromevent"] = timeContinuity(findEvent(mov))
        df['mov.numlocmax'] = np.nansum(lmax, axis = (1,2))

    if 'dfof' in globals():
        df["brain.data"] = np.around(sameSizeUp(dfof, mov), 3)
        df["brain.eventrest"] = findEvent(sameSizeUp(dfof, mov))
        u_switch, d_switch, n_array = findingRangeValues(dfof)
        df["brain.rangemaxmin"] = sameSizeUp(rangeOfSections(u_switch, d_switch, n_array), mov)
        df["brain.firstder"] = sameSizeUp(findingFirstDerivativePoints(dfof), mov)
        df["brain.secondder"] = sameSizeUp(findingSecondDerivativePoints(dfof), mov)
        df["diff.brainmov"] = comparison(sameSizeUp(dfof, mov), findMeans(mov))
        df["diff.brainmovfirstder"] = comparison(sameSizeUp(findingFirstDerivativePoints(dfof), mov), findMeans(mov))
        df["diff.brainmovsecder"] = comparison(sameSizeUp(findingSecondDerivativePoints(dfof), mov), findMeans(mov))

    
    df.to_csv(savepath)

            # create_movie = False
            # movmax = np.max(mov)
            # print(movmax)
            # movmean = np.mean(mov)
            # movstd = np.std(mov)
            # for i, frame in enumerate(mov):
            #     #find local maxima
            #     fmax = localMaxima2d(frame)
            #     fmaxl = label(fmax)
            #     #create a mask
            #     mask = frame.copy() * 0
            #     mask[frame>0]=1
            #     #seperate all local events
            #     wshed = watershed(-frame, fmaxl, mask=mask)
            #     #numbeer of local maxzima
            #     nfmax = fmaxl.max()
            #     #create precent frame to determine local event direction
            #     percent = frame.copy()
            #     x = [] 
            #     y = []
            #     u = []
            #     v = []
            #     if create_movie:
            #         if i== 0:
            #             a = np.zeros((angs.shape[0], angs.shape[1], angs.shape[2], 3))
            #         frame = wb.rescaleMovie(frame, low=0, high=movmax, verbose=False)
            #         cframe = np.stack((frame[:,:], frame[:,:], frame[:,:]), axis = 2).astype(np.uint8)
            #         cframe = cv2.applyColorMap(cframe, cv2.COLORMAP_HOT)
            #     #get the correct angle frame
            #     angframe = angs[i]
            #     #find coordinates of all local maxima
            #     for region in regionprops(fmaxl):
            #         x.append(region.coords[0][1])
            #         y.append(region.coords[0][0])
            #     #find the weighted angle
            #     for region in np.arange(1,nfmax+1,1):
            #         regionsum = np.nansum(frame[wshed==region])
            #         # print(regionsum)
            #         percent[wshed==region]/=regionsum
            #         percent[wshed==region]*=angframe[wshed==region]
            #         angle = np.nansum(percent[wshed==region])
            #         # print('Angle: ', angle)
            #         try:
            #             #find relative points to local maxima
            #             u = np.cos(np.radians(angle)) * 10
            #             v = np.sin(np.radians(angle)) * 10
            #             # print('Start: ', (int(x[region-1]),int(y[region-1])),'End: ',(int(u),int(v)))
            #             if create_movie:
            #                 cframe = cv2.arrowedLine(cframe, (int(x[region-1]),int(y[region-1])), (int(x[region-1] - u), int(y[region-1] - v)), color = (255,255,255), thickness = 1)
            #         except Exception as e:
            #             print(e)
            #     if create_movie:
            #         a[i] = cframe

                    
            # raw = raw[start_stop[0]:start_stop[1]]
            # print(raw.shape)
            # raw = np.stack((raw, raw, raw), axis = 3).astype(np.uint8)
            # print(raw.shape)
            # pad_size = (raw.shape[1] - a.shape[1])/2

            # if (pad_size%1) == 0:
            #     pad = np.zeros((a.shape[0], pad_size, a.shape[2], a.shape[3]), dtype=np.uint8)
            #     a = np.hstack((pad, a, pad))
            # else:
            #     pad = np.zeros((a.shape[0], int(pad_size + 0.5), a.shape[2], a.shape[3]), dtype=np.uint8)
            #     a = np.hstack((pad, a, pad[:,1:,:,:]))

            # wb.playMovie(np.dstack((a[:1000], raw[:1000])))

