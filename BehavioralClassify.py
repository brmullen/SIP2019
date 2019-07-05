#!/usr/bin/env python3

'''
Functions for classifying behavioral states

Authors: Jimmy Chen, Shreya Mantripragada, Emma Dione, Brian R. Mullen
Date: 2019-07-03
'''

import numpy as np
import matplotlib.pyplot as plt
from hdf5manager import hdf5manager as h5
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage
import scipy
from skimage.measure import label, regionprops
from scipy.ndimage.filters import gaussian_filter, convolve
from skimage.morphology import disk, watershed
from skimage.morphology import erosion, dilation, opening, closing
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.ndimage.filters import maximum_filter
from opticFlow import opticFlow as of
import colorsys
import wholeBrain as wb
import cv2
import os

# FUNCTIONS

def getAnglemap(xydim=25):
    #creates angle maps
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
    neighborhood = np.ones((5,5))
    local_max = maximum_filter(array_2d, footprint=neighborhood)==array_2d
    background = (array_2d==0)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    lmax = local_max ^ eroded_background
    return lmax

def standardDeviation(array_3d):
    all_deviations = []
    x = []
    for i, frame in enumerate(array_3d):
        mean = np.mean(frame)
        x.append(i)
        deviations = []
        for r in frame:
            for c in range(len(r)):
                deviations.append(((r[c]) - mean) ** 2)
        mean1 = np.mean(deviations)
        current_deviation = mean1 ** (1/2)
        all_deviations.append(current_deviation)
    plt.scatter(x, all_deviations) 
    return all_deviations
    
set_dev = standardDeviation(mov)

# Average magnitude in multiple dimensions
def motionCharacterize(array3d):
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
    for i, st in enumerate(start):
        mag_per_event[st:end[i]] = np.sum(array3d[st:end[i]])/event_frames[i]
    
    #   Magnitude events
#     fig = plt.figure(figsize = (10,5))
#     plt.plot(brain_magnitude, color='k')
#     plt.plot(mag_mean,color='g')
#     plt.plot(threshold,color='r')
#     plt.ylim([0,2])
#     plt.show()
#     plt.plot(mag_per_event)
#     plt.show()
    
    return mag_per_event
mag_data = motionCharacterize(mov)


# frames = [2822, 2825, 2916, 3016, 3384, 3378]

# fig, axs = plt.subplots(len(frames), 2)
# for i, frame in enumerate(frames):
#     f = mov[frame]
#     fmax = localMaxima2d(f)
#     fmaxl = label(fmax)
#     axs[i,0].set_title('Frame {}'.format(frame))
#     axs[i,0].imshow(f)
#     mask = f.copy() * 0
#     mask[f>0]=1
#     wshed = watershed(-f, fmaxl, mask=mask)
#     nfmax = fmaxl.max()
#     percent = f.copy()
#     x = []
#     y = []
#     u = []
#     v = []
#     for region in regionprops(fmaxl):
#         x.append(region.coords[0][1])
#         y.append(region.coords[0][0]) 
#     for region in np.arange(1,nfmax+1,1):
#         regionmean = np.sum(f[wshed==region])
#         percent[wshed==region]/=regionmean
#         percent[wshed==region]*=angs[frame, wshed==region]
#         angle = np.sum(percent[wshed==region])
#         u.append(np.cos(angle))
#         v.append(np.sin(angle))
#     print(x)
#     print(y)
#     print(u+x)
#     print(u+v)
#     axs[i,0].quiver(x,y,u,v, angles='xy', scale_units='xy', scale = 1)
#     axs[i,1].imshow(percent)
#     axs[i,0].axis('off')    
#     axs[i,1].axis('off')
# plt.show()

der = np.zeros_like(dfof)*10
# Brain Activity Boundaries
for i, val in enumerate(dfof):
    if i == 0 or i == dfof.shape[0]:
        continue
    else:
        der[i] = val - dfof[i-1]
der *= 10

d_switch = []
u_switch = []

# Derivative of the graph by points 
for i, val in enumerate(der):
    if i == 0 or i == der.shape[0]:
        continue
    elif (val > 0) and (der[i-1] < 0):
        u_switch.append(i)
    elif (val < 0) and (der[i-1] > 0):
        d_switch.append(i)
        
u_switch = np.array(u_switch)/10
d_switch = np.array(d_switch)/10
  
time = np.arange(dfof.shape[0])/10

fig, axis = plt.subplots(1,figsize = (20,5))

# Line-Scatter Graph
plt.scatter(time[der<0], der[der<0], linewidths =0.005)
plt.scatter(time[der>0], der[der>0])
plt.plot(time, dfof, color = 'k')
for line in u_switch:
    plt.axvline(x = line, color='g')
for line in d_switch:
    plt.axvline(line, color='r')
plt.xlim([0,134])
plt.show()

# Average magnitude in multiple dimensions
def motionCharacterize(array3d):
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
        
# #     Duration intervals
#     frame_durations = []
#     for i in range(len(start)):
#         interval = end[i] - start[i]
#         frame_durations.append(interval)
#     print(frame_durations)
    
    #   Magnitude events
#     fig = plt.figure(figsize = (10,5))
#     plt.plot(brain_magnitude, color='k')
#     plt.plot(mag_mean,color='g')
#     plt.plot(threshold,color='r')
#     plt.ylim([0,2])
#     plt.show()
#     plt.plot(mag_per_event)
#     plt.show()

#     print(frame_durations[1])
#     for i in frame_durations:
#     print(dfDur)

    
    return mag_per_event, duration, rest
dfDur = pd.DataFrame()

dfDur['mag_per_event'], dfDur['duration'], dfDur['rest']= motionCharacterize(mov)
# mag_data = motionCharacterize(mov)
# plt.plot(dfDur['mag_per_event'])

fig = plt.figure(figsize=(10,5))
plt.plot(dfDur['duration'] * 100, label='duration')
plt.plot(dfDur['rest'], label='rest')
plt.legend()
plt.show()

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
        help = 'path to the processed ica file(s)')
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

    # #reshape the angles movie and rotate the angles 90 degress
    # shape = mov.shape
    # angs = angs.reshape(shape)
    # angs -= 90
    # angs[angs < 0] += 360

    # #mask the angles
    # mask3d = np.zeros_like(mov) * np.nan
    # mask3d[mov>0] = 1
    # angs *= mask3d

    #find the average movtion vectors

    if args['input'] != None:
        paths = [path.name for path in args['input']]
        print('Input found:')
        [print('\t'+path) for path in paths]

        for path in paths:
            print('Processing file:', path)
            assert path.endswith('.hdf5'), 'Unknown data type.  Please load .hdf5 only'
            if path.endswith('.hdf5'):
                # assert path.endswith('opticFlow.hdf5'), "Path did not end in 'opticFlow.hdf5'"
                savepath = path.replace('.hdf5', '_metrics.csv')
                base = os.path.basename(path)

            print('\nLoading data to create classifier metrics\n------------------------------------------------')
            f = h5(path)
            f.print()

            mov = f.load('mags')
            angs = f.load('rot_angs')
            start_stop = f.load('start_stop_index')
        
            #make data frame

            df = pd.DataFrame()



            #load non-looped variables into 
            df["move_mean"] = np.mean(mov, axis=(1,2))

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
