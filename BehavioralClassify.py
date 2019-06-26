#!/usr/bin/env python3

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

#load the cleaned-up magnitude movie
h = h5('test_angs_mags.hdf5')
h.keys()
mov = h.load('mags')
angs = h.load('rot_angs')
start_stop = h.load('start_stop_index')

#load the raw video
raw = wb.loadMovie('170721_07_c1-body_cam.mp4')

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
movmean = np.mean(mov, axis=(1,2))

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

create_movie = True
movmax = np.max(mov)
print(movmax)
movmean = np.mean(mov)
movstd = np.std(mov)
for i, frame in enumerate(mov):
    #find local maxima
    fmax = localMaxima2d(frame)
    fmaxl = label(fmax)
    #create a mask
    mask = frame.copy() * 0
    mask[frame>0]=1
    #seperate all local events
    wshed = watershed(-frame, fmaxl, mask=mask)
    #numbeer of local maxzima
    nfmax = fmaxl.max()
    #create precent frame to determine local event direction
    percent = frame.copy()
    x = [] 
    y = []
    u = []
    v = []
    if create_movie:
        if i== 0:
            a = np.zeros((angs.shape[0], angs.shape[1], angs.shape[2], 3))
        frame = wb.rescaleMovie(frame, low=0, high=movmax, verbose=False)
        cframe = np.stack((frame[:,:], frame[:,:], frame[:,:]), axis = 2).astype(np.uint8)
        cframe = cv2.applyColorMap(cframe, cv2.COLORMAP_HOT)
    #get the correct angle frame
    angframe = angs[i]
    #find coordinates of all local maxima
    for region in regionprops(fmaxl):
        x.append(region.coords[0][1])
        y.append(region.coords[0][0])
    #find the weighted angle
    for region in np.arange(1,nfmax+1,1):
        regionsum = np.nansum(frame[wshed==region])
        # print(regionsum)
        percent[wshed==region]/=regionsum
        percent[wshed==region]*=angframe[wshed==region]
        angle = np.nansum(percent[wshed==region])
        # print('Angle: ', angle)
        try:
            #find relative points to local maxima
            u = np.cos(np.radians(angle)) * 10
            v = np.sin(np.radians(angle)) * 10
            # print('Start: ', (int(x[region-1]),int(y[region-1])),'End: ',(int(u),int(v)))
            if create_movie:
                cframe = cv2.arrowedLine(cframe, (int(x[region-1]),int(y[region-1])), (int(x[region-1] - u), int(y[region-1] - v)), color = (255,255,255), thickness = 1)
        except Exception as e:
            print(e)
    if create_movie:
        a[i] = cframe

        
raw = raw[start_stop[0]:start_stop[1]]
print(raw.shape)
raw = np.stack((raw, raw, raw), axis = 3).astype(np.uint8)
print(raw.shape)
pad_size = (raw.shape[1] - a.shape[1])/2

if (pad_size%1) == 0:
    pad = np.zeros((a.shape[0], pad_size, a.shape[2], a.shape[3]), dtype=np.uint8)
    a = np.hstack((pad, a, pad))
else:
    pad = np.zeros((a.shape[0], int(pad_size + 0.5), a.shape[2], a.shape[3]), dtype=np.uint8)
    a = np.hstack((pad, a, pad[:,1:,:,:]))

wb.playMovie(np.dstack((a[:1000], raw[:1000])))
