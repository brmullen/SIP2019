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

'''
Functions for classifying behavioral states

Authors: Jimmy Chen, Shreya Mantripragada, Emma Dionne, Brian R. Mullen
Date: 2019-07-03
'''


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


def common_occurences(array_3d):
   dictionary = {}
   for i in array_3d:
       if i in dictionary:
           dictionary[i] += 1
       else:
           dictionary[i] = 1
   return dictionary


def same_size(list_1, list_2):
   #returns the lists with the same size
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
    means = findMeans(array_3d)
    mode = common_occurences(means)
    m = []
    for i in means:
        m.append(mode.get(i))
    return m


def findEvent(array3d):
    means = findMeans(array3d)
    is_event = []
    for i in means:
        if i > 0:
            is_event.append(1)
        else:
            is_event.append(0)
    return is_event


def findRange(array3d):
    return np.around(np.max(array3d, axis = (1,2)), 3)


def same_size_up(small, big):
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


def finding_range_values(array2d): 
    print("entered the finding_range_values function")
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


def range_of_sections(u_switch, d_switch, array2d):
    print("entered the range_of_sections function")
    #u_switch is the list of the indices of all the maxs
    #d_switch is the list of the indices of all the mins
    difference_list = []
    for i in range(len(u_switch)):
        difference_list.append(abs(abs(array2d[u_switch[i]] - array2d[d_switch[i]])))
    return np.around(difference_list, 3)


def max_value_of_event(array_3d):
    #rests_events = []
    print("entered the max_value_of_event function")
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


def list_of_total_magnitude(array3d):
    print("entered the list_of_total_magnitude function")
    total = []
    for i in array3d:
        total.append(np.sum(i))
    return np.around(total, 3)


def surface_area(array3d):
    print("entered the surface_area function")
    area_count = []
    for i in array3d:
        count = 0
        for r in i:
            for c in r:
                if c > 0:
                    count += 1          
        area_count.append(count)
    return area_count


def total_magnitude(array3d):
    total = np.sum(array3d)
    
    return total


def motion_percentage(array_3d):
    total = total_magnitude(array_3d)
    means = findMeans(array_3d)
    percentages = []
    for i in means:
        percentages.append((i/total)*100)
        
    return np.around(percentages, 3)


def time_continuity(timecourse, forward_or_backward='forward'):
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


def finding_first_derivative_points(array_2d): 
    print("entered the finding_derivative_points function")
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


def finding_second_derivative_points(array_2d): 
    print("entered the finding_second_derivative_points function")
    first_derivative = finding_first_derivative_points(array_2d)
    second_derivative = finding_first_derivative_points(first_derivative)
    
    return second_derivative


def comparison(list_1, list_2):
    print("entered the comparison function")
    new_list = []
    for i in range(len(list_1)):
        difference = list_1[i] - list_2[i]
        new_list.append(difference)
    
    new_new_list = np.around(new_list, 3)
    return new_new_list


def standard_deviation_y(array_3d):
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


def finding_distance_between_max_of_event(array_3d):
    print("entered the finding_distance_between_max_of_event function")
    
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
            
    return diff_values


def standard_deviation_x(array_3d):
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


def percent_error(array3d):
    surface_areas = surface_area(array3d)
    percent_error_list = []
    for i in range(len(array3d)):
        percent_error_list.append((surface_areas[i]/(105 * 141)) * 100)
    return percent_error_list
    
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

            dfof = f.load('dfof')
            mov = f.load('mags')
            angs = f.load('rot_angs')
            start_stop = f.load('start_stop_index')
        
            #make data frame

            df = pd.DataFrame()

            #load non-looped variables into s
            df["move_mean"] = findMeans(mov)
            df["move_standard_deviation"] = standardDeviation(mov)
            df["move_mode"] = findMode(mov)
            df["move_range"] = findRange(mov)
            df["move_event_or_rest"] = findEvent(mov)
            df["move_max_value_of_event"] = max_value_of_event(mov)
            df["move_surface_area"] = surface_area(mov)
            df["move_total_magnitude"] = list_of_total_magnitude(mov)
            df["move_first_derivative"] = finding_first_derivative_points(findMeans(mov))
            df["move_second_derivative"] = finding_second_derivative_points(findMeans(mov))
            df["move_standard_deviation_of_x"] = standard_deviation_x(mov)
            df["move_standard_deviation_of_y"] = standard_deviation_y(mov)
            df["move_difference_between_x_and_y_standard_deviation"] = comparison(standard_deviation_x(mov), standard_deviation_y(mov))
            df["move_difference_between_max_of_events"] = finding_distance_between_max_of_event(mov)
            df["move_percentage"] = motion_percentage(mov)
            df["move_percent_error"] = percent_error(mov)
            df["move_time_to_event"] = time_continuity(findEvent(mov), forward_or_backward='backward')
            df["move_time_from_event"] = time_continuity(findEvent(mov))
            df["brain_data"] = np.around(same_size_up(dfof, mov), 3)
            df["brain_event_or_rest"] = findEvent(same_size_up(dfof, mov))
            u_switch, d_switch, n_array = finding_range_values(dfof)
            df["brain_range_between_max_min"] = same_size_up(range_of_sections(u_switch, d_switch, n_array), mov)
            df["brain_first_derivative"] = same_size_up(finding_first_derivative_points(dfof), mov)
            df["brain_second_derivative"] = same_size_up(finding_second_derivative_points(dfof), mov)
            df["difference_between_brain_and_move"] = comparison(same_size_up(dfof, mov), findMeans(mov))
            df["difference_between_brain_and_move_first_derivative"] = comparison(same_size_up(finding_first_derivative_points(dfof), mov), findMeans(mov))
            df["difference_between_brain_and_move_second_derivative"] = comparison(same_size_up(finding_second_derivative_points(dfof), mov), findMeans(mov))
            #u_switch, d_switch = finding_range_values(same_size_up(dfof, mov))
            #df["range_between_max_min"] = range_of_sections(u_switch, d_switch)
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