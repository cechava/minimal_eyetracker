#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 2018

@author: cesarechavarria
"""
import matplotlib
#matplotlib.use('Agg')
import cv2
import os
import sys
import optparse
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc,interpolate,stats,signal, spatial, ndimage
import json
import re
import pylab as pl
import seaborn as sns
import pandas as pd
import h5py
sns.set_style("darkgrid", {"axes.facecolor": ".9"})

# ==================================== miscellaneous functions ========================================
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    l.sort(key=alphanum_key)
    
def load_obj(name):
    with open(name, 'r') as r:
        fileinfo = json.load(r)
    return fileinfo
    
def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def save_obj(obj, name ):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def write_to_hdf5_struct(file_grp,path,array):
    if path not in file_grp.keys():
        dset = file_grp.create_dataset(path,array.shape,array.dtype)
        dset[...] = array

# ==================================== MAIN FUNCTIONS ========================================        

def block_mean(im0, fact):
    #function to block-downsample an image by a given factor
    
    im1 = cv2.boxFilter(im0,0, (fact, fact), normalize = 1)
    im2 = cv2.resize(im1,None,fx=1.0/fact, fy=1.0/fact, interpolation = cv2.INTER_CUBIC)
    return im2
    
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


def get_frame_rate(relevant_dir):
    #get frame rate
    
    # READ IN FRAME TIMES FILE
    pfile=open(os.path.join(relevant_dir,'performance.txt'))

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=pfile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]=='frame_rate':
            rate_idx=count
            break
        count = count + 1

    #read just first line
    for line in pfile:
        x=line.split()
        frame_rate = x[rate_idx]
        break
    pfile.close()
    return float(frame_rate)

def get_frame_attribute(relevant_dir,attr_string):
    #get frame-by-frame details
    
    # READ IN FRAME TIMES FILE
    pfile=open(os.path.join(relevant_dir,'frame_times.txt'))

    #READ HEADERS AND FIND RELEVANT COLUMNS
    headers=pfile.readline()
    headers=headers.split()

    count = 0
    while count < len(headers):
        if headers[count]== attr_string:
            sync_idx=count
            break
        count = count + 1

    frame_attr=[]
    # GET DESIRED DATA
    for line in pfile:
        x=line.split()
        frame_attr.append(x[sync_idx])

    frame_attr=np.array(map(float,frame_attr))
    pfile.close()

    return frame_attr


def get_feature_info(process_img, box_pt1, box_pt2, feature_thresh, target_feature='pupil',criterion='area'):
    #get feature info for pupil or corneal reflection
    
    #get grayscale
    if process_img.ndim>2:
        process_img=np.mean(process_img,2)
    #apply restriction box 
    process_img = process_img[box_pt1[1]:box_pt2[1],box_pt1[0]:box_pt2[0]]

    #threshold
    img_roi = np.zeros(process_img.shape)
    if target_feature == 'pupil':
        thresh_array =process_img<feature_thresh
    else:
        thresh_array =process_img>feature_thresh
    if np.sum(thresh_array)>0:#continue if some points have passed threshold
        if criterion == 'area':
            #look for largest area
            labeled, nr_objects = ndimage.label(thresh_array) 
            pix_area = np.zeros((nr_objects,))
            for i in range(nr_objects):
                pix_area[i] = len(np.where(labeled==i+1)[0])
            img_roi[labeled == (np.argmax(pix_area)+1)]=255
            img_roi = img_roi.astype('uint8')
        else:
            #look for region closes to center of box
            x = np.linspace(-1, 1, process_img.shape[1])
            y = np.linspace(-1, 1, process_img.shape[0])
            xv, yv = np.meshgrid(x, y)

            [radius,theta]=cart2pol(xv,yv)

            img_roi = np.zeros(process_img.shape)
            labeled, nr_objects = ndimage.label(thresh_array) 
            pix_distance = np.zeros((nr_objects,))
            for i in range(nr_objects):
                pix_distance[i] = np.min((labeled==i+1)*radius)
            img_roi[labeled == (np.argmin(pix_distance)+1)]=255
            img_roi = img_roi.astype('uint8')

        #get contours
        tmp, contours, hierarchy = cv2.findContours(img_roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        #find contour with most points
        pt_array =np.zeros((len(contours),))
        for count,cnt in enumerate(contours):
            pt_array[count] = len(cnt)
        if len(contours)>0 and np.max(pt_array)>=5:#otherwise ellipse fit will fail
            elp_idx = np.argmax(pt_array)

            #fit ellipse
            elp = cv2.fitEllipse(contours[elp_idx])

            #unpack values
            elp_center = tuple((elp[0][0]+box_pt1[0], elp[0][1]+box_pt1[1]))
            elp_axes = elp[1]
            elp_orientation = elp[2]
            return elp_center, elp_axes, elp_orientation
        else:
            return tuple((0,0)), tuple((0,0)), 0
    else:
        return tuple((0,0)), tuple((0,0)), 0
 

def get_interp_ind(idx, interp_sites, step):
    redo = 0
    interp_idx = idx + step
    if interp_idx in interp_sites:
        redo = 1
    while redo:
        redo = 0
        interp_idx = interp_idx + step
        if interp_idx in interp_sites:
            redo = 1
    return interp_idx
         
def interpolate_sites(var_array, interp_sites):
    array_interp = np.copy(var_array)
    for target_idx in interp_sites:
        if target_idx < var_array.size:
            ind_pre = get_interp_ind(target_idx, interp_sites, -1)
            ind_post = get_interp_ind(target_idx, interp_sites, 1)
        
            x_good = np.array([ind_pre,ind_post])
            y_good = np.hstack((var_array[ind_pre],var_array[ind_post]))
            interpF = interpolate.interp1d(x_good, y_good,1)
            new_value=interpF(target_idx)
        else:
            new_value = var_array[target_idx-1]
       

        array_interp[target_idx]=new_value
    return array_interp

def process_signal(var_array, interp_flag, filt_kernel=11):
    #Clean up singal by interpolating the odd missing values and median filtering
    
    #get sites to interpolate
    interp_list1 = np.where(var_array == 0)[0]
    interp_list2 = np.where(interp_flag== 1)[0]
    interp_sites = np.concatenate((interp_list1,interp_list2))
    interp_sites = np.unique(interp_sites)
    
    #get interpolated list
    var_interp = interpolate_sites(var_array, interp_sites)
    
    #do some medial filtering
    nice_array = signal.medfilt(var_interp, filt_kernel)
    return nice_array

def plot_whole_timecourse(tstamps,value,stim_on_times,blink_times,label,filename):
    
    #figure formatting
    sns.set_style("white")
    sns.set_context("paper",font_scale = 2.5)
    fig,ax=plt.subplots(figsize = (24, 6))

    #mark stimulus onset times
    for f in stim_on_times:
        ax.axvline(x=f, linewidth=1, color='k',alpha = 0.2, linestyle = '--')
        
    
    #plot variable of interest along with blink events
    ax.plot(tstamps,value)
    ymin,ymax = ax.get_ylim()

    if len(blink_times)>0:
                ax.plot(blink_times, np.ones((len(blink_times),))*(ymax+1),'m*')
            
    # Label axes
    ax.set_xlabel('Time (secs)',fontsize = 18,weight = 'bold')
    ax.set_ylabel('%s'%(label),fontsize = 18,weight = 'bold')


    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    sns.despine(trim=True, offset=0, bottom=False, left=False, top = True, right = True,ax = ax)
    
    # Save figure
    fig.savefig(filename,dpi = 300)
    plt.close()

def process_data(options):
    # Get relevant directories an files
    im_dir = os.path.join(options.source_dir,'frames')
    times_dir = os.path.join(options.source_dir,'times')

    # Load stimulus presentation times (for visualization)
    para_file =  [f for f in os.listdir(times_dir) if f.endswith('.json')][0]#assuming a single file for all tiffs in run

    # Get relevant image list
    im_list = [name for name in os.listdir(im_dir) if os.path.isfile(os.path.join(im_dir, name))]
    sort_nicely(im_list)
    im0 = cv2.imread(os.path.join(im_dir,im_list[0]))

    # Load user-specificed restriciton box
    user_rect = load_obj(os.path.join(options.source_dir,'user_restriction_box.json'))

    # Make output directories
    output_file_dir = os.path.join(options.output_dir,'files')
    if not os.path.exists(output_file_dir):
        os.makedirs(output_file_dir)
        
    output_plot_dir = os.path.join(options.output_dir,'plots/complete_timecourse')
    if not os.path.exists(output_plot_dir):
        os.makedirs(output_plot_dir)

    if options.make_movie:
        output_mov_dir = os.path.join(options.output_dir,'movie')
        if not os.path.exists(output_mov_dir):
            os.makedirs(output_mov_dir)

        tmp_dir = os.path.join(output_mov_dir, 'tmp')
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir)

    # Unpack values for pupil and corneal reflection (if defined)
    if 'pupil' in user_rect:
    
        # unpack values
        pupil_x1_orig = int(min([user_rect['pupil'][0][0],user_rect['pupil'][1][0]]))
        pupil_x2_orig = int(max([user_rect['pupil'][0][0],user_rect['pupil'][1][0]]))
        pupil_y1_orig = int(min([user_rect['pupil'][0][1],user_rect['pupil'][1][1]]))
        pupil_y2_orig = int(max([user_rect['pupil'][0][1],user_rect['pupil'][1][1]])) 

        pupil_x1 = pupil_x1_orig
        pupil_y1 = pupil_y1_orig
        pupil_x2 = pupil_x2_orig
        pupil_y2 = pupil_y2_orig

        # use user-defined-threshold otherwise use-average pixel value within user-defined box
        if 'pupil_thresh' in user_rect:
            pupil_thresh = int(user_rect['pupil_thresh'])
        else:
            pupil_thresh = np.mean(im0[pupil_y1:pupil_y2,pupil_x1:pupil_x2])
        print 'threshold value for pupil: %10.4f'%(pupil_thresh)

        # Create empty arrays. 
        # keeping track of center, length of axes, orientation, and missing value flag for pupil and corneal reflection
        pupil_center_list = np.zeros((len(im_list),2))
        pupil_axes_list = np.zeros((len(im_list),2))
        pupil_orientation_list = np.zeros((len(im_list),))
        pupil_flag_event = np.zeros((len(im_list),))

    if 'cr' in user_rect:
        
        # unpack values
        cr_x1_orig = int(min([user_rect['cr'][0][0],user_rect['cr'][1][0]]))
        cr_x2_orig = int(max([user_rect['cr'][0][0],user_rect['cr'][1][0]]))
        cr_y1_orig = int(min([user_rect['cr'][0][1],user_rect['cr'][1][1]]))
        cr_y2_orig = int(max([user_rect['cr'][0][1],user_rect['cr'][1][1]])) 

        cr_x1 = cr_x1_orig
        cr_y1 = cr_y1_orig
        cr_x2 = cr_x2_orig
        cr_y2 = cr_y2_orig
        
        # use user-defined-threshold otherwise use-average pixel value within user-defined box
        if 'cr_thresh' in user_rect:
            cr_thresh = int(user_rect['cr_thresh'])
        else:
            cr_thresh = np.mean(im0[cr_y1:cr_y2,cr_x1:cr_x2])
            
        print 'threshold value for corneal reflection: %10.4f'%(cr_thresh)

        # Create empty arrays. 
        # keeping track of center, length of axes, orientation, and missing value flag for pupil and corneal reflection
        cr_center_list = np.zeros((len(im_list),2))
        cr_axes_list = np.zeros((len(im_list),2))
        cr_orientation_list = np.zeros((len(im_list),))
        cr_flag_event = np.zeros((len(im_list),))

    # Process all images in folder
    for im_count in range(len(im_list)):
        # display count to keep track of progress
        if im_count%1000==0:
            print 'Processing Image %d of %d....' %(im_count,len(im_list))

        # load image
        im0 = cv2.imread(os.path.join(im_dir,im_list[im_count]))
        im_disp = np.copy(im0)#clone for drawing on, later
        
        if options.space_filt_size is not None:
            # smooth image
            im0= cv2.boxFilter(im0,0, (int(options.space_filt_size), int(options.space_filt_size)), normalize = 1)

        if 'pupil' in user_rect:
            # get pupil info
            pupil_center, pupil_axes, pupil_orientation = get_feature_info(im0, (pupil_x1,pupil_y1), (pupil_x2,pupil_y2),\
                                                                           pupil_thresh, 'pupil')

            pupil_ratio = np.true_divide(pupil_axes[0],pupil_axes[1])
            
            if pupil_center[0]==0 or pupil_ratio <=.6 or pupil_ratio>(1.0/.6):#flag this frame, probably blinking
                pupil_flag_event[im_count] = 1


            # draw bounding box and ellipse
            ellipse_params = tuple((pupil_center,pupil_axes,pupil_orientation))
            if options.make_movie:
                cv2.rectangle(im_disp,(pupil_x1,pupil_y1),(pupil_x2,pupil_y2),(0,255,255),1)
                if pupil_flag_event[im_count] == 0:
                    cv2.ellipse(im_disp, ellipse_params,(0,0,255),1)
                else:
                    cv2.ellipse(im_disp, ellipse_params,(0,255,0),1)
               
                
            if pupil_flag_event[im_count] == 0:#not blinking
                 # Re-fit bounding box with some extra room
                dummy_img = np.zeros(im0.shape)
                cv2.ellipse(dummy_img, ellipse_params,255,-1)
                dummy_img = np.mean(dummy_img,2)
                tmp, contours, hierarchy = cv2.findContours(dummy_img.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv2.boundingRect(contours[0])
                pupil_x1 = int(x-3)
                pupil_y1 = int(y-3)
                pupil_x2 = int(x+w+3)
                pupil_y2 = int(y+h+3)

                # save to array
                pupil_center_list[im_count,:] = pupil_center
                pupil_axes_list[im_count,:] = pupil_axes
                pupil_orientation_list[im_count] = pupil_orientation

            # use original user-provided bouding box if we have multiple timepoints with inability to fit ellipses
            if im_count >10:
                if sum(pupil_flag_event[im_count-10:im_count])>= 5:
                    # back to beginning with latest size
                    pupil_x1 = int(pupil_x1_orig-10)
                    pupil_y1 = int(pupil_y1_orig-10)
                    pupil_x2 = int(pupil_x2_orig+10)
                    pupil_y2 = int(pupil_y2_orig+10)

                else:
                    # give yourself room for error after event
                    if pupil_flag_event[im_count-1]<1:
                        pupil_x1 = int(pupil_x1-5)
                        pupil_y1 = int(pupil_y1-5)
                        pupil_x2 = int(pupil_x2+5)
                        pupil_y2 = int(pupil_y2+5)
                        
        if 'cr' in user_rect:
            #get corneal reflection info
            cr_center, cr_axes, cr_orientation = get_feature_info(im0, (cr_x1,cr_y1), (cr_x2,cr_y2),\
                                                                           cr_thresh, 'cr')

            cr_ratio = np.true_divide(cr_axes[0],cr_axes[1])
            cr_radius = np.mean(cr_axes)
            if cr_center[0]==0 or cr_radius>=20:#flag this frame, probably blinking
                cr_flag_event[im_count] = 1

             # draw bounding box and ellipse
            ellipse_params = tuple((cr_center,cr_axes,cr_orientation))
            if options.make_movie:
                cv2.rectangle(im_disp,(cr_x1,cr_y1),(cr_x2,cr_y2),(255,255,0),1)
                if cr_flag_event[im_count] == 0:
                    cv2.ellipse(im_disp, ellipse_params,(255,0,0),1)
                else:
                    cv2.ellipse(im_disp, ellipse_params,(0,255,0),1)

            if cr_flag_event[im_count] == 0: # Re-fit bounding box with some extra room
                # Re-fit bounding box with some extra room
                dummy_img = np.zeros(im0.shape)
                cv2.ellipse(dummy_img, ellipse_params,255,-1)
                dummy_img = np.mean(dummy_img,2)
                tmp, contours, hierarchy = cv2.findContours(dummy_img.astype('uint8'),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                x,y,w,h = cv2.boundingRect(contours[0])
                cr_x1 = int(x-2)
                cr_y1 = int(y-2)
                cr_x2 = int(x+w+2)
                cr_y2 = int(y+h+2)

                # save to array
                cr_center_list[im_count,:] = cr_center
                cr_axes_list[im_count,:] = cr_axes
                cr_orientation_list[im_count] = cr_orientation
                
            # use original user-provided bouding box if we have multiple timepoints with inability to fit ellipses
            if im_count >10:    
                if sum(cr_flag_event[im_count-10:im_count])>= 5:
                    # back to beginning with latest size
                    cr_x1 = int(cr_x1_orig-4)
                    cr_y1 = int(cr_y1_orig-4)
                    cr_x2 = int(cr_x2_orig+4)
                    cr_y2 = int(cr_y2_orig+4)


                else:
                    # give yourself room for error after event
                    if cr_flag_event[im_count-1]<1:
                        cr_x1 = int(cr_x1-2)
                        cr_y1 = int(cr_y1-2)
                        cr_x2 = int(cr_x2+2)
                        cr_y2 = int(cr_y2+2)

        if options.make_movie:
            #write annotated image to file
            cv2.imwrite((os.path.join(tmp_dir,im_list[im_count])), im_disp)

    # Get camera timestamps
    frame_rate = get_frame_rate(times_dir)
    frame_period = 1.0/frame_rate
    frame_idx = get_frame_attribute(times_dir,'frame_number')
    # Assume a steady frame rate
    camera_time = np.arange(0,frame_period*len(frame_idx),frame_period)
    # Correct for unmatched vector lengths
    if camera_time.shape[0]>frame_idx.shape[0]:
        camera_time = np.delete(camera_time,-1)

    if options.make_movie:
        # Convert images to mp4 movie
        video_file = os.path.join(output_mov_dir,'annotated_movie')
        cmd = 'ffmpeg -y -r %10.4f -i %s/%s.png -vcodec libx264 -f mp4 -pix_fmt yuv420p %s.mp4'%(frame_rate,tmp_dir,'%d',video_file)
        os.system(cmd)

        shutil.rmtree(tmp_dir)

    # Get stimulus onset times
    print 'Getting paradigm info from: %s'%(os.path.join(times_dir, para_file))
    with open(os.path.join(times_dir, para_file), 'r') as f:
        trial_info = json.load(f)

    stim_on_times = np.zeros((len(trial_info)))
    stim_off_times = np.zeros((len(trial_info)))

    for ntrial in range(len((trial_info))):
        trial_string = 'trial%05d'%(ntrial+1)
        stim_on_times[ntrial]=trial_info[trial_string]['stim_on_times']/(1E3*60)#convert to minutes
        stim_off_times[ntrial]=trial_info[trial_string]['stim_off_times']/(1E3*60)#convert to minutes

    # Process extracted metrics
    print 'Processing traces for extracted features'
    # get blink event times
    if 'pupil' in user_rect and 'cr' in user_rect:
        blink_flag_event = np.logical_and(pupil_flag_event,cr_flag_event)
    else:
        blink_flag_event = pupil_flag_event
    blink_events = np.where(blink_flag_event==1)[0]#get indices of probable blink events    
    if len(blink_events>0):
        blink_times = camera_time[blink_events]/60.0
    else:
        blink_times = []

    if 'pupil' in user_rect:

        #pixel radius
        tmp = np.mean(pupil_axes_list,1)#collapse across ellipse axes
        pupil_radius = process_signal(tmp, pupil_flag_event, int(options.time_filt_size))


        #pupil aspect ratio
        tmp = np.true_divide(pupil_axes_list[:,0],pupil_axes_list[:,1])#collapse across ellipse axes
        pupil_aspect = process_signal(tmp, pupil_flag_event, int(options.time_filt_size))

        #pupil orientation
        pupil_orientation = process_signal(pupil_orientation_list[:], pupil_flag_event, int(options.time_filt_size))

        # pupil position X 
        pupil_pos_x = np.squeeze(process_signal(pupil_center_list[:,1], pupil_flag_event, int(options.time_filt_size)))

        # pupil position Y
        pupil_pos_y = np.squeeze(process_signal(pupil_center_list[:,0], pupil_flag_event, int(options.time_filt_size)))

        #distance from first frame
        tmp_pos = np.transpose(np.vstack((pupil_pos_x,pupil_pos_y)))
        pupil_dist = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))

    if 'cr' in user_rect:

        #pixel radius
        tmp = np.mean(cr_axes_list,1)#collapse across ellipse axes
        cr_radius = process_signal(tmp, cr_flag_event, int(options.time_filt_size))

        #cr aspect ratio
        tmp = np.true_divide(cr_axes_list[:,0],cr_axes_list[:,1])#collapse across ellipse axes
        cr_aspect = process_signal(tmp, cr_flag_event, int(options.time_filt_size))

        #cr orientation
        cr_orientation = process_signal(cr_orientation_list[:], cr_flag_event, int(options.time_filt_size))

        # cr position X 
        cr_pos_x = np.squeeze(process_signal(cr_center_list[:,1], cr_flag_event, int(options.time_filt_size)))

        # cr position Y
        tmp = cr_center_list[:,0].copy()
        cr_pos_y = np.squeeze(process_signal(cr_center_list[:,0], cr_flag_event, int(options.time_filt_size)))

        #distance from 1st frame
        tmp_pos = np.transpose(np.vstack((cr_pos_x,cr_pos_y)))
        cr_dist = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))

    #use corneal reflection postion to correct pupil location
    if 'pupil' in user_rect and 'cr' in user_rect:

        #relative pupil-position-x
        pupil_pos_x_rel = np.squeeze(process_signal( pupil_center_list[:,1]-cr_center_list[:,1], blink_flag_event, int(options.time_filt_size)))

        #relative pupil-position-x
        pupil_pos_y_rel = np.squeeze(process_signal( pupil_center_list[:,0]-cr_center_list[:,0], blink_flag_event, int(options.time_filt_size)))

        # pupil - distance- relative to CR
        tmp_pos = np.transpose(np.vstack((pupil_pos_x_rel,pupil_pos_y_rel)))
        pupil_dist_rel = np.squeeze(spatial.distance.cdist(tmp_pos,tmp_pos[0:1,:]))
        
        # pupil - motion - relative to CR
        tmp = np.diff(pupil_dist_rel)
        pupil_motion_rel = np.hstack((0,tmp)) # pad with 0 to keep same size of array

    # Output plots with timecourse for some key metrics
    out_fn = os.path.join(output_plot_dir,'pupil_radius_vs_time.png')
    plot_whole_timecourse(np.true_divide(camera_time,60),pupil_radius,stim_on_times,blink_times,'Pupil Radius',out_fn)

    out_fn = os.path.join(output_plot_dir,'pupil_position_vs_time.png')
    plot_whole_timecourse(np.true_divide(camera_time,60),pupil_dist_rel,stim_on_times,blink_times,'Pupil Position (Relative to CR)',out_fn)

    out_fn = os.path.join(output_plot_dir,'pupil_motion_vs_time.png')
    plot_whole_timecourse(np.true_divide(camera_time,60),pupil_motion_rel,stim_on_times,blink_times,'Pupil Motion (Relative to CR)',out_fn)

    # Save all metrics to file
    output_fn = os.path.join(output_file_dir,'full_session_eyetracker_data_sample_data.h5')
    print 'Saving feature info for the whole session in :%s'%(output_fn)

    file_grp = h5py.File(output_fn, 'w')#open file
    #save some general attributes
    file_grp.attrs['options.source_dir'] = im_dir
    file_grp.attrs['nframes'] = len(im_list)
    file_grp.attrs['frame_rate'] = frame_rate
    file_grp.attrs['time_filter_size'] = options.time_filt_size

    # Define and store relevant metrics
    write_to_hdf5_struct(file_grp,'camera_time',camera_time)
    write_to_hdf5_struct(file_grp,'blink_events',blink_flag_event)

    if len(blink_times)>0:
        write_to_hdf5_struct(file_grp,'blink_times',blink_times)
        
    if 'pupil' in user_rect:
        # write to file metrics related to pupil
        write_to_hdf5_struct(file_grp,'pupil_events',pupil_flag_event)
        write_to_hdf5_struct(file_grp,'pupil_radius',pupil_radius)
        write_to_hdf5_struct(file_grp,'pupil_orientation',pupil_orientation) 
        write_to_hdf5_struct(file_grp,'pupil_aspect_ratio',pupil_aspect)
        write_to_hdf5_struct(file_grp,'pupil_distance',pupil_dist)
        write_to_hdf5_struct(file_grp,'pupil_x',pupil_pos_x)
        write_to_hdf5_struct(file_grp,'pupil_y',pupil_pos_y)

    if 'cr' in user_rect:
        # write to file metrics related to corneal reflection
        write_to_hdf5_struct(file_grp,'cr_events',cr_flag_event)
        write_to_hdf5_struct(file_grp,'cr_radius',cr_radius)
        write_to_hdf5_struct(file_grp,'cr_orientation',cr_orientation) 
        write_to_hdf5_struct(file_grp,'cr_aspect_ratio',cr_aspect)
        write_to_hdf5_struct(file_grp,'cr_distance',cr_dist)
        write_to_hdf5_struct(file_grp,'cr_x',cr_pos_x)
        write_to_hdf5_struct(file_grp,'cr_y',cr_pos_y)
    if 'pupil' in user_rect and 'cr' in user_rect:
        # write to file metrics related to pupil distance relative to corneal reflection
        write_to_hdf5_struct(file_grp,'pupil_x_rel',pupil_pos_x_rel)
        write_to_hdf5_struct(file_grp,'pupil_y_rel',pupil_pos_y_rel)
        write_to_hdf5_struct(file_grp,'pupil_dist_rel',pupil_dist_rel)
        write_to_hdf5_struct(file_grp,'pupil_motion_rel',pupil_motion_rel)
        
    file_grp.close()

def extract_options(options):

    parser = optparse.OptionParser()

    # PATH opts:
    parser.add_option('-S', '--source', action='store', dest='source_dir', default='../sample_data', help='source directory containing data subfolders')
    parser.add_option('-O', '--output', action='store', dest='output_dir', default='../sample_output', help='main output directory')

    parser.add_option('-m', '--movie', action='store_true', dest='make_movie', default=True, help='Boolean to indicate whether to make anotated movie of frames')

    parser.add_option('-f', '--smooth', action='store', dest='space_filt_size', default=5, help='size of box filter for smoothing images(integer)')
    parser.add_option('-t', '--timefilt', action='store', dest='time_filt_size', default=11, help='Size of median filter to smooth signals over time(integer)')

    parser.add_option('-b', '--baseline', action='store', dest='baseline', default=1, help='Length of baseline period (secs) for trial parsing')


    (options, args) = parser.parse_args(options)


    return options

#-----------------------------------------------------
#           MAIN SET OF ACTIONS
#-----------------------------------------------------

def main(options):

    options = extract_options(options)
    print 'Processing raw frames'
    process_data(options)


if __name__ == '__main__':
    main(sys.argv[1:])



