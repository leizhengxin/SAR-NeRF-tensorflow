'''
    name:poses_generator.py
    function:gen SAR image poses
    date:2022.3.7
'''
from cmath import acos, cos, pi, sin, sqrt, tan
import os
import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
import scipy.io as sio 
import cv2 as cv
from skimage import transform,data

def rotate_metrix(theta):
    metrix = np.array([[1,0,0],[0,cos(theta),sin(theta)],[0,-sin(theta),cos(theta)]])
    return np.real(metrix)

def load_json_file(json_path,scale=0.25):
    with open(json_path,'r') as load_f:
        data_dict = json.load(load_f)
    
    print(data_dict)

    if "h_r" in data_dict.keys():
        h_r = data_dict["h_r"]
    if "theta" in data_dict.keys():
        theta = data_dict["theta"]
        theta = theta * pi / 180
    if "pixel" in data_dict.keys():
        pixel = data_dict["pixel"]
    if "Na" in data_dict.keys():
        Na = int(data_dict["Na"] * scale)
    if "Nr" in data_dict.keys():
        Nr = int(data_dict["Nr"] * scale)
    
    return h_r,theta,pixel,Na,Nr

def gen_rader_rays(image,h_r,theta,pixel,Na,Nr,perturb,N_theta = 20,scale=1):
    h_unit = np.array([[1],[0],[0]],dtype=np.float)
    v_unit = np.array([[0],[1],[0]],dtype=np.float)
    k_unit = np.array([[0],[0],[1]],dtype=np.float)

    file_name = image.split("/")[-1]
    phi = (float(file_name.split("_")[0]) + 90)%360
    phi = phi * pi / 180

    Pr = np.array([[np.real(tan(theta)*sin(phi))],[1],[np.real(tan(theta)*cos(phi))]],dtype=np.float)
    Pr = h_r * Pr

    #生成雷达矩阵转化世界坐标系矩阵
    Rr = [[-cos(phi),-cos(theta)*sin(phi),-sin(theta)*sin(phi)],
            [    0    ,      sin(theta)    ,    -cos(theta)     ],
            [sin(phi) ,-cos(theta)*cos(phi),-sin(theta)*cos(phi)]]
    Rr = np.array(np.real(Rr),dtype=np.float)

    V_r = np.zeros([3,Na])
    delta_a = pixel / scale
    for i in range(Na):
        v_r = (-(Na+1)/2 + (i+1))*h_unit*delta_a
            
        V_r[:,i] = v_r.T

    # N_theta = 20          #扫角的采样数 
    OD = np.real(tan(theta) * h_r)

    Rmin = np.real(sqrt(h_r*h_r + (OD-Nr/2*delta_a) * (OD-Nr/2*delta_a)))
    Rmax = np.real(sqrt(h_r*h_r + (OD+Nr/2*delta_a) * (OD+Nr/2*delta_a)))
    Rc = np.real(sqrt(h_r * h_r + OD * OD))

    theta_min = np.real(acos(h_r/Rmin))
    theta_max = np.real(acos(h_r/Rmax))

    t_vals = tf.linspace(0., 1., N_theta)
    z_vals = 1./(1./theta_min * (1.-t_vals) + 1./theta_max * (t_vals))

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    D_r = np.zeros([3,N_theta]) #雷达坐标系的方向集合
    D_w = np.zeros([3,N_theta]) #世界坐标系的方向集合
    for k in range(N_theta):
        d_r = np.dot(rotate_metrix(z_vals[k] -theta),k_unit)
        d_w = np.dot(Rr,d_r)
        
        D_r[:,k] = d_r.T
        D_w[:,k] = d_w.T

    V_r = np.array(V_r,dtype=np.float)
        # 空间采样点的表示
    delta_r = (Rmax -Rmin) / Nr
    V_r_sample = np.zeros([3,Na,Nr,N_theta])
    V_w_sample = np.zeros([3,Na,Nr,N_theta])

    for i in range(Na):
        for k in range(N_theta):
            for j in range(Nr):
                v_r_sample = V_r[:,i] +D_r[:,k] * (Rc + (-(Nr+1) / 2 + j + 1) * delta_r)
                v_w_sample = np.dot(Rr,v_r_sample.reshape(3,1)) + Pr
                
                # V_r_sample[:,i,j,k] = v_r_sample
                V_w_sample[:,i,j,k] = v_w_sample.T
    
    #print("test")
    return np.array(V_w_sample,dtype=np.float),np.array(D_w,dtype=np.float)
    
def gen_rader_rays_phi(phi,h_r,theta,pixel,Na,Nr,perturb,N_theta,scale=1):
    h_unit = np.array([[1],[0],[0]],dtype=np.float)
    v_unit = np.array([[0],[1],[0]],dtype=np.float)
    k_unit = np.array([[0],[0],[1]],dtype=np.float)

    # file_name = image.split("/")[-1]
    # phi = (float(file_name.split("_")[0]) + 90)%360
    phi = phi * pi / 180

    Pr = np.array([[np.real(tan(theta)*sin(phi))],[1],[np.real(tan(theta)*cos(phi))]],dtype=np.float)
    Pr = h_r * Pr

    #生成雷达矩阵转化世界坐标系矩阵
    Rr = [[-cos(phi),-cos(theta)*sin(phi),-sin(theta)*sin(phi)],
            [    0    ,      sin(theta)    ,    -cos(theta)     ],
            [sin(phi) ,-cos(theta)*cos(phi),-sin(theta)*cos(phi)]]
    Rr = np.array(np.real(Rr),dtype=np.float)

    V_r = np.zeros([3,Na])
    delta_a = pixel / scale
    for i in range(Na):
        v_r = (-(Na+1)/2 + (i+1))*h_unit*delta_a
            
        V_r[:,i] = v_r.T

    # N_theta = 20          #扫角的采样数 
    OD = np.real(tan(theta) * h_r)

    Rmin = np.real(sqrt(h_r*h_r + (OD-Nr/2*delta_a) * (OD-Nr/2*delta_a)))
    Rmax = np.real(sqrt(h_r*h_r + (OD+Nr/2*delta_a) * (OD+Nr/2*delta_a)))
    Rc = np.real(sqrt(h_r * h_r + OD * OD))

    theta_min = np.real(acos(h_r/Rmin))
    theta_max = np.real(acos(h_r/Rmax))

    t_vals = tf.linspace(0., 1., N_theta)
    z_vals = 1./(1./theta_min * (1.-t_vals) + 1./theta_max * (t_vals))

    if perturb > 0.:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = tf.concat([mids, z_vals[..., -1:]], -1)
        lower = tf.concat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = tf.random.uniform(z_vals.shape)
        z_vals = lower + (upper - lower) * t_rand
    D_r = np.zeros([3,N_theta]) #雷达坐标系的方向集合
    D_w = np.zeros([3,N_theta]) #世界坐标系的方向集合
    for k in range(N_theta):
        d_r = np.dot(rotate_metrix(z_vals[k] -theta),k_unit)
        d_w = np.dot(Rr,d_r)
        
        D_r[:,k] = d_r.T
        D_w[:,k] = d_w.T

    V_r = np.array(V_r,dtype=np.float)
        # 空间采样点的表示
    delta_r = (Rmax -Rmin) / Nr
    V_r_sample = np.zeros([3,Na,Nr,N_theta])
    V_w_sample = np.zeros([3,Na,Nr,N_theta])

    for i in range(Na):
        for k in range(N_theta):
            for j in range(Nr):
                v_r_sample = V_r[:,i] +D_r[:,k] * (Rc + (-(Nr+1) / 2 + j + 1) * delta_r)
                v_w_sample = np.dot(Rr,v_r_sample.reshape(3,1)) + Pr
                
                # V_r_sample[:,i,j,k] = v_r_sample
                V_w_sample[:,i,j,k] = v_w_sample.T
    
    #print("test")
    return np.array(V_w_sample,dtype=np.float),np.array(D_w,dtype=np.float)

def load_mstar_data(path,scale=0.25):
    image_paths = []
    phis = []
    json_path = ""
    for i in os.listdir(path):
        if i.endswith(".tiff"):
            os.rename(os.path.join(path,i),os.path.join(path,os.path.splitext(i)[0]+".tif"))
            i = os.path.splitext(i)[0]+".tif"
            abs_path = os.path.join(path,i)
            image_paths.append(abs_path)
        elif os.path.splitext(i)[1] in [".tif",".jpg",".png"]:
            abs_path = os.path.join(path,i)
            image_paths.append(abs_path)
        elif i.endswith("json"):
            abs_path = os.path.join(path,i)
            json_path = abs_path

    with open(json_path,'r') as load_f:
        data_dict = json.load(load_f)
    
    print(data_dict)

    if "h_r" in data_dict.keys():
        h_r = data_dict["h_r"]
    if "theta" in data_dict.keys():
        theta = data_dict["theta"]
        theta = theta * pi / 180
    if "pixel" in data_dict.keys():
        pixel = data_dict["pixel"]
    if "Na" in data_dict.keys():
        Na = int(data_dict["Na"] * scale)
    if "Nr" in data_dict.keys():
        Nr = int(data_dict["Nr"] * scale)

    #设置雷达坐标系的单位向量
    h_unit = np.array([[1],[0],[0]],dtype=np.float)
    v_unit = np.array([[0],[1],[0]],dtype=np.float)
    k_unit = np.array([[0],[0],[1]],dtype=np.float)

    total_sample = []
    total_images = []
    total_D = []
    for image in image_paths:
        print(image)
        total_images.append(transform.resize(imageio.imread(image),[Nr,Na]))
        start_time = time.time()
        # img = cv.imread(i,1) 
        # cv.imshow("I",img)
        # cv.waitKey(0)
        file_name = image.split("/")[-1]
        phi = (float(file_name.split("_")[0]) + 90)%360
        phis.append(phi)
        # print(file_name,":",phi)
        phi = phi * pi / 180

        #获取雷达坐标系原点坐标
        Pr = np.array([[np.real(tan(theta)*sin(phi))],[1],[np.real(tan(theta)*cos(phi))]],dtype=np.float)
        Pr = h_r * Pr

        #生成雷达矩阵转化世界坐标系矩阵
        Rr = [[-cos(phi),-cos(theta)*sin(phi),-sin(theta)*sin(phi)],
              [    0    ,      sin(theta)    ,    -cos(theta)     ],
              [sin(phi) ,-cos(theta)*cos(phi),-sin(theta)*cos(phi)]]
        Rr = np.array(np.real(Rr),dtype=np.float)

        # 计算雷达坐标系中的射线表征
        # 计算雷达坐标系下的射线原点 
        V_r = np.zeros([3,Na])
        delta_a = pixel
        for i in range(Na):
            v_r = (-(Na+1)/2 + (i+1))*h_unit*delta_a
              
            V_r[:,i] = v_r.T
        
        
        # 计算雷达坐标系下射线的方向
        # 假设雷达在zOx平面上的投影为D
        N_theta = 40          #扫角的采样数 
        OD = np.real(tan(theta) * h_r)

        Rmin = np.real(sqrt(h_r*h_r + (OD-Nr/2*delta_a) * (OD-Nr/2*delta_a)))
        Rmax = np.real(sqrt(h_r*h_r + (OD+Nr/2*delta_a) * (OD+Nr/2*delta_a)))
        Rc = np.real(sqrt(h_r * h_r + OD * OD))

        theta_min = np.real(acos(h_r/Rmin))
        theta_max = np.real(acos(h_r/Rmax))

        theta_s = (theta_max - theta_min)

        delta_theta = theta_s / N_theta#扫角的采样间隔

        k_r_min = np.dot(rotate_metrix(theta_min -theta),k_unit)

        D_r = np.zeros([3,N_theta]) #雷达坐标系的方向集合
        D_w = np.zeros([3,N_theta]) #世界坐标系的方向集合
        for k in range(N_theta):
            d_r = np.dot(rotate_metrix((k+1-0.5)*delta_theta),k_r_min)
            d_w = np.dot(Rr,d_r)
            
            D_r[:,k] = d_r.T
            D_w[:,k] = d_w.T
            

        V_r = np.array(V_r,dtype=np.float)
        # 空间采样点的表示
        delta_r = (Rmax -Rmin) / Nr
        V_r_sample = np.zeros([3,Na,Nr,N_theta])
        V_w_sample = np.zeros([3,Na,Nr,N_theta])

        for i in range(Na):
            for k in range(N_theta):
                for j in range(Nr):
                    v_r_sample = V_r[:,i] +D_r[:,k] * (Rc + (-(Nr+1) / 2 + j + 1) * delta_r)
                    v_w_sample = np.dot(Rr,v_r_sample.reshape(3,1)) + Pr
                    
                    # V_r_sample[:,i,j,k] = v_r_sample
                    V_w_sample[:,i,j,k] = v_w_sample.T

        end_time = time.time()
        total_sample.append(V_w_sample)
        total_D.append(D_w)
        print("spend time: ",(end_time - start_time))
        print("开始雷达坐标系转世界坐标系")
        
    # print(phis)
    # print(image_paths)
    
    return np.array(total_sample,dtype=np.float),np.array(total_D,dtype=np.float),np.array(total_images,dtype=np.float)



