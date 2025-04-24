import os, sys
from run_nerf import *
import numpy as np
from tqdm import tqdm
import imageio
import pprint
import tensorflow as tf


import matplotlib.pyplot as plt


(expname,basedir,render_kwargs_train,
    render_kwargs_test, start, grad_vars, models) = get_fn()

net_fn = render_kwargs_test['network_query_fn']

Na = 128
Nr = 128
pixel = 0.3

#计算SAR图像截取的长宽
h = Na * 0.3
w = Nr * 0.3

#计算最小临界矩形
# Lx = np.real(sqrt(2) * h)
# Lz = np.real(sqrt(2) * w)
Lx = 15
Lz = 15
Ly = 10 # 目标的高度应小于10

#计算世界坐标系下的采样点
N = 256
tx = np.linspace(-Lx/2.0, Lx/2.0,N)
tz = np.linspace(-Lz/2.0, Lz/2.0,N)
# ty = np.linspace(0, Ly,int((N * Ly / Lx)))
ty = np.linspace(0, Ly,N)

query_pts = np.stack(np.meshgrid(tx, ty, tz), -1).astype(np.float32)
print(query_pts.shape)
sh = query_pts.shape
flat = query_pts.reshape([-1,3])

def batchify(fn, chunk):
    if chunk is None:
        return fn
    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

fn = lambda i0, i1 : net_fn(flat[i0:i1,None,:], viewdirs=np.expand_dims(np.zeros_like(flat[i0:i1]),axis=1), network_fn=render_kwargs_test['network_fn'])

chunk = 1024*64
raw = np.concatenate([fn(i, i+chunk).numpy() for i in range(0, flat.shape[0], chunk)], 0)
raw = np.reshape(raw, list(sh[:-1]) + [-1])
sigma = np.maximum(raw[...,-1], 0.)
# sigma = tf.exp(-tf.nn.relu(sigma)).numpy()
sigma = (-tf.nn.relu(sigma)).numpy()

print(raw.shape)
# plt.hist(np.exp(-np.maximum(0,sigma.ravel())), log=True)
plt.hist(sigma.ravel(), log=True)
plt.show()
# plt.savefig("hist.png")

## 选择sigma得阈值
import mcubes

threshold = -5
print('fraction occupied', np.mean(sigma > threshold))
vertices, triangles = mcubes.marching_cubes(sigma, threshold)
print('done', vertices.shape, triangles.shape)
print("gen mesh")

## 显示模型
import trimesh

mesh = trimesh.Trimesh(vertices / N - .5, triangles)
mesh.show()