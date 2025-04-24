# key code used to train SAR-NeRF net
# 2022/3/14
# leizx

# 在v1版本中，实现了sar-nerf的训练流程
# 包括sar的成像模型模型，网络的构建
# 以及SAR的体素渲染算法
# 在v2版本中，对已有的训练结构进行改进
# 在生成射线的过程中，仅获取雷达的姿态
# 将采样点的获取以及采样的方向获取放入
# 训练中
# 同时，对采样点的获取加入扰动
# 在v3版本中加入预训练的功能
# 并加入根据角度进行渲染的功能



import os

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import sys
import tensorflow as tf
import numpy as np
import imageio
import json
import random
import time
import scipy.io as sio 
from tqdm import tqdm
from poses_generator import *
from run_nerf_helper import *
from skimage import transform,data 
# from renderer import *

tf.compat.v1.enable_eager_execution()

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches."""
    if chunk is None:
        return fn

    def ret(inputs):
        return tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'."""

    inputs_flat = tf.reshape(inputs, [-1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    if viewdirs is not None:
        # input_dirs = tf.broadcast_to(viewdirs[:, None], inputs.shape)
        input_dirs = tf.broadcast_to(viewdirs, inputs.shape)
        input_dirs_flat = tf.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        # input_dirs_flat = tf.reshape(viewdirs, [-1, viewdirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = tf.concat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = tf.reshape(outputs_flat, list(
        inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def render_rays(pts, 
                viewdirs,
                network_query_fn,
                network_fn,
                N_samples,
                use_viewdirs,
                perturb=0.,
                raw_noise_std=0.):
    def raw2outputs(raw):
        # Extract scattering intensity of each sample position along each ray.
        S = tf.math.sigmoid(raw[..., 0]) # 获取散射强度

        noise = 0.
        if raw_noise_std > 0.:
            noise = tf.random.normal(raw[..., 0].shape) * raw_noise_std
        
        # 限制alpha的值大于0,alpha为衰减系数
        alpha =  -tf.nn.relu(raw[...,1] + noise)
        
        I = tf.zeros(alpha.shape[0:2],dtype=tf.float32)

        for k in range(alpha.shape[2]):
            S_k = S[:,:,k]
            tri = tf.linalg.band_part(tf.ones(alpha.shape[0:2],dtype=tf.float32),0,-1)
            E_k = tf.exp(tf.matmul(alpha[:,:,k],tri))
            I_k = S_k * E_k
            I = I + I_k

        # 背景较暗的时候使用
        # for k in range(alpha.shape[2]):
        #     S_k = S[:,:,k]
        #     tri = tf.linalg.band_part(tf.ones(alpha.shape[0:2],dtype=tf.float32),0,-1)
        #     E_k = tf.math.tanh(-alpha[:,:,k]) * tf.exp(tf.matmul(alpha[:,:,k],tri))
        #     I_k = S_k * E_k
        #     I = I + I_k

        
        # s_map = np.zeros(ii.shape[0:2],dtype=np.float) #获取能量矩阵
        # for i in range(ii.shape[0]):
        #     for j in range(ii.shape[1]):
        #         i_list = ii[i,j,:]  #散射强度
        #         E_list = np.zeros(ii.shape[2]) #衰减系数
        #         for k in range(ii.shape[2]):
        #             alpha_list = alpha[i,:j,k]
        #             E = tf.exp(tf.reduce_sum(alpha_list))
        #             E_list[k] = E.numpy()
                
        #         E_list = tf.cast(E_list,tf.float32)
        #         S_list = i_list * E_list
        #         s_map[i,j] = tf.reduce_sum(S_list).numpy()
        return I
    raw = network_query_fn(pts, viewdirs, network_fn)
    I = raw2outputs(raw)
    #print("test")
    return I


def render(sample,d,**kwargs):
    # 对d进行归一化
    viewdirs = d
    viewdirs = viewdirs / tf.linalg.norm(viewdirs, axis=-1, keepdims=True)

    pts = sample

    pts = tf.cast(pts,dtype=tf.float32)
    viewdirs = tf.cast(viewdirs,dtype=tf.float32)

    I = render_rays(pts,viewdirs,**kwargs)
    return I

def render_phi(phi,h_r,theta,pixel,Na,Nr,N_theta,scale=1,**kwargs):
    sample,d = gen_rader_rays_phi(phi, h_r, theta, pixel, Na, Nr,0, N_theta,scale)
    
    sample = tf.transpose(sample,[1,2,3,0])
    d = tf.transpose(d,[1,0])
    s_map = render(sample,d,**kwargs)

    gen_image = to8b(flip180(s_map.numpy()))
    return gen_image

# def create_nerf(multires=10,i_embed=0,use_viewdirs=True,multires_views=4
#                         ,netdepth=8, netwidth=256,netchunk=65536
#                         ,N_samples=64,perturb=True,raw_noise_std=1.0):
def create_nerf(basedir,expname,args):
    # 生成空间点位置的频率编码
    # embed_fn代表频率编码，input_ch代表MLP结构的输入
    embed_fn, input_ch = get_embedder(args.multires, args.i_embed) 
    
    # 生成空间点方向的频率编码
    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(   #input_ch_views 视角编码后的输出维度 
            args.multires_views, args.i_embed)
    
    output_ch = 4
    skips = [4]
    model = init_nerf_model(
        D=args.netdepth, W=args.netwidth,
        input_ch=input_ch, output_ch=output_ch, skips=skips,
        input_ch_views=input_ch_views, use_viewdirs=args.use_viewdirs)
    grad_vars = model.trainable_variables
    models = {'model': model}


    def network_query_fn(inputs, viewdirs, network_fn): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk)

    render_kwargs_train = {
        'network_query_fn': network_query_fn,
        'perturb': args.perturb,
        'N_samples': args.N_samples,
        'network_fn': model,
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
    }

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.

    
    #预训练
    start = 0
    
    ckpts_path = os.path.join(basedir, expname,'checkout')
    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else:
        ckpts = [os.path.join(ckpts_path, f) for f in sorted(os.listdir(ckpts_path)) if
                 ('model_' in f and 'optimizer' not in f)]
    
    print('Found ckpts', ckpts)
    if len(ckpts) > 0 and not args.no_reload:
        ft_weights = ckpts[-1]
        print('Reloading from', ft_weights)
        start = int(ft_weights[-10:-4]) + 1
        if start != 0:
            model.set_weights(np.load(ft_weights, allow_pickle=True))
            
            print('Resetting step to', start)

    return render_kwargs_train, render_kwargs_test, start, grad_vars, models
    

def config_parser():
    # 训练网络使用得参数，后期补充
    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--random_seed", type=int, default=None,
                        help='fix random seed for repeatability')
    parser.add_argument("--dataset_type", type=str, default='mstar',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    # parser.add_argument("--use_viewdirs", action='store_true',
    #                     help='use full 5D input instead of 3D')
    parser.add_argument("--use_viewdirs", type=bool,default=True,
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')

    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--datadir", type=str,
                        default='./dataset/angle_10_train', help='input data directory')
    parser.add_argument("--lrate", type=float,
                        default=5e-4, help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250,
                        help='exponential learning rate decay (in 1000s)')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000,
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')
    parser.add_argument("--N_theta",   type=int, default=20,
                        help='frequency of render_poses video saving')
    parser.add_argument("--scale", type=float, default=0.25,
                        help='resize scale')
    return parser

def train():
    # 训练网络
    
    parser = config_parser()
    args = parser.parse_args()

    scale = args.scale

    #解析basedir expname ,并创建输出文件夹
    print("常见输出的文件夹")
    if len(args.datadir.split("/")[-1]) != 0:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-1])
        expname = args.datadir.split("/")[-1]
    else:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-2])
        expname = args.datadir.split("/")[-2]
    
    path_out_checkout = os.path.join(path_out,"checkout")
    path_out_viz = os.path.join(path_out,"viz")
    mkdir("./out")
    mkdir(path_out)
    mkdir(path_out_checkout)
    mkdir(path_out_viz)
    basedir = "./out"

    
    # 随机数种子随机，每次结果会不一样
    if args.random_seed is not None:
        print('Fixing random seed', args.random_seed)
        np.random.seed(args.random_seed)
        tf.compat.v1.set_random_seed(args.random_seed)

    # Load data
    # 该代码使用得数据集为mstar
    if args.dataset_type == 'mstar':
        # total_sample,total_D,total_image = load_mstar_data(args.datadir)
        h_r,theta,pixel,Na,Nr = load_json_file(os.path.join(args.datadir,"params.json"),scale)
        image_paths = [os.path.join(args.datadir,image_path) for image_path in os.listdir(args.datadir) if os.path.splitext(image_path)[1] in [".tif",".jpg",".png"]]
    else:
        print("Unknown dataset type")


    # Create log dir and copy the config file

    # Create nerf model
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        basedir,expname,args)

    print("完成建立模型")

    # Create optimizer
    lrate = args.lrate
    if args.lrate_decay > 0:
        lrate = tf.keras.optimizers.schedules.ExponentialDecay(lrate,
                                                               decay_steps=args.lrate_decay * 1000, decay_rate=0.1)
    optimizer = tf.keras.optimizers.Adam(lrate)
    models['optimizer'] = optimizer

    global_step = tf.compat.v1.train.get_or_create_global_step()
    global_step.assign(start)

    # 数据集处理，在NeRF中可以将射线拆分为独立个体由于训练
    # 但是在sar中不行，因为SAR成像一个像素点的值应当为不同扫角下的积分
    # 因此训练的时候将一副图像为最小对象

    # i_train = [i for i in np.arange(len(image_paths)) if(i%4  != 0)]
    # i_test = [i for i in np.arange(len(image_paths)) if(i%4  == 0)]
    i_train = [i for i in np.arange(len(image_paths))]
    i_test = [i for i in np.arange(len(image_paths))]
    i_val = i_test
    N_iters = 1000000

    print('Begin')
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    
    # expname = args.datadir.split("/")[-1]

    # Summary writers
    writer = tf.summary.create_file_writer(
        os.path.join(basedir, 'summaries', expname))
    # writer = tf.contrib.summary.create_file_writer(
    #     os.path.join(basedir, 'summaries', expname))
    writer.set_as_default()

    for i in range(start,N_iters):
        time0 = time.time()

        # image_index = i % len(i_train)
        # image_index = i_train[image_index]
        
        # # 获取采样点、方向以及目标
        # image_path = image_paths[image_index]

        img_i = np.random.choice(i_train)
        image_path = image_paths[img_i]
        sample,d = gen_rader_rays(image_path,h_r,theta,pixel,Na,Nr,args.perturb,args.N_theta,scale)
        image = transform.resize(imageio.imread(image_path),[Nr,Na])


        # sample = total_sample[image_index]
        # d = total_D[image_index]
        # image = total_image[image_index]

        sample = tf.transpose(sample,[1,2,3,0])
        d = tf.transpose(d,[1,0])
        image = np.expand_dims(image,axis=-1)

        # d = np.broadcast_to(d,np.shape(sample))


        # batch_rays = tf.concat([sample,d],axis=-1)
        
        target_s = tf.cast(flip180(image),dtype=tf.float32)

        #####  Core optimization loop  #####
        with tf.GradientTape() as tape:
            
            # 预测损失系数和散射强度
            s_map = render(sample,d,**render_kwargs_train)

            # 计算 MSE 损失（使用真实图像和预测图像）
            img_loss = img2mse(s_map,target_s)
            loss = img_loss
            psnr = mse2psnr(img_loss)
            
        
        
        gradients = tape.gradient(loss, grad_vars)
        optimizer.apply_gradients(zip(gradients, grad_vars))

        dt = time.time()-time0

        ###                   训练部分结束                  ####
        # Rest is logging
        def save_weights(net, prefix, i):
            path = os.path.join(
                basedir, expname,'checkout', '{}_{:06d}.npy'.format(prefix, i))
            np.save(path, net.get_weights())
            print('saved weights at', path)

        if i % args.i_weights == 0:
            for k in models:
                save_weights(models[k], k, i)

        if i % args.i_print == 0 or i < 10:

            print(expname, i, psnr.numpy(), loss.numpy(), global_step.numpy())
            print('iter time {:.05f}'.format(dt))

            if i % args.i_img == 0:
                # Log a rendered validation view to Tensorboard
                img_i = np.random.choice(i_val)
                image_path = image_paths[img_i]
                sample,d = gen_rader_rays(image_path,h_r,theta,pixel,Na,Nr,0,args.N_theta,scale)
                image = transform.resize(imageio.imread(image_path),[Nr,Na])
                # sample = total_sample[img_i]
                # d = total_D[img_i]
                # image = total_image[img_i]

                sample = tf.transpose(sample,[1,2,3,0])
                d = tf.transpose(d,[1,0])
                
                #image = np.expand_dims(image,axis=-1)

                # d = np.broadcast_to(d,np.shape(sample))

                # batch_rays = tf.concat([sample,d],axis=-1)
                target = flip180(image)
                s_map = render(sample,d,**render_kwargs_test)

                psnr = mse2psnr(img2mse(s_map, target))

                testimgdir = os.path.join(basedir, expname, 'viz')
                if i==0:
                    os.makedirs(testimgdir, exist_ok=True)


                imageio.imwrite(os.path.join(testimgdir, '{:06d}_false.png'.format(i)), to8b(flip180(s_map.numpy())))
                imageio.imwrite(os.path.join(testimgdir, '{:06d}_true.png'.format(i)), to8b(image))

        global_step.assign_add(1)

def test_phi(phi_list,is_video=False):
   
    parser = config_parser()
    args = parser.parse_args()
 
    scale = args.scale

    #解析basedir expname ,并创建输出文件夹
    print("常见输出的文件夹")
    if len(args.datadir.split("/")[-1]) != 0:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-1])
        expname = args.datadir.split("/")[-1]
    else:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-2])
        expname = args.datadir.split("/")[-2]
    
    path_out_checkout = os.path.join(path_out,"checkout")
    path_out_phi = os.path.join(path_out,"phi")
    mkdir("./out")
    mkdir(path_out)
    mkdir(path_out_checkout)
    mkdir(path_out_phi)
    basedir = "./out"

    #加载训练好的网络
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        basedir, expname,args)

    # Load data
    # 该代码使用得数据集为mstar
    if args.dataset_type == 'mstar':
        # total_sample,total_D,total_image = load_mstar_data(args.datadir)
        h_r,theta,pixel,Na,Nr = load_json_file(os.path.join(args.datadir,"params.json"),scale)
        
    else:
        print("Unknown dataset type")
    
    images = []
    for phi in tqdm(phi_list):
        image = render_phi(phi,h_r,theta,pixel,Na,Nr,args.N_theta,scale,**render_kwargs_test)
        images.append(image)
        imageio.imwrite(os.path.join(path_out_phi, '{:06f}_false.png'.format(phi)), image)
    if is_video:
        print("开始生成video")
        imageio.mimwrite(os.path.join(path_out_phi ,'{:06d}_rolate.mp4'.format(start)),
                             images, fps=10)

def val(path):
    # scale = 0.25
    parser = config_parser()
    args = parser.parse_args()
    
    scale = args.scale
    #解析basedir expname ,并创建输出文件夹
    print("常见输出的文件夹")
    if len(args.datadir.split("/")[-1]) != 0:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-1])
        expname = args.datadir.split("/")[-1]
    else:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-2])
        expname = args.datadir.split("/")[-2]
    
    path_out_checkout = os.path.join(path_out,"checkout")
    path_out_val = os.path.join(path_out,"val")
    
    mkdir(path_out_val)
    basedir = "./out"

    val_images = [os.path.join(path,image_path) for image_path in os.listdir(path) if os.path.splitext(image_path)[-1] in [".tif",".jpg",".png"]]

    #加载训练好的网络
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        basedir, expname,args)

     # Load data
    # 该代码使用得数据集为mstar
    if args.dataset_type == 'mstar':
        # total_sample,total_D,total_image = load_mstar_data(args.datadir)
        h_r,theta,pixel,Na,Nr = load_json_file(os.path.join(args.datadir,"params.json"),scale)
        
    else:
        print("Unknown dataset type")

    for image_path in tqdm(val_images):
        sample,d = gen_rader_rays(image_path,h_r,theta,pixel,Na,Nr,0,args.N_theta,scale)
        image = transform.resize(imageio.imread(image_path),[Nr,Na])
        # sample = total_sample[img_i]
        # d = total_D[img_i]
        # image = total_image[img_i]

        sample = tf.transpose(sample,[1,2,3,0])
        d = tf.transpose(d,[1,0])
        
        #image = np.expand_dims(image,axis=-1)

        # d = np.broadcast_to(d,np.shape(sample))

        # batch_rays = tf.concat([sample,d],axis=-1)
        target = flip180(image)
        s_map = render(sample,d,**render_kwargs_test)

        psnr = mse2psnr(img2mse(s_map, target))

        file_name = image_path.split("/")[-1]
        phi = (float(file_name.split("_")[0]) + 90)%360
        # phi = phi * pi / 180

        imageio.imwrite(os.path.join(path_out_val, '{:03f}_false.png'.format(phi)), to8b(flip180(s_map.numpy())))
        imageio.imwrite(os.path.join(path_out_val, '{:03f}_true.png'.format(phi)), to8b(image))
        pass
    pass

def get_fn():
    parser = config_parser()
    args = parser.parse_args()
    
    scale = args.scale
    #解析basedir expname ,并创建输出文件夹
    print("常见输出的文件夹")
    if len(args.datadir.split("/")[-1]) != 0:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-1])
        expname = args.datadir.split("/")[-1]
    else:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-2])
        expname = args.datadir.split("/")[-2]
    
    path_out_val = os.path.join(path_out,"mesh")
    
    mkdir(path_out_val)
    basedir = "./out"
    render_kwargs_train, render_kwargs_test, start, grad_vars, models = create_nerf(
        basedir, expname,args)
    
    return expname,basedir,render_kwargs_train, render_kwargs_test, start, grad_vars, models 

def gen_config_file():
    parser = config_parser()
    args = parser.parse_args()
    
    scale = args.scale
    #解析basedir expname ,并创建输出文件夹
    print("常见输出的文件夹")
    if len(args.datadir.split("/")[-1]) != 0:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-1])
        expname = args.datadir.split("/")[-1]
    else:
        path_out = os.path.join("./out/" ,args.datadir.split("/")[-2])
        expname = args.datadir.split("/")[-2]
    
    f = os.path.join(path_out,"config.txt")
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))

    basedir = "./out"


if __name__ == '__main__':
    train()
