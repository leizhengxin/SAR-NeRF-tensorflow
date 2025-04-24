def gen_rader_rays_phi(phi,h_r,theta,pixel,Na,Nr,perturb,N_theta):
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
    delta_a = pixel
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

def render_phi(phi,h_r,theta,pixel,Na,Nr,N_theta,**kwargs):
    sample,d = gen_rader_rays_phi(phi, h_r, theta, pixel, Na, Nr,0, N_theta)
    
    sample = tf.transpose(sample,[1,2,3,0])
    d = tf.transpose(d,[1,0])
    s_map = render(sample,d,**kwargs)

    gen_image = to8b(flip180(s_map.numpy()))
    return gen_image
def test_phi(phi_list):
    scale = 0.25
    parser = config_parser()
    args = parser.parse_args()

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
    mkdir(path_out_viz)
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
    
    for phi in phi_list:
        image = render_phi(phi,h_r,theta,pixel,Na,Nr,args.N_theta,**render_kwargs_test)
        imageio.imwrite(os.path.join(path_out_phi, '{:06f}_false.png'.format(phi)), image)