from run_nerf import *
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    
    phi_list = np.arange(0,360,2)
    phi_list = (phi_list+90)%360
    
    test_phi(phi_list,True)