import torch 
import pickle 
import numpy as np  
import argparse
from tqdm import tqdm 
import random
import json 
import os 
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt 


def get_velocity( translation ):
    
    
    X = torch.tensor(translation)
    x_behind = torch.roll(X, 1 , 0)
    x_ahead = torch.roll( X, -1 , 0 )
    gradient = (x_ahead - x_behind )/2
    velocity = gradient[1:-1, : ]

    return velocity.numpy()

def get_canonical_se2_path( position , rotation ):
    
    velocity = torch.from_numpy( get_velocity(position)[:, 0:2] )
    heading =  torch.from_numpy( rotation@np.asarray( [1,0,0] ) )
    angles = torch.atan2( heading[:, 1] , heading[:, 0] )

    x_behind = torch.roll(angles, 1 , 0)
    x_ahead = torch.roll( angles , -1 , 0 )
    gradient = (x_ahead - x_behind )/2
    angular_velocity = gradient[1:-1  ]

    data = torch.cat( [ velocity , angular_velocity.unsqueeze(1) ] ,dim=-1 )
    return data 

def center_origin( trajectory ):
    
    first_pose = trajectory[0, :]
    return trajectory - first_pose 

def check_fittness( trajectory , drift_threshold):
    
    
    fit = True 
    num_poses = len(trajectory)

    for i in range( 1, num_poses):
        past_pose = trajectory[i-1]
        current_pose = trajectory[i]
        distance = np.linalg.norm( current_pose- past_pose)

        if distance >= drift_threshold:
            fit = False 
            return fit 
    
    return fit 

def read_trajectory( traj , drift_threshold):
    
    translation = [] 
    rotation = []
    num_poses = len(traj)

    for i in range(num_poses):
        x = traj[i].position.x 
        y = traj[i].position.y 
        z = traj[i].position.z 
        
        translation.append( [x,y,z] )

        r_x= traj[i].rotation.x 
        r_y =traj[i].rotation.y
        r_z =traj[i].rotation.z
        r_q = traj[i].rotation.w 

        rotation.append( [ r_x, r_y, r_z, r_q ] )  

    translation = np.asarray( translation )  

    fit = check_fittness( translation , drift_threshold)
    if fit == True:
        rotation = np.asarray(rotation)
        traj = center_origin( translation )
        Rot_r = R.from_quat(rotation).as_matrix()
        rot = Rot_r[ 0 , : , : ].T
        traj = (rot@traj.T).T
        Rot_r2 = rot@Rot_r

        return fit, traj , Rot_r2

    else :

        return fit, None, None 

def read_pickle( data_path ):
    
    with open( data_path , "rb") as f:
        data = pickle.load(f)
    
    return data 

def subsample_sequence(  data_list , sample_idx ):
    
    res_list = [data_list[int(i)] for i in sample_idx]

    return res_list

def split_and_load_se2( global_poses, joint_states , trajectory_len, drift_threshold ):
    
    assert len(global_poses) == len(joint_states)
    
    num_poses = len( global_poses )
    start_idx = np.arange(0, num_poses, trajectory_len )
    end_idx = start_idx + trajectory_len 

    multiple_trajectories = []
    unfit_cntr = 0 

    for i in range( len(start_idx) ):
        traj = global_poses[ start_idx[i]:end_idx[i] ]
        if len(traj) == trajectory_len :
            fit, position, rotation = read_trajectory(traj , drift_threshold)
            if fit==True: 

                se2_path = get_canonical_se2_path( position, rotation )
                multiple_trajectories.append(se2_path.unsqueeze(0) )
            else:
                unfit_cntr +=1 
    

    return torch.cat(multiple_trajectories , dim=0 )

def load_spot_data(  data_folder , meta_data_path , augment_strategy ):
    
    files = os.listdir( data_folder )
    data_list = []
    seq_lens = []
    global_poses_list= []
    file_names = []

    for file in tqdm(files):
        # print(file)
        data_path = os.path.join(  data_folder , file )
        file_names.append( file.split(".")[0] )
        meta_path = os.path.join( meta_data_path , file.split(".")[0] +".json" )
        f = open(meta_path)
        meta_data = json.load(f)
        data = read_pickle(data_path)
        joint_states = data['joint_states']
        global_poses = data['poses']

        subsampled_global_poses = subsample_sequence( global_poses , meta_data['time_sample_idx'])
        subsampled_joint_states = subsample_sequence( joint_states , meta_data['time_sample_idx'])
        

        trajectories = split_and_load_se2(subsampled_global_poses , 
         subsampled_joint_states , meta_data['trajectory_len'] , 
         meta_data['drift_threshold'] )        
                            
        global_poses_list.append( trajectories )

    return global_poses_list , file_names

def get_reconstructed_traj( se2_traj , save_dir ):

    lin_vel = se2_traj[:, :,  0:2]
    ang_vel = se2_traj[:, :,  -1]

    position = torch.cumsum(lin_vel , dim =1 ).numpy()
    angles =  torch.cumsum( ang_vel , dim =-1)

    dir_x , dir_y = torch.cos( angles ).unsqueeze(2) , torch.sin( angles ).unsqueeze(2)
    orientations = torch.cat( [ dir_x , dir_y] , dim =-1 ).numpy()

    R = np.asarray( [[ 0,1,0 ], [1,0,0] , [0,0,1] ] )

    num_traj = len(position)

    for i in range(num_traj):
        
        poses = ( R[0:2, 0:2]@position[i, :, :].T).T
        orients = ( R[0:2, 0:2]@orientations[i,  :].T ).T

        plt.figure(figsize=(8, 6))

        x = poses[:, 0]
        y = poses[:, 1]
        idx = np.arange(0, len(x) , int(len(x) /10 ) ) #np.random.choice(trajectory.shape[0], 5,  replace=False )
        X  = x[idx ]
        Y = y[idx]
        plt.scatter(x[0], y[0], color='red', label='start' , s=60)
        plt.scatter(x[-1], y[-1], color='green', label='end', s=60)
        plt.plot(x, y , label="trajectory" )
        q = plt.quiver(X, Y , orients[idx, 0], orients[idx, 1] , scale=40 )
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory with Orientations (Quiver Plot)')
        plt.xlim( -5 ,7 )
        plt.ylim( -5 , 5)
        plt.grid(True)
        plt.legend()

        save_name =  save_dir + "/" + str(i) + ".png"

        plt.savefig(save_name)

        plt.clf()