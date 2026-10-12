import numpy as np 
import os 
import argparse
from tqdm import tqdm 
import torch 
import pickle 
from scipy.spatial.transform import Rotation as R
import json
import random 


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    # print(q.shape)
    q = q.contiguous().view(-1, 4)
    v = v.contiguous().view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qinv(q):
    assert q.shape[-1] == 4, 'q must be a tensor of shape (*, 4)'
    mask = torch.ones_like(q)
    mask[..., 1:] = -mask[..., 1:]
    return q * mask


def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos

def recover_from_ric(data, joints_num):
    
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

def calculate_mean_and_std( poses ):

    data = poses.to("cpu")[0, ...].numpy()
    joints_num = 22
    
    Mean = data.mean(axis=0)
    Std = data.std(axis=0)
    Std[0:1] = Std[0:1].mean() / 1.0
    Std[1:3] = Std[1:3].mean() / 1.0
    Std[3:4] = Std[3:4].mean() / 1.0
    Std[4: 4+(joints_num - 1) * 3] = Std[4: 4+(joints_num - 1) * 3].mean() / 1.0
    Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9] = Std[4+(joints_num - 1) * 3: 4+(joints_num - 1) * 9].mean() / 1.0
    Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3] = Std[4+(joints_num - 1) * 9: 4+(joints_num - 1) * 9 + joints_num*3].mean() / 1.0
    Std[4 + (joints_num - 1) * 9 + joints_num * 3: ] = Std[4 + (joints_num - 1) * 9 + joints_num * 3: ].mean() / 1.0

    return Mean , Std**2 

def calculate_fid( mu, var, mu_gt, var_gt ):

    diff_mean = np.linalg.norm( mu - mu_gt )**2 
    # diff_var = np.sum( var+ var_gt - 2*(np.reshape(var , (1, 263) )@ np.reshape(var_gt , (263, 1) ) )**0.5 )

    fid = diff_mean #- diff_var 
    return fid  

def get_velocity( translation ):

    
    X = torch.tensor(translation)
    x_behind = torch.roll(X, 1 , 0)
    x_ahead = torch.roll( X, -1 , 0 )
    gradient = (x_ahead - x_behind )/2
    velocity = gradient[1:-1, : ]

    return velocity.numpy()

def canonicalize_global_pose( global_poses , time_stpes , sample_fps ) :

    num_samples = len(global_poses)
    original_fps =  num_samples / (time_stpes[-1] - time_stpes[0] )
    resample_ratio = original_fps // sample_fps 
    sample_idx = np.arange(0, num_samples, resample_ratio )

    translation = []
    rotation = []

    for i in sample_idx :
        i = int(i)
        x = global_poses[i].position.x 
        y= global_poses[i].position.y 
        z = global_poses[i].position.z 

        translation.append( [x,y,z] )

        r_x= global_poses[i].rotation.x 
        r_y =global_poses[i].rotation.y
        r_z =global_poses[i].rotation.z
        r_q = global_poses[i].rotation.w 

        rotation.append( [ r_q, r_x, r_y, r_z ] )
    
    translation = np.asarray( translation )

    rotation = np.asarray( rotation )

    rotation_6d = get_6d_poses( rotation )[1:-1, :]
    velocity = get_velocity(translation)[:, 0:2]
    Z= np.expand_dims( translation[1:-1 , : ][:, 2] , axis =1 )

    canonical_global_pose = np.hstack( [ velocity, Z, rotation_6d ] )

    return canonical_global_pose 

def get_all_time_stamps( joint_states ):

    num_samples = len(joint_states)
    time_steps = []

    for i in range(num_samples):
        t = joint_states[i].robot_kinematics.acquisition_timestamp.seconds 
        time_steps.append(t)
    
    return time_steps 

def collect_joint_data( joint_states , joint_trajectory):

    for j in range( len(joint_states)  ):

        if joint_states[j].name.split(".")[1] == "hr0" or joint_states[j].name.split(".")[1] == "hr1":
            continue
        else:
            joint_trajectory[ joint_states[j].name ].append( joint_states[j].position.value )
    
    return joint_trajectory

def canonicalize_joint_states( joint_states , time_stpes , sample_fps ):

    num_samples = len(joint_states)
    original_fps =  num_samples / (time_stpes[-1] - time_stpes[0] )
    resample_ratio = original_fps // sample_fps 
    sample_idx = np.arange(0, num_samples, resample_ratio )

    joint_trajectory = {}

    joint_names = [ "fl.hx" , "fl.hy" , "fl.kn" , "fr.hx" , "fr.hy" , "fr.kn" ,  "hl.hx" , "hl.hy" , 
    'hl.kn' , 'hr.hx' , 'hr.hy' , 'hr.kn' , 'arm0.sh0' , 'arm0.sh1' , 
    'arm0.el0' , 'arm0.el1' , 'arm0.wr0' , 'arm0.wr1' , 'arm0.f1x']

    for name in joint_names:
        joint_trajectory[name] = []

    for i in sample_idx :
        i = int(i)
        common_feature = []
        all_joints = joint_states[i].robot_kinematics.joint_states 
        
        for j in range( len(all_joints)  ):
            if all_joints[j].name.split(".")[1] == "hr0" or all_joints[j].name.split(".")[1] == "hr1":
                continue
            else:
                joint_trajectory[ all_joints[j].name ].append( all_joints[j].position.value )
    
    
    joint_trajectory_matrix = np.zeros( ( len(sample_idx) , len(joint_names) ) )

    col = 0
    for name in joint_names:
        joint_traj = joint_trajectory[name]
        joint_trajectory_matrix[:, col] = joint_traj 
        col +=1 
    
    return joint_trajectory_matrix[1:-1,  : ] 

def read_pickle( data_path ):
    
    with open( data_path , "rb") as f:
        data = pickle.load(f)
    
    return data 

def get_6d_poses( quat ):
    
    permuted_quat = np.zeros( quat.shape )
    # permuted_quat[:, 0] ,permuted_quat[:, 1] ,permuted_quat[:, 2] ,permuted_quat[:, 3] = quat[:, 1] , quat[:, 2] , quat[:, 3] , quat[:, 0]

    r = R.from_quat( np.asarray( [quat[:, 1] , quat[:, 2] ,
     quat[:, 3] , quat[:, 0 ] ] ).T ).as_matrix()[:, 0:2, :] 

    rot_6d_1, rot_6d_2 = r[:, 0, :] , r[:, 1, : ]
    rot_6d = np.hstack( [ rot_6d_1 , rot_6d_2] )

    return rot_6d

# def load_robot_data( robot_data_dir , sample_fps ):

#     '''
#     canoinical pose is consists of 
#     [V, Z, R,  J ]
#     V = root planar velocity velocity R^{2}
#     Z = height above the ground R^{1}
#     R = Global 6d continous Rotation  R^{6}
#     J = Joint angles R^{19} (including arm)
#     total dimesion R^{29}
#     '''

#     gaits = os.listdir( robot_data_dir )
#     data_list = []
#     seq_lens = []
#     for gait in  gaits :
#         gait_path = os.path.join( robot_data_dir , gait  )
#         for file in os.listdir( gait_path ):
#             file_path = os.path.join( gait_path, file )
#             data = read_pickle(file_path)
#             joint_states = data['joint_states']
#             global_poses = data['poses']

#             time_stpes = get_all_time_stamps(joint_states)
#             canonical_global_pose = canonicalize_global_pose( global_poses , time_stpes , sample_fps )
#             joint_trajectory_matrix = canonicalize_joint_states( joint_states,  time_stpes , sample_fps )

#             canonical_pose = np.hstack( [ canonical_global_pose, joint_trajectory_matrix ] )
            
#             data_list.append( canonical_pose)
#             seq_lens.append( len(canonical_pose ) )
    
#     return data_list , seq_lens


def subsample_sequence(  data_list , sample_idx ):
    
    res_list = [data_list[int(i)] for i in sample_idx]

    return res_list

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

def center_origin( trajectory ):
    
    first_pose = trajectory[0, :]
    return trajectory - first_pose 

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

def so3_to_6d( rotation_so3 ):
    
    rot_6d_1, rot_6d_2 = rotation_so3[:, 0, :] , rotation_so3[:, 1, : ]
    rot_6d = np.hstack( [ rot_6d_1 , rot_6d_2] )

    return rot_6d

def get_canonical_global_poses( position, rotation):
    
    rotation_6d = so3_to_6d( rotation )[1:-1, :]
    velocity = get_velocity(position)[:, 0:2]
    Z= np.expand_dims( position[1:-1 , : ][:, 2] , axis =1 )
    
    canonical_global_pose = np.hstack( [ velocity, Z, rotation_6d ] )
    
    return canonical_global_pose 

def get_canonical_joint_poses( joint_states ):
    
    joint_trajectory = {}
    
    joint_names = [ "fl.hx" , "fl.hy" , "fl.kn" , "fr.hx" , "fr.hy" , "fr.kn" ,  "hl.hx" , "hl.hy" , 
    'hl.kn' , 'hr.hx' , 'hr.hy' , 'hr.kn' , 'arm0.sh0' , 'arm0.sh1' , 
    'arm0.el0' , 'arm0.el1' , 'arm0.wr0' , 'arm0.wr1' , 'arm0.f1x']
    
    for name in joint_names:
        joint_trajectory[name] = []
    
    for i in range(len(joint_states)) :
        i = int(i)
        common_feature = []
        all_joints = joint_states[i].robot_kinematics.joint_states 
        
        for j in range( len(all_joints)  ):
            if all_joints[j].name.split(".")[1] == "hr0" or all_joints[j].name.split(".")[1] == "hr1":
                continue
            else:
                joint_trajectory[ all_joints[j].name ].append( all_joints[j].position.value )
    
    
    joint_trajectory_matrix = np.zeros( ( len(joint_states) , len(joint_names) ) )

    col = 0
    for name in joint_names:
        joint_traj = joint_trajectory[name]
        joint_trajectory_matrix[:, col] = joint_traj 
        col +=1 
    
    return joint_trajectory_matrix[1:-1,  : ] 

def split_to_multiple_trajectories( global_poses, joint_states , trajectory_len, drift_threshold ):

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
                global_c = get_canonical_global_poses( position, rotation )
                global_j = get_canonical_joint_poses( joint_states[ start_idx[i]:end_idx[i] ]  )
                canonical_pose = np.hstack( [ global_c, global_j  ] )

                canonical_pose = torch.from_numpy( canonical_pose ).unsqueeze(0)
                multiple_trajectories.append(canonical_pose)
        else:
            unfit_cntr +=1 
            
    return torch.cat( multiple_trajectories, dim =0 )

def canonicalize_global_pose( global_poses , time_stpes , sample_fps ) :
    
    num_samples = len(global_poses)
    original_fps =  num_samples / (time_stpes[-1] - time_stpes[0] )
    resample_ratio = original_fps // sample_fps 
    sample_idx = np.arange(0, num_samples, resample_ratio )

    translation = []
    rotation = []

    for i in sample_idx :
        i = int(i)
        x = global_poses[i].position.x 
        y= global_poses[i].position.y 
        z = global_poses[i].position.z 

        translation.append( [x,y,z] )

        r_x= global_poses[i].rotation.x 
        r_y =global_poses[i].rotation.y
        r_z =global_poses[i].rotation.z
        r_q = global_poses[i].rotation.w 

        rotation.append( [ r_x, r_y, r_z, r_q ] )
    
    translation = np.asarray( translation )
    rotation = np.asarray( rotation )
    heading , rot_R = extract_heading(rotation)

    return translation, heading  , rot_R

def split_trajectory( complete_trajectory , heading ,Rot_R ):
    
    num_poses = len( complete_trajectory )
    start_idx = np.arange(0, num_poses, 200 )
    end_idx = start_idx + 200 

    multiple_trajectories = []
    multiple_heading = []

    for i in range( len(start_idx) ):
        traj = complete_trajectory[ start_idx[i]:end_idx[i] , : ]
        fit = check_fittness(traj , 0.5)
        # print( len(traj) , fit)
        if fit == True:
            traj = center_origin( traj )
            rot = Rot_R[ start_idx[i] , : , : ].T
            traj = (rot@traj.T).T
            head = heading[start_idx[i]:end_idx[i] , : ]
            head = ((rot@head.T).T)[:, 0:2]

            multiple_trajectories.append(traj)
            multiple_heading.append( head )
    
    return multiple_trajectories, multiple_heading

def extract_heading( rot_q ):
    rot = R.from_quat(rot_q).as_matrix()

    orientation_vector = rot@np.asarray( [1,0,0] )

    return orientation_vector , rot


# def extract_all_cntrls( multiple_trajectories , multiple_heading ): 
    
#     num_traj = len( multiple_trajectories )

#     assert len(multiple_trajectories) == len(multiple_heading)

#     all_lin_vel =[]
#     all_ang_vel = []

#     for i in range( num_traj ):
#         traj_idx = i
#         trajectory1 = multiple_trajectories[traj_idx]
#         angular_velocity= extract_angular_velocities(multiple_heading[traj_idx] )
#         trajectory = np.zeros( ( len( trajectory1) , 3  ) )
#         trajectory[:, 0] = trajectory1[:, 1 ]
#         trajectory[:, 1] = trajectory1[:, 0 ]
#         trajectory[:, 2] = trajectory1[:, 2]        

#         X = torch.tensor(trajectory)
#         x_ahead = torch.roll( X, -1 , 0 )
#         gradient = (x_ahead - X )
#         gradient = process_gradients(gradient)
#         # gradient = gradient[~torch.all(gradient == 0, dim=1)]
#         command_velocities = gradient[0:-1, : ]
#         command_velocities = torch.nn.functional.interpolate(command_velocities.unsqueeze(0).permute(0,2,1) , ( 260 ) )

#         angular_velocity = torch.nn.functional.interpolate(  angular_velocity.unsqueeze(0).unsqueeze(0) , ( 260 ) ) 

#         command_velocities = command_velocities.permute(0,2,1)#.squeeze(0)
#         all_lin_vel.append( command_velocities )
#         all_ang_vel.append( angular_velocity )
    
#     all_lin_vel, all_ang_vel = torch.cat( all_lin_vel, dim =0 ) , torch.cat(all_ang_vel, dim =0 )
#     return all_lin_vel, all_ang_vel , num_traj

# def process_angualar_velocities( angular_vel ):
    
#     zero_cntr =0
#     processed_grad = []
#     idx =0  
#     fist_non_zero_id = 0
#     fist_non_zero= False
#     for idx in range(len( angular_vel)):
#         g = angular_vel[idx].unsqueeze(0)
#         sum_val = torch.abs( torch.sum(g) )
#         if sum_val > 0.00005:
#             fist_non_zero = True
#             processed_grad.append(g)
#         if fist_non_zero == False and sum_val > 0.00005 :
#             processed_grad.append(g)
#         if sum_val < 0.00005  and fist_non_zero:
#             zero_cntr +=1 
#         elif sum_val > 0.00005 and fist_non_zero :
#             a = g.repeat( zero_cntr+1 )
#             processed_grad.append( a / (zero_cntr+1) )
#             zero_cntr =0
    
#     grad = torch.cat( processed_grad , dim =0  )

#     return grad

def extract_angualar_velocities( orientations ):

    # R = np.asarray( [[ 0,1,0 ], [1,0,0] , [0,0,1] ] )
    # orientations = ( R[0:2, 0:2]@orientations.T ).T
    angles = -np.arctan2(  orientations[:, 1 ] , orientations[:, 0 ] )
    angles_ahead = torch.roll( torch.from_numpy( angles) , -1 , 0 ).numpy()

    angular_velocity = (angles_ahead - angles )#[0:-1 ]

    angular_velocity = process_angualar_velocities( torch.from_numpy( angular_velocity ) )[0:-1]

    return angular_velocity

def process_gradients( gradient ):

    zero_cntr =0
    processed_grad = [] 
    for g in gradient:
        if g[0] == 0 and g[1] ==0:
            zero_cntr +=1 
        else:
            a = g.repeat( zero_cntr+1, 1 )
            processed_grad.append( a / (zero_cntr+1) )
            zero_cntr =0
    
    grad = torch.cat( processed_grad , dim =0  )
    return grad

def load_robot_data( robot_data_dir , meta_data_path ):

    '''
    canoinical pose is consists of 
    [V, Z, R,  J ]
    V = root planar velocity velocity R^{2}
    Z = height above the ground R^{1}
    R = Global 6d continous Rotation  R^{6}
    J = Joint angles R^{19} (including arm)
    total dimesion R^{29}
    '''

    files = os.listdir( robot_data_dir )
    # print(files)
    data_list = []
    seq_lens = []
    global_poses_list= []

    for file in files:
        data_path = os.path.join(  robot_data_dir , file )
        meta_path = os.path.join( meta_data_path , file.split(".")[0] +".json" )
        f = open(meta_path)
        meta_data = json.load(f)
        data = read_pickle(data_path)
        joint_states = data['joint_states']
        global_poses = data['poses']

        subsampled_global_poses = subsample_sequence( global_poses , meta_data['time_sample_idx'])
        subsampled_joint_states = subsample_sequence( joint_states , meta_data['time_sample_idx'])
        
        trajectories = split_to_multiple_trajectories(subsampled_global_poses , 
         subsampled_joint_states , meta_data['trajectory_len'] , 
         meta_data['drift_threshold'] )

        a,b,c = trajectories.shape 
        for l in range(a):
            data_list.append( trajectories[l, : ] )
            global_poses_list.append( torch.tensor( trajectories[l, : , 0:9] ) )
            seq_lens.append( meta_data['trajectory_len'] - 4  )

        # data_list.append( trajectories )
        # seq_lens.append( meta_data['trajectory_len'] - 2  )
    
    # data = torch.cat( data_list, dim =0 )

    
    return data_list  , seq_lens , global_poses_list 

def get_canonical_se2_path( position , rotation ):

    velocity = torch.from_numpy( get_velocity(position)[:, 0:2] )
    heading =  torch.from_numpy( rotation@np.asarray( [1,0,0] ) )
    angles = -torch.atan2( heading[:, 1] , heading[:, 0] )

    x_behind = torch.roll(angles, 1 , 0)
    x_ahead = torch.roll( angles , -1 , 0 )
    gradient = (x_ahead - x_behind )/2
    angular_velocity = gradient[1:-1  ]

    data = torch.cat( [ velocity , angular_velocity.unsqueeze(1) ] ,dim=-1 )
    return data 

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

def load_spot_data(  data_folder , meta_data_path ):

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

def load_sim_spot_data( pose_path , batch_size , device_id ):

    data = torch.load( pose_path )

    traj_data = []
    start_idx = torch.arange( 0, len(data) , batch_size )


    for i in range(len(start_idx)):

        end_idx = start_idx[i] + batch_size
        traj = data[start_idx[i] : end_idx, : ]
        traj_data.append(traj )
    
    batches_per_device = len(traj_data ) // 8 
    batch_start_id = batches_per_device * device_id 
    batch_end_id = batch_start_id + batches_per_device
    return traj_data[batch_start_id:batch_end_id ] 

def load_all_sim_data( pose_path , batch_size ):
    
    data = torch.load( pose_path )

    traj_data = []
    start_idx = torch.arange( 0, len(data) , batch_size )


    for i in range(len(start_idx)):

        end_idx = start_idx[i] + batch_size
        traj = data[start_idx[i] : end_idx, : ]
        traj_data.append(traj )

    return traj_data 

# def time_scale_trajectories( data ):

def load_and_augment_spot_data( data_folder , meta_data_path ):
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
        
        print( trajectories.shape  )

def split_list(input_list, n):
    # Determine the length of each sublist
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

def load_robot_dataset(  data_path , batch_size ,device_id ):

    all_files = os.listdir(data_path)
    trajectories = []

    for i in range(len(all_files)):
        path =os.path.join( data_path , all_files[i] )
        traj = torch.load( path).unsqueeze(0)
        trajectories.append(traj)
    
    trajectories = torch.cat( trajectories , dim =0)
    train_traj , valid_traj = trajectories[0:-100 , :] , trajectories[-100:, : ]


    start_idx = np.arange( 0, len(train_traj) , batch_size )
    end_idx = start_idx + batch_size

    batch_split_traj = [] 

    for i in range( len(start_idx)):
        st = start_idx[i]
        ed = end_idx[i]

        batch_data = train_traj[st:ed , : ]
        batch_split_traj.append( batch_data )
    
    device_split_data = split_list( batch_split_traj , 8 )

    return device_split_data[device_id] , valid_traj

def read_raw_txt(file_path , readlines):
    f = open(file_path, "r")
    if(readlines == True ):
        content  = f.readlines()
    else:
        content = f.read()
    return content

def tokenize_text( prompt , VOCAB , max_len , end_idx):
    
    per_prompt_data = []

    for word in prompt.lower().split(" "):
        id = VOCAB.index(word)
        per_prompt_data.append( torch.tensor(id).unsqueeze(0) )
        
    for n in range( max_len - len(per_prompt_data) ):
        per_prompt_data.append( torch.tensor(end_idx).unsqueeze(0) )
    
    return per_prompt_data 

def get_negative_idx(n:int, ran: range, exclude:list):
    exclusive_range = range(ran.start, ran.stop-len(exclude))
    randoms = []
    for i in range(n):
        r = random.choice(exclusive_range)
        if r in exclude:
            r = exclusive_range.stop + exclude.index(r)
        randoms.append(r)
    return randoms



def load_data_action_classifier( data_path, prompts_path , batch_size, device_id ):
    all_files = os.listdir(data_path)
    trajectories = []
    traj_labels = []

    all_label_path = "process_robot_motion/text_templates.json"
    labels =  list( json.load(open(all_label_path) ).keys() )
    concept_map = json.load( open("process_robot_motion/concept_map.json" ) )
    concept_order = ["F" , "B" , "KL" , "KR" , "TL" , "TR" , "CC" , "CCC", "SR", "SL", "N" , "S" ]
    multi_labels = []

    vocab_path = "process_robot_motion/t2mt_vocab_new.pkl"
    with open( vocab_path, "rb") as f:
        VOCAB = pickle.load(f)
   
    
    start_idx = len(VOCAB) +1 
    end_idx = start_idx +1 
    vocab_size = 128 
    text_data = []
    max_len = 12
    per_prompt_negative_token = []

    for i in tqdm(range(len(all_files ))):
        path =os.path.join( data_path , all_files[i] )
        traj = torch.load( path).unsqueeze(0)[:, 0:180, : ] 
        prompt = read_raw_txt( os.path.join( prompts_path , all_files[i].split(".")[0] + ".txt" ) , True )[0].split(".\n")[0]#.replace("A robot ", "")

        per_prompt_data = torch.cat( tokenize_text( prompt , VOCAB , max_len , end_idx )  , dim =-1 ).unsqueeze(0)
        text_data.append( per_prompt_data )


        negative_labels = get_negative_idx( 15 , range(0, len(labels) ) , [labels.index(prompt.replace("A robot ", "")  ) ]  )

        negative_tokens = []

        for neg_label in negative_labels:
            neg_prmt = "A robot " + labels[neg_label]
            neg_idx = tokenize_text( neg_prmt , VOCAB , max_len , end_idx )

            negative_tokens.append( torch.cat( neg_idx , dim=0 ).unsqueeze(0)  )

        per_prompt_negative_token.append( torch.cat( negative_tokens , dim =0 ).unsqueeze(0) )
        # traj_labels.append( concept_labels )
        trajectories.append( torch.tensor(traj).to("cuda:" + str(device_id) ) )   

    new_data = "REPLACE_WITH_LOCAL_PATH"     
    all_files = os.listdir( new_data )
    num_files = len(all_files)

    f = open( "test_prompts.txt" , "r")
    all_lines = f.readlines()

    for _ in range(500):
        for j in range(num_files):
            traj = torch.load( os.path.join( new_data , str(j) +".pt" ) ).unsqueeze(0) 
            trajectories.append( torch.tensor(traj).to("cuda:" + str(device_id) ))
            per_prompt_data = torch.cat( tokenize_text( all_lines[j].split("\n")[0] , VOCAB , max_len , end_idx )  , dim =-1 ).unsqueeze(0)
            text_data.append( per_prompt_data )

            negative_labels2 = get_negative_idx( 15 , range(0, len(all_lines) ) , [j]  )
            negative_tokens = []

            for neg_label in negative_labels2:
                neg_prmt = all_lines[neg_label].split("\n")[0]
                neg_idx = tokenize_text( neg_prmt , VOCAB , max_len , end_idx )

                negative_tokens.append( torch.cat( neg_idx , dim=0 ).unsqueeze(0)  )

            per_prompt_negative_token.append( torch.cat( negative_tokens , dim =0 ).unsqueeze(0) )

        
    # labels = torch.cat( traj_labels , dim =0  )
    trajectories = torch.cat( trajectories , dim =0 )
    text_idx = torch.cat( text_data , dim =0 )
    negative_idx = torch.cat(per_prompt_negative_token , dim =0  )

    # train_traj, train_text = trajectories[:-100, :] , text_idx[:-100, : ]
    # valid_traj  , valid_text = trajectories[-100:, : ] , text_idx[-100: , : ]
    # negative_text = negative_idx[:-100, : ]

    train_traj, train_text = trajectories , text_idx
    valid_traj  , valid_text = 0 , 0
    negative_text = negative_idx

    start_idx = np.arange( 0, (len(train_traj)// batch_size)*batch_size -batch_size , batch_size )
    end_idx = start_idx + batch_size

    batch_split_traj = [] 
    batch_split_label = []
    batch_split_neg = []

    for i in range( len(start_idx)):
        st = start_idx[i]
        ed = end_idx[i]

        batch_data = train_traj[st:ed , : ].unsqueeze(0)
        batch_label = train_text[st:ed , : ].unsqueeze(0)
        batch_neg_text = negative_text[st:ed , : ].unsqueeze(0)

        batch_split_traj.append( batch_data )
        batch_split_label.append( batch_label )
        batch_split_neg.append( batch_neg_text )
    
    device_split_data = split_list( batch_split_traj , 8 )
    device_split_label = split_list( batch_split_label , 8 )
    device_split_neg = split_list( batch_split_neg  , 8 )



    return torch.cat(device_split_data[device_id] , dim =0 ) , torch.cat(device_split_label[device_id] , dim =0) , torch.cat(device_split_neg[device_id] , dim =0 ) ,  valid_traj , valid_text



def load_data_rae( data_path, prompts_path , batch_size, device_id ):
    all_files = os.listdir(data_path)
    trajectories = []
    traj_labels = []

    all_label_path = "process_robot_motion/text_templates.json"
    labels =  list( json.load(open(all_label_path) ).keys() )
    concept_map = json.load( open("process_robot_motion/concept_map.json" ) )
    concept_order = ["F" , "B" , "KL" , "KR" , "TL" , "TR" , "CC" , "CCC", "SR", "SL", "N" , "S" ]
    multi_labels = []

    vocab_path = "process_robot_motion/t2mt_vocab_new.pkl"
    with open( vocab_path, "rb") as f:
        VOCAB = pickle.load(f)
   
    
    start_idx = len(VOCAB) +1 
    end_idx = start_idx +1 
    vocab_size = 128 
    text_data = []
    max_len = 12
    per_prompt_negative_token = []

    for i in tqdm(range(len(all_files ))):
        path =os.path.join( data_path , all_files[i] )
        traj = torch.load( path).unsqueeze(0)[:, 0:180, : ] 
        prompt = read_raw_txt( os.path.join( prompts_path , all_files[i].split(".")[0] + ".txt" ) , True )[0].split(".\n")[0]#.replace("A robot ", "")

        per_prompt_data = torch.cat( tokenize_text( prompt , VOCAB , max_len , end_idx )  , dim =-1 ).unsqueeze(0)
        text_data.append( per_prompt_data )


        negative_labels = get_negative_idx( 15 , range(0, len(labels) ) , [labels.index(prompt.replace("A robot ", "")  ) ]  )

        negative_tokens = []

        for neg_label in negative_labels:
            neg_prmt = "A robot " + labels[neg_label]
            neg_idx = tokenize_text( neg_prmt , VOCAB , max_len , end_idx )

            negative_tokens.append( torch.cat( neg_idx , dim=0 ).unsqueeze(0)  )

        per_prompt_negative_token.append( torch.cat( negative_tokens , dim =0 ).unsqueeze(0) )
        # traj_labels.append( concept_labels )
        trajectories.append( torch.tensor(traj).to("cuda:" + str(device_id) ) )   

    new_data = "REPLACE_WITH_LOCAL_PATH"     
    all_files = os.listdir( new_data )
    num_files = len(all_files)

    f = open( "test_prompts.txt" , "r")
    all_lines = f.readlines()

    for j in range(num_files):
        traj = torch.load( os.path.join( new_data , str(j) +".pt" ) ).unsqueeze(0) 
        trajectories.append( torch.tensor(traj).to("cuda:" + str(device_id) ))
        per_prompt_data = torch.cat( tokenize_text( all_lines[j].split("\n")[0] , VOCAB , max_len , end_idx )  , dim =-1 ).unsqueeze(0)
        text_data.append( per_prompt_data )

        negative_labels2 = get_negative_idx( 15 , range(0, len(all_lines) ) , [j]  )
        negative_tokens = []

        for neg_label in negative_labels2:
            neg_prmt = all_lines[neg_label].split("\n")[0]
            neg_idx = tokenize_text( neg_prmt , VOCAB , max_len , end_idx )

            negative_tokens.append( torch.cat( neg_idx , dim=0 ).unsqueeze(0)  )

        per_prompt_negative_token.append( torch.cat( negative_tokens , dim =0 ).unsqueeze(0) )

    
    # labels = torch.cat( traj_labels , dim =0  )
    trajectories = torch.cat( trajectories , dim =0 )
    text_idx = torch.cat( text_data , dim =0 )
    negative_idx = torch.cat(per_prompt_negative_token , dim =0  )

    train_traj, train_text = trajectories[:-100, :] , text_idx[:-100, : ]
    valid_traj  , valid_text = trajectories[-100:, : ] , text_idx[-100: , : ]
    negative_text = negative_idx[:-100, : ]



    start_idx = np.arange( 0, len(train_traj) , batch_size )
    end_idx = start_idx + batch_size

    batch_split_traj = [] 
    batch_split_label = []
    batch_split_neg = []

    for i in range( len(start_idx)):
        st = start_idx[i]
        ed = end_idx[i]

        batch_data = train_traj[st:ed , : ].unsqueeze(0)
        batch_label = train_text[st:ed , : ].unsqueeze(0)
        batch_neg_text = negative_text[st:ed , : ].unsqueeze(0)

        batch_split_traj.append( batch_data )
        batch_split_label.append( batch_label )
        batch_split_neg.append( batch_neg_text )
    
    device_split_data = split_list( batch_split_traj , 8 )
    device_split_label = split_list( batch_split_label , 8 )
    device_split_neg = split_list( batch_split_neg  , 8 )



    return torch.cat(device_split_data[device_id] , dim =0 ) , torch.cat(device_split_label[device_id] , dim =0) , torch.cat(device_split_neg[device_id] , dim =0 ) ,  valid_traj , valid_text
