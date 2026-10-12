import torch
import pickle
import numpy as np
import argparse
from tqdm import tqdm
import random
import data_process_utils as DPU
import os
import json
import random
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap



def plot_motion(all_trajectories , all_angles ,  id ):

        plt.figure(figsize=(8, 6))
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.title(title )
        plt.xlim( -4 ,4 )
        plt.ylim( -4 , 4)
        plt.grid(True)

        R = np.asarray( [[ 0,1,0 ], [1,0,0] , [0,0,1] ] )

        for i in range(len(all_trajectories)):

            dir_x , dir_y = torch.cos( all_angles[i] ).unsqueeze(1) , torch.sin( all_angles[i] ).unsqueeze(1)
            orients = torch.cat( [ dir_x , dir_y] , dim =-1 ).detach().to("cpu").numpy()

            orients = ( R[0:2, 0:2]@orients.T ).T

            x = all_trajectories[i][:, 0]
            y = all_trajectories[i][:, 1]
            idx = np.arange(0, len(x) , 30 )  #np.random.choice(trajectory.shape[0], 5,  replace=False )
            X  = x[idx ]
            Y = y[idx]

            plt.scatter(x[0], y[0], color='red', label='start' , s=60)
            plt.scatter(x[-1], y[-1], color='green', label='end', s=60)
            # plt.plot(x, y , label="trajectory" )
            q = plt.quiver(X, Y , orients[idx, 0], orients[idx, 1] , scale=30 )

            for j in range(len(X) - 1):
                # plt.plot(x[j:j+2], y[j:j+2], color=cmap(j / len(x)) , linewidth=2.0)

                plt.arrow( X[j] , Y[j] , X[j+1] , Y[j+1] , width=0.01 )

        save_name = args.save_path + "/" + str(id) +".png"

        plt.savefig(save_name)


def read_txt_data( file_path ):
    file1 = open(file_path , "r")
    data = file1.readlines()
    return data

def get_concepts( sentence , all_concepts ):

    concept_seq = []
    sentence = sentence.split(".\n")[0]

    for word in sentence.split(" "):

        if word in all_concepts:
            concept_seq.append( word )

    if 'sideways' in concept_seq:
        idx = concept_seq.index( 'sideways' )
        if 'right' in concept_seq:
            concept_seq[idx] = 'sideways to the right'
            concept_seq.remove('right')
        else:
            concept_seq[idx] = 'sideways to the left'
            concept_seq.remove('left')

    out = ""
    for c in concept_seq:
        out += c + " "


    return out +"\n"


def load_all_traj():

    num_files=len( test_files )
    xy_trajectories = []
    yaw_angles = []

    concept_list = ['forward', 'backward', 'left', 'right' , 'clockwise', 'counter-clockwise' , 'sideways' ]
    all_concepts = []
    cntr = 0

    for i in range(num_files):

        traj =  torch.load( os.path.join( args.pose_folder , test_files[i] ) )
        text_data = read_txt_data( os.path.join( args.text_folder , text_files[i] ) )[0]
        concept_seq = get_concepts( text_data , concept_list )
        if concept_seq == "\n":
            continue
        all_concepts.append(concept_seq )

        traj2 = torch.zeros( traj.shape )
        traj2[:, 1] = traj[:, 0]
        traj2[:, 0] = traj[:, 1]
        traj2[:, 2] = traj[:, 2]
        poses = torch.cumsum( traj2  , dim =0 ).unsqueeze(0)#.numpy()

        plot_motion( poses[:, :, 0:2 ], poses[:, :,  -1] , cntr )
        cntr +=1

    save_txt_path = args.save_path + "/" + "concepts.txt"

    f = open(save_txt_path , "w")
    f.writelines(all_concepts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("data loader for robot trajectory tokenizers ")
    parser.add_argument("--pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--text_folder", help="point to the folder with text annotation" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--num_test_files", help="number of files to select" , default=100, type =int)
    parser.add_argument("--save_path", help="test path" , default="REPLACE_WITH_LOCAL_PATH", type =str )
    args = parser.parse_args()

    all_traj_files = os.listdir( args.pose_folder )

    test_files = random.sample( all_traj_files , args.num_test_files)

    text_files = []
    cmap = ListedColormap(np.linspace([1, 0, 0], [0, 1, 0], 192))

    for file in test_files:
        name = file.split(".")[0] + ".txt"
        text_files.append(name)

    load_all_traj()









