import torch 
import pickle 
import numpy as np  
import argparse
from tqdm import tqdm 
import random
import data_process_utils as DPU
import os 
import json 

def save_all_trajectories():

    for i in range(len( file_names )):
        name = file_names[i] + ".pt"
        save_file_path = os.path.join( args.save_data_path , name )

        torch.save( all_trajectories[i] , save_file_path )

def reconstruct_and_plot_trajectories():

    for i in tqdm( range( len( file_names) ) ):
        name = file_names[i]
        traj = all_trajectories[i]

        save_dir = os.path.join( args.save_img_path , file_names[i] )
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        DPU.get_reconstructed_traj( traj , save_dir )

def scale_augment( scale, trajectory , sub_name , id , scale_save_val , dir_path ):

    new_traj = ( 1 + scale/100)* trajectory

    save_name  = "S_" + scale_save_val + sub_name +"_" + str(id) + ".pt"
    path = os.path.join( dir_path  , save_name )
    return new_traj , path , save_name

def read_txt_data( file_path ):
    file1 = open(file_path , "r")
    data = file1.readlines()
    return data

def add_prefix( data ):

    return "A robot " + data

def load_multiple_annotations( original_input ):


    prmpt = original_input.split(".\n")[0]
    prmpt = prmpt.split(".")[0]

    all_annots = multiple_annotators[prmpt]
    common_annots = [ add_prefix(prmpt) +".\n" ] 
    for i in range(len(all_annots)):
        common_annots.append( add_prefix( all_annots[i] ) + "\n" )
    
    return common_annots

def load_mirror_annotations( original_input ):
    
    prmpt = original_input.split(".\n")[0]
    prmpt = prmpt.split(".")[0]

    corresp_mirror_annot = mirror_map[prmpt]
    all_annots = multiple_annotators[corresp_mirror_annot]

    common_annots = [ add_prefix( corresp_mirror_annot ) +".\n" ] 
    for i in range(len(all_annots)):
        common_annots.append( add_prefix(all_annots[i] ) + "\n" )
    
    return common_annots

def write_txt( description , save_path ):
    
    file1 = open(save_path , "w")
    file1.writelines(description)

def augment_and_save_trajectories():

    for i  in tqdm( range( len(file_names))):

        sub_name = replaced_name[  all_names.index(file_names[i]) ]
        num_traj = len( all_trajectories[i] )
        file_path = os.path.join( args.manual_annotation_path , sub_name + ".txt" )
        text_annotations = read_txt_data(  file_path )

        for j in range(num_traj):
            traj = all_trajectories[i][j]
            
            save_name = sub_name +"_" + str(j) +".pt"
    
            save_traj_path = os.path.join( args.save_data_path , save_name )

            mirror_traj = torch.zeros( traj.shape )
            mirror_traj[:, 0:2] = - traj[:, 0:2]
            mirror_traj[:, 2] = traj[:, 2]
 
            mirror_save_name = "M_" + sub_name +"_" + str(j) +".pt"
            
            mirror_save_traj_path = os.path.join( args.save_data_path , mirror_save_name )

            scale_5_traj , scale_5_path , scale_5_name = scale_augment( 5, traj ,sub_name, j , "5_" , args.save_data_path )
            scale_10_traj , scale_10_path , scale_10_name  = scale_augment(10, traj , sub_name , j, "10_" , args.save_data_path)
            scale_15_traj , scale_15_path , scale_15_name = scale_augment( 15, traj ,sub_name, j , "15_" , args.save_data_path )
            scale_20_traj , scale_20_path  , scale_20_name = scale_augment( 20, traj ,sub_name, j , "20_" , args.save_data_path )

            original_annotation = text_annotations[j]

            all_annots = load_multiple_annotations( original_annotation )

            all_mirror_annots = load_mirror_annotations( original_annotation ) 

            original_annotation_path =  os.path.join( args.save_annotation_path ,  save_name.split(".")[0] + ".txt" )
            mirror_annotation_path =  os.path.join( args.save_annotation_path ,   mirror_save_name.split(".")[0] + ".txt" )
            scale_5_annotation_path =  os.path.join(args.save_annotation_path ,   scale_5_name.split(".")[0] + ".txt" ) 
            scale_10_annotation_path =  os.path.join( args.save_annotation_path ,   scale_10_name.split(".")[0] + ".txt" )
            scale_15_annotation_path =  os.path.join( args.save_annotation_path ,   scale_15_name.split(".")[0] + ".txt" )
            scale_20_annotation_path =  os.path.join( args.save_annotation_path ,   scale_20_name.split(".")[0] + ".txt" ) 

            write_txt( all_annots , original_annotation_path  )
            write_txt( all_mirror_annots , mirror_annotation_path  )
            write_txt( all_annots , scale_5_annotation_path  )
            write_txt( all_annots , scale_10_annotation_path  )
            write_txt( all_annots , scale_15_annotation_path  )
            write_txt( all_annots , scale_20_annotation_path  )


            torch.save( traj , save_traj_path )
            torch.save( mirror_traj , mirror_save_traj_path )
            torch.save( scale_5_traj , scale_5_path )
            torch.save( scale_10_traj , scale_10_path )
            torch.save( scale_15_traj , scale_15_path )
            torch.save( scale_20_traj , scale_20_path )



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data loader for robot trajectory tokenizers ")
    parser.add_argument("--pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--meta_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--save_data_path", help="path to save the results" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--save_annotation_path", help="path to save the results" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--save_img_path", help="path to save the results" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--augment_strategy", help="path to save the results" , default="set to either 'None', 'mirror' , 'scale' , 'mirror_x' ", type =None )
    parser.add_argument("--manual_annotation_path", help="path to save the results" , default="REPLACE_WITH_LOCAL_PATH", type =str )

    args = parser.parse_args()

    f1, f2 = open('text_templates.json') ,  open('mirror_map.json')

    multiple_annotators = json.load(f1)
    mirror_map = json.load(f2)


    all_names = ['long_range_jog_1', 'long_range_jog_2', 'long_range_jog_3', 'long_range_jog_4', 'long_range_jog_5', 'long_range_jog_6', 'long_range_jog_7', 'long_range_jog_8', 'long_range_jog_9', 'long_range_walk_1', 'long_range_walk_2', 'long_range_walk_3', 'long_range_walk_4', 'long_range_walk_5', 'long_range_walk_6', 'long_range_walk_7', 'long_range_walk_8', 'long_range_walk_9']
    replaced_name =  [ 'J_1' , 'J_2' , 'J_3', 'J_4', 'J_5' , 'J_6' , 'J_7' , 'J_8' , 'J_9' , 'W_1' , 'W_2' , 'W_3' , 'W_4' , 'W_5' , 'W_6' , 'W_7' , 'W_8' , 'W_9' ]

    all_trajectories , file_names = DPU.load_spot_data( args.pose_folder , args.meta_path , args.augment_strategy )

    augment_and_save_trajectories()

    # save_all_trajectories()
    # reconstruct_and_plot_trajectories()
