import os
import torch
import zipfile
from tokenizers import (decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer)
from transformers import GPT2Tokenizer, GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments
import logging
import time
from datasets import Dataset, DatasetDict
from transformers import Trainer, TrainingArguments
import argparse
import pickle
from transformers import Trainer, TrainingArguments
import re
import options.option_vq as option_vq
import models.vqvae as vqvae
from transformers import GenerationConfig
import numpy as np
import post_process
from matplotlib import pyplot as plt
from tqdm import tqdm
import random 
from process_robot_motion.motion_features import MotionDiscriminator
from utils import  eval_trans

def define_robot_token_model():

    args2 = option_vq.get_args_parser()
    args2.dataname = "spot"
    args2.nb_code = 128
    robot_token_model =  vqvae.HumanVQVAE(args2, ## use args to define different parameters in different quantizers
                       args2.nb_code,
                       args2.code_dim,
                       args2.output_emb_width,
                       args2.down_t,
                       args2.stride_t,
                       args2.width,
                       args2.depth,
                       args2.dilation_growth_rate,
                       args2.vq_act,
                       args2.vq_norm)

    print ('loading checkpoint for robot model from {}'.format(args.robot_model_path))
    ckpt = torch.load(args.robot_model_path, map_location='cuda')
    robot_token_model.load_state_dict(ckpt  )
    robot_token_model.eval()
    robot_token_model.to(args.device)

    return robot_token_model.to("cuda")

def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def GPT_model( model_path ):
    model = GPT2LMHeadModel.from_pretrained(model_path ).to(args.device)
    model.eval()
    return model

class StopOnWord:
    def __init__(self, word):
        self.word = word

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token

        return 51691 in input_ids

def get_pixel_id( data , centers  ):

    data =  data.unsqueeze(1)
    centers = centers.unsqueeze(0)
    id =   torch.argmin( torch.abs( data - centers ) , dim =-1)
    return id

def get_goal_tokens( robot_motion_data , goal_id , goal_num ):
    num_batches = len(robot_motion_data)
    all_xy = []
    all_trajectories = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i][:, 0:192, : ].to("cuda")

        input_data = torch.zeros( batch_data.shape ).to("cuda")

        input_data[:, :, 0] = batch_data[:, :, 1]
        input_data[:, :, 1] = batch_data[:, :, 0]
        input_data[:, :, 2]  =batch_data[:, :, 2]

        reconstruction, _, _ = r_token_model.forward(input_data)
        translation = torch.cumsum( reconstruction , dim =1 )[:, :, 0:2 ]
        all_xy.append(translation)
        all_trajectories.append(reconstruction)

    xy = torch.cat( all_xy , dim =0 )
    goals = xy[:, -1 , : ].detach().to("cpu").numpy()

    N_c = ( grid_shape[1] - grid_shape[0] - cell_size ) // (cell_size )

    K = np.arange(0 , N_c+1 , 1)

    x_c = grid_shape[0] + cell_size /2  + K* (cell_size  )
    y_c = grid_shape[1] - cell_size /2  -  K* (cell_size   )

    idx1 = get_pixel_id( torch.from_numpy( goals[:, 0] ) , torch.from_numpy( x_c )  )
    idy = get_pixel_id( torch.from_numpy( goals[:, 1 ] ) , torch.from_numpy( y_c )  )

    idx = idx1 * (N_c+1) + idy 
    all_trajectories = torch.cat( all_trajectories  , dim =0 )

    traj_idx = torch.argwhere( idx == idx[goal_id] )
    goal_traj_gt = all_trajectories[traj_idx]

    save_path = save_dir + "/" +"GT/"

    complete_path = save_path + str(goal_num) +"/"

    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    else:
        print(f"Folder already exists at: {complete_path}")

    for i in range( len(goal_traj_gt) ):
        traj_i = goal_traj_gt[i , : ][0, 0:180, : ]
        traj_path = complete_path + str(i) +".pt"

        torch.save( traj_i , traj_path )

    return idx[goal_id] ,  x_c[ int(idx1[goal_id] ) ]  ,  y_c[ int(idy[goal_id]) ]

def create_goal_strings( goal_idx ):
    num_goals = len(goal_idx )
    goal_string = []

    for i in tqdm(range(num_goals)):

        data = goal_idx[i]
        if data == []:
            string_out = []
        else:

            data2 =  '<goal_id_' + str( int( goal_idx[i] )) + '>'
            # ids, mask = tokenize_string(data2)
            string_out = data2

        goal_string.append( string_out )

    return goal_string

def plot_trajectory(translation , px, py ):

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlim(-7, 7)
    ax.set_ylim( -7 , 7)

    # Set major ticks
    ax.set_xticks(np.arange(-7, 7.5, 0.5))
    ax.set_yticks(np.arange(-7, 7.5, 0.5))

    ax.grid(which='both', color='gray', linestyle='-', linewidth=0.5)

    cell_width = 0.5
    cell_height = 0.5


    # ax.add_patch(rect)
    for i in range( len(translation) ):
        plt.plot( translation[i][:, 0] , translation[i][:, 1 ]  )
    rect = plt.Rectangle((px - cell_width/2  , py - cell_height /2 ), cell_width, cell_height, linewidth=2, edgecolor='r', facecolor='r', alpha=0.5)
    ax.add_patch(rect)

    plt.savefig('validation_results/traj_dist.png')
    plt.clf()

def extract_motion_from_string( motion_string , goal_num , traj_id):

    idx = re.findall(r'<robot_motion_id_(\d+)>' , motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )

    idx_tensor = torch.tensor( int_idx )
    vel_commands = r_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)

    save_path = save_dir + "/" +"Ours/"

    complete_path = save_path + str(goal_num) +"/" 

    if not os.path.exists(complete_path):
        os.makedirs(complete_path)
    
    # torch.save( vel_commands[0:180, : ]  , complete_path + str(traj_id) +".pt" )


    cost = get_smoothnes_cost(vel_commands)

    xy = vel_commands[:, 0:2].detach().to("cpu")
    translation = torch.cumsum(xy, dim =0 ).numpy()
    # translation = torch.tensor( [ [0 , -1 ] , [1 , 0 ] ] , dtype=torch.float32) @ translation.T
    # translation = translation.T

    return translation, cost , idx_tensor


def estimate_success( end_pt, px, py , cell_height ):

    end_x, end_y = end_pt[0] , end_pt[1]

    # print(end_pt ,px, py)
    top_left_x, top_left_y = px - cell_height/2 ,py + cell_height/2
    bottom_right_x, bottom_right_y = px + cell_height/ 2 , py - cell_height/2

    if end_x >= top_left_x -0.25 and end_x <= bottom_right_x +0.25 and end_y >= bottom_right_y -0.25 and end_y <= top_left_y + 0.25:
        return True
    else:
        False

def get_smoothnes_cost( velocities ):

    accel = ( (velocities[1:] - velocities[:-1])*20 )**2
    return torch.sum( accel*1/20  )



if __name__ == "__main__":

    parser = argparse.ArgumentParser("data pre-process for body movements ")

    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str ) #tokenizer_all_tasks5
    parser.add_argument("--robot_pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--meta_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--get_metrics", help="learning rate" , default= 0 , type = int )

    args = parser.parse_args()

    print(args)
    cell_size = 0.5
    grid_shape =[ -7, 7 ]
    r_token_model = define_robot_token_model()
    GPT = GPT_model(args.fine_tune_model_path)
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)


    robot_traj_data , robot_file_order = post_process.load_spot_data( args.robot_pose_folder  , args.meta_path )
    
    select_idx = random.sample(range(1, 1090), 1 )
    all_mean = []

    save_dir = "REPLACE_WITH_LOCAL_PATH"
    cntr = 0

    if args.get_metrics == 0 :
    
        for id in select_idx:
        
            goal_token_idx, px, py = get_goal_tokens( robot_traj_data , id , cntr )
            print(px, py)
            goal_strings = create_goal_strings( [goal_token_idx] )

            prompt = goal_strings[0] + "<eop>" + "<tog>" + " \n "
            prompt_token =  tokenizer(prompt, return_tensors="pt"  )


            output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
            do_sample = True, num_beams =100, num_return_sequences=20, stopping_criteria = [StopOnWord("<eog>")],
                temperature=1, attention_mask =prompt_token.attention_mask.to(args.device) )


            all_trans = []
            sucess_metric = 0
            mean_cost = 0
            for i in range(len(output_text)):
                completed_text = tokenizer.decode(output_text[i], skip_special_tokens=False)
                translation , cost ,idx_tensor = extract_motion_from_string(completed_text , cntr , i )

                sucess = estimate_success(translation[-1, :] ,px, py, 0.5)
                if sucess == True:
                    sucess_metric +=1

                mean_cost += cost

                all_trans.append(translation)

            # print(sucess_metric / len(all_trans) , mean_cost/len(all_trans))

            all_mean.append( sucess_metric / len(all_trans) )

            plot_trajectory(all_trans ,  px , py )

            print("SAVING PATH")
            cntr += 1

            with open("validation_results/goal_reach.pkl", "wb") as f:
                pickle.dump( all_trans , f )
            
            # soomething = input("press something if you dont want to save this image ... ")
            # break 

            # print("*******************")
        
            print( np.mean( all_mean ) )
    
    if args.get_metrics == 1 :

        ouptput_path = "REPLACE_WITH_LOCAL_PATH"

        GT_path = ouptput_path + "GT/"
        reults_path= ouptput_path + "Ours/"

        model = MotionDiscriminator( 3,128,3,  "cuda" , 29).to("cuda")
        ckpt = torch.load( "robot_token_model/robot_text_mot_match_new.pth" , map_location='cuda')
        model.load_state_dict( ckpt  )
        model.eval()
        model.to("cuda")

        gt_embds , pred_embds = [] , [] 
        all_gt , all_pred  =[] , []

        num_goals = os.listdir( GT_path )
        gt_mm_dist = 0
        pred_mm_dist = 0
        cntr_mm =0

        all_fid = []
        all_div = [] 
        all_mmod = []

        for _ in range(10):

            for i in tqdm( range(len( num_goals) ) ):
                gt_embds , pred_embds = [] , [] 

                folder_gt = GT_path + str(i)
                folder_pred = reults_path + str(i)

                gt_files = os.listdir( folder_gt )
                pred_files = os.listdir( folder_pred )

                for p in range( len( gt_files) ):
                    gt_file_path = folder_gt +"/" + str(p) +".pt"
                    gt_tensor = torch.load(gt_file_path)
                    gt_tensor = torch.cumsum( gt_tensor , dim =0 )
                    gt_embd = model.inference_forward( gt_tensor.unsqueeze(0) , "motion")
                    gt_embds.append( gt_embd )
                    all_gt.append( gt_embd )

                for p2 in range( len( pred_files ) ):
                    pred_file_path = folder_pred +"/" + str(p2) +".pt"
                    pred_tensor = torch.load( pred_file_path )
                    pred_tensor = torch.cumsum( pred_tensor , dim =0 )

                    pred_embd = model.inference_forward( pred_tensor.unsqueeze(0) , "motion")

                    pred_embds.append( pred_embd )
                    all_pred.append( pred_embd )
                
                gt  = torch.cat( gt_embds  , dim =0  ).unsqueeze(0)
                pred = torch.cat( pred_embds  , dim =0  ).unsqueeze(0)

                if gt.size(1) >=  10:
                    gt_mm_dist += eval_trans.calculate_multimodality( gt.detach().to("cpu") , 9 )
                    pred_mm_dist += eval_trans.calculate_multimodality( pred.detach().to("cpu") , 19 )
                    cntr_mm += 1
            
            GT = torch.cat( all_gt , dim =0  )
            Pred = torch.cat( all_pred  , dim =0 )

            gt_div = eval_trans.calculate_diversity( GT.detach().to("cpu").numpy() , 100 )
            pred_div = eval_trans.calculate_diversity( Pred.detach().to("cpu").numpy() , 100 )

            gt_mu, gt_cov  = eval_trans.calculate_activation_statistics(GT.detach().to("cpu").numpy())
            pred_mu , pred_conv = eval_trans.calculate_activation_statistics(Pred.detach().to("cpu").numpy())

            fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_conv)

            all_fid.append(fid)
            all_div.append(pred_div  )
            all_mmod.append( pred_mm_dist /cntr_mm )

        # print(  gt_mm_dist /cntr_mm , pred_mm_dist /cntr_mm , gt_div, pred_div , fid )
    
        print( np.mean( all_fid) , np.mean( all_div) , np.mean( all_mmod))
        print( np.std( all_fid)*1.95/3.16227766 , np.std( all_div)*1.95 /3.16227766, np.std( all_mmod)*1.95/3.16227766)
    

    if args.get_metrics == 2:

        num_batches = len(robot_traj_data)
        all_xy = []
        all_trajectories = []

        model = MotionDiscriminator( 3,128,3,  "cuda" , 29).to("cuda")
        ckpt = torch.load( "robot_token_model/robot_text_mot_match_new.pth" , map_location='cuda')
        model.load_state_dict( ckpt  )
        model.eval()
        model.to("cuda")

        V1 ,V2 = [] , []


        for i in tqdm( range(num_batches) ):
            batch_data = robot_traj_data[i][:, 0:192, : ].to("cuda")

            input_data = torch.zeros( batch_data.shape ).to("cuda")

            input_data[:, :, 0] = batch_data[:, :, 1]
            input_data[:, :, 1] = batch_data[:, :, 0]
            input_data[:, :, 2]  =batch_data[:, :, 2]

            reconstruction, _, _ = r_token_model.forward(input_data)
            translation = torch.cumsum( reconstruction , dim =1 )[:, 0:180, :]
            gt_embd = model.inference_forward( translation  , "motion")
            if i%2 == 0:
                V1.append(gt_embd)
            else:
                V2.append( gt_embd )
        
        GT = torch.cat( V1 , dim =0  )
        Pred = torch.cat( V2  , dim =0 )
        gt_mu, gt_cov  = eval_trans.calculate_activation_statistics(GT.detach().to("cpu").numpy())
        pred_mu , pred_conv = eval_trans.calculate_activation_statistics(Pred.detach().to("cpu").numpy())

        fid = eval_trans.calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_conv)

        print(fid)