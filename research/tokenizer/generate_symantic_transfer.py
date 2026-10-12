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
from matplotlib.colors import ListedColormap


def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def GPT_model( model_path ):
    model = GPT2LMHeadModel.from_pretrained(model_path ).to(args.device)
    model.eval()
    return model

def define_human_token_model():

    args2 = option_vq.get_args_parser()
    args2.dataname = "t2m"
    human_token_model =  vqvae.HumanVQVAE(args2, ## use args to define different parameters in different quantizers
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

    ckpt = torch.load(args.human_model_path, map_location= args.device)
    human_token_model.load_state_dict(ckpt['net'], strict=True)

    return human_token_model.to(args.device)

def define_robot_token_model():

    args2 = option_vq.get_args_parser()
    args2.dataname = "spot"
    args2.nb_code = 128
    # args2.down_t = 3
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
    ckpt = torch.load(args.robot_model_path, map_location=args.device )
    robot_token_model.load_state_dict(ckpt  )
    robot_token_model.eval()
    robot_token_model.to(args.device)

    return robot_token_model.to( args.device )

class StopOnWord_t2hm:
    def __init__(self, word):
        self.word = word
    def __call__(self, input_ids, scores, **kwargs):
        return 51687 in input_ids

class StopOnWord_hm2t:
    def __init__(self, word):
        self.word = word
    def __call__(self, input_ids, scores, **kwargs):
        return 51689 in input_ids

class StopOnWord_t2rm:
    def __init__(self, word):
        self.word = word
    def __call__(self, input_ids, scores, **kwargs):
        return 51688 in input_ids

class StopOnWord_rm2t:
    def __init__(self, word):
        self.word = word

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token

        return 51688 in input_ids

def extract_motion_string(motion_string , id):
    idx = re.findall(r"\d+", motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )

    idx_tensor = torch.tensor( int_idx )
    poses = h_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)
    poses  = poses*std + mean
    poses = poses.float()
    joint_coords = post_process.recover_from_ric( poses , 22 ).squeeze(0).to("cpu").detach()

    torch.save( joint_coords, "symantic_transfer_output/human/"+ str(id) +".pt" )

    data = ''.join( [f'<human_motion_id_{int(j)}>' for j in idx_tensor ] )

    return data

def extract_text( output_text ):

    text = output_text.split("<mtt>")[1]
    return text

def extract_robot_motion_from_string(  motion_string , save_name ):


    idx = re.findall(r'<robot_motion_id_(\d+)>' , motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )

    if len( int_idx) > 1:

        idx_tensor = torch.tensor( int_idx )
        vel_commands = r_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)

        save_path = "symantic_transfer_output/robot_tensor/" + save_name + ".pt"

        torch.save(vel_commands ,  save_path )
        angles = torch.cumsum( vel_commands[:, -1 ] , dim =0 )

        xy = vel_commands[:, 0:2].detach().to("cpu")

        translation = torch.cumsum(xy, dim =0 ).numpy()
        # translation = torch.tensor( [ [0 , -1 ] , [1 , 0 ] ] , dtype=torch.float32) @ translation.T
        # translation = translation.T
        # translation[:, 0] *= -1

        return translation , angles
    else:
        return np.zeros( (10,2) )

def plot_robot_motion(all_trajectories , all_angles , title , id ):

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
            for j in range(len(x) - 1):
                plt.plot(x[j:j+2], y[j:j+2], color=cmap(j / len(x)) ,linewidth=1.5 )
            q = plt.quiver(X, Y , orients[idx, 0], orients[idx, 1]  )

        save_name = 'symantic_transfer_output/robot_img/' + str(id) +".png"

        plt.savefig(save_name)

def symantic_transfer( prompt , id ):

    try:

        t2hm_prompt= "give human motion: "+  prompt + "<eop>" + "<toh>" + " \n "
        t2hm_prompt_token =  tokenizer(t2hm_prompt, return_tensors="pt"  )

        output_text  = GPT.generate(t2hm_prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
        do_sample = False ,num_beams=4,  num_return_sequences=1, stopping_criteria = [StopOnWord_t2hm("<eog>")],
            temperature=1, attention_mask =t2hm_prompt_token.attention_mask.to(args.device) )[0]

        motion_text = tokenizer.decode(output_text, skip_special_tokens=False)
        human_motion_string = extract_motion_string(motion_text , id )

        hm2t_prompt =  "give text description: " + human_motion_string  + "<eop>" + "<mtt>"

        hm2t_prompt_token =  tokenizer(hm2t_prompt, return_tensors="pt"  )

        output_text  = GPT.generate(hm2t_prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 15,
        do_sample = False, num_beams=4,   num_return_sequences=1, stopping_criteria = [StopOnWord_hm2t("<eog>")],
            temperature=1, attention_mask =hm2t_prompt_token.attention_mask.to(args.device) )[0]

        motion_text = tokenizer.decode(output_text, skip_special_tokens=True)

        text = extract_text(motion_text)
        text_clean = text.split("<eog_hmtt>")[0] #.replace("person" , "robot")

        if "person" in text_clean:
            text_clean = text_clean.replace("person", "robot")
        elif "man" in text_clean:
            text_clean = text_clean.replace("man", "robot")
        else:
            text_clean = "a robot" + text_clean

        print(text_clean)
        print("****************")

        t2rm_prompt = "give robot motion: " + "a robot walks in a clockwise circle." + "<eop>" + "<tor>" + ' \n '
        ttrm_prompt_token =  tokenizer(t2rm_prompt, return_tensors="pt"  )

        robot_motion_text = GPT.generate(ttrm_prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 15,
            do_sample = True,  num_return_sequences=1, stopping_criteria = [StopOnWord_rm2t("<eog>")],
                temperature=1, attention_mask =ttrm_prompt_token.attention_mask.to(args.device) )[0]
        robot_completed_text = tokenizer.decode( robot_motion_text , skip_special_tokens=False)
        translation , angles = extract_robot_motion_from_string(robot_completed_text , str(id) )
        plot_robot_motion( [translation] , [angles], None, str(id))
    except :
        pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser("data pre-process for body movements ")

    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )
    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--test_prompt_list", help=" path to folder with tokenizer " , default= "symantic_transfer_prompts.txt", type= str )


    args = parser.parse_args()

    print(args)

    config = GenerationConfig()
    config.stop_strings = "<eog>"
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)

    h_token_model = define_human_token_model()
    r_token_model = define_robot_token_model()
    cmap = ListedColormap(np.linspace([1, 0, 0], [0, 1, 0], 192))
    mean , std = torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") ).to(args.device) , torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") ).to(args.device)

    GPT = GPT_model(args.fine_tune_model_path)

    f = open( args.test_prompt_list , "r")

    cntr = 0

    for line in f:
        prompt = line.split("\n")[0]
        symantic_transfer(prompt , cntr )
        cntr +=1
