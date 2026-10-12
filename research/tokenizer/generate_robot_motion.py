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

        return 51688 in input_ids

def extract_motion_from_string( motion_string ):

    idx = re.findall(r'<robot_motion_id_(\d+)>' , motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )

    if len( int_idx) > 1:

        idx_tensor = torch.tensor( int_idx )
        vel_commands = r_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)

        torch.save(vel_commands ,  "robot_0.pt")
        angles = torch.cumsum( vel_commands[:, -1 ] , dim =0 )

        xy = vel_commands[:, 0:2].detach().to("cpu")

        translation = torch.cumsum(xy, dim =0 ).numpy()
        # translation = torch.tensor( [ [0 , -1 ] , [1 , 0 ] ] , dtype=torch.float32) @ translation.T
        # translation = translation.T
        # translation[:, 0] *= -1

        return translation , angles
    else:
        return torch.zeros( (10,2) ) , torch.zeros( (10,) )

# def plot_motion( all_trajectories ):

#     plt.figure(figsize=(8, 6))

#     plt.grid(True)

#     plt.xlim(-5, 7)
#     plt.ylim(-5, 5)

#     for i in range(len(all_trajectories)):

#         print(all_trajectories[i].shape )

#         plt.plot( all_trajectories[i][:, 0] , all_trajectories[i][:, 1 ] , '--' , color='b' )

#     plt.savefig('validation_results/xy_trans.png')
#     plt.clf()

def plot_motion(all_trajectories , all_angles ):

        plt.figure(figsize=(8, 6))
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Trajectory with Orientations (Quiver Plot)')
        plt.xlim( -5 ,7 )
        plt.ylim( -5 , 5)
        plt.grid(True)

        R = np.asarray( [[ 0,1,0 ], [1,0,0] , [0,0,1] ] )

        for i in range(len(all_trajectories)):

            dir_x , dir_y = torch.cos( all_angles[i] ).unsqueeze(1) , torch.sin( all_angles[i] ).unsqueeze(1)
            orients = torch.cat( [ dir_x , dir_y] , dim =-1 ).detach().to("cpu").numpy()

            orients = ( R[0:2, 0:2]@orients.T ).T

            x = all_trajectories[i][:, 0]
            y = all_trajectories[i][:, 1]
            idx = np.arange(0, len(x) , int(len(x) /10 ) ) #np.random.choice(trajectory.shape[0], 5,  replace=False )
            X  = x[idx ]
            Y = y[idx]

            plt.scatter(x[0], y[0], color='red', label='start' , s=60)
            plt.scatter(x[-1], y[-1], color='green', label='end', s=60)
            plt.plot(x, y , label="trajectory" )
            # q = plt.quiver(X, Y , orients[idx, 0], orients[idx, 1] , scale=40 )

        plt.savefig('validation_results/xy_trans.png')


if __name__ == "__main__":

    parser = argparse.ArgumentParser("data pre-process for body movements ")

    # parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    # parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    # parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    # parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    # parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )

    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )


    args = parser.parse_args()

    print(args)

    r_token_model = define_robot_token_model()
    GPT = GPT_model(args.fine_tune_model_path)
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)

    prompt= "give robot motion: "+ "A robot walks in a clockwise circle." + "<eop>" + "<tor>" + ' \n '
    prompt_token =  tokenizer(prompt, return_tensors="pt"  )

    output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
    do_sample = True,   num_return_sequences=1, stopping_criteria = [StopOnWord("<eog>")],
        temperature= 1, attention_mask =prompt_token.attention_mask.to(args.device) )

    all_translation = []
    all_angles = []

    for i in range( len(output_text)):

        completed_text = tokenizer.decode(output_text[i], skip_special_tokens=False)

        translation , angles = extract_motion_from_string(completed_text)
        all_translation.append(translation)
        all_angles.append( angles )

    plot_motion(all_translation , all_angles)