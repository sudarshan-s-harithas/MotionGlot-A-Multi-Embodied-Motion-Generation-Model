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
import time 


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

        return 51689 in input_ids#51405

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

def extract_motion_from_string( motion_string ):
    
    idx = re.findall(r"\d+", motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )
    
    idx_tensor = torch.tensor( int_idx )
    poses = h_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)

    print(poses.shape )
    
    poses  = poses*std + mean 
    poses = poses.float()
    print(poses.shape)
    joint_coords = post_process.recover_from_ric( poses , 22 ).squeeze(0).to("cpu").detach()

    torch.save( joint_coords, "0.pt")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")

    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )


    args = parser.parse_args()

    print(args)

    config = GenerationConfig()
    config.stop_strings = "<eog>"
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)


    mean , std = torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") ).to(args.device) , torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") ).to(args.device)
    
    GPT = GPT_model(args.fine_tune_model_path)
    prompt= "give human motion: "+ "The person stands, crosses their left leg in front of the right, lowers themselves until they are sitting with both hands on the floor, and then stands back up, uncrossing their legs." + "<eop>" + "<toh>" + " \n "
    # prompt= "give human motion: "+ "A person is lying on the ground, stands up and walks in a circle." + "<eop>" + "<toh>" + " \n "
    # prompt = "give human motion: a person does a backflip." + "<eop>" + "<toh>" + " \n "
    # prompt = "give human motion: Show me a motion that captures the essence of  Input: a person does pushups." + "<eop>" + "<toh>" + " \n "
    # prompt = "give human motion: a person walks in a clockwise circle." + "<eop>" + "<toh>" + " \n "
    # prompt = "give human motion: a person walks forward then steps onto an object before stepping back down again around and going back again." + "<eop>" + "<toh>" + " \n "
    prompt = "give human motion: How does a person practice in a gymnastics class?" + "<eop>" + "<toh>" + " \n "

    st = time.time()
    prompt_token =  tokenizer(prompt, return_tensors="pt"  )

    output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 20,
    do_sample =False ,num_beams=4  , num_return_sequences=1, 
        temperature=1, attention_mask =prompt_token.attention_mask.to(args.device) )[0]
    
    

    h_token_model = define_human_token_model()
    completed_text = tokenizer.decode(output_text, skip_special_tokens=False)

    ed= time.time()

    print( ed- st)

    print(completed_text)

    extract_motion_from_string(completed_text)