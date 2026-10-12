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
import post_process  
from transformers import GenerationConfig
from tqdm import tqdm 
import evaluate 
import spacy
import nltk
import string
import numpy as np 
from models.evaluator_wrapper import EvaluatorModelWrapper
from dataset import dataset_Egpt
from dataset import dataset_qa 
import models.vqvae as vqvae
from utils.word_vectorizer import WordVectorizer
import utils.eval_trans as eval_trans
from utils.eval_trans import compute_metrics
from options.get_eval_option import get_opt



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
    human_token_model.eval()

    return human_token_model.to(args.device)

def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer 

def GPT_model( model_path ):
    model = GPT2LMHeadModel.from_pretrained(model_path ).to(args.device)
    model.eval()
    return model 


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser("Evaluate Text 2 Motion  ")
    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_data_path", help=" set path to fine tune data " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )
    parser.add_argument("--test_file" , help="path to humanml test set" , default="REPLACE_WITH_LOCAL_PATH") 
    parser.add_argument("--text_folder", help="point to the folder with original text prompts" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--gt_data_folder" , help="path to the GT motion dataset" ,default="REPLACE_WITH_LOCAL_PATH" )
    parser.add_argument("--dataname" , help="set to 't2m" ,default="t2m" )

    args = parser.parse_args()

    print(args)


    config = GenerationConfig()
    config.stop_strings = "<eog>"

    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)
    h_token_model = define_human_token_model()
    GPT = GPT_model(args.fine_tune_model_path)


    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_loader = dataset_Egpt.DATALoader(args.dataname, True , 32 , w_vectorizer)
    # test_loader = dataset_qa.DATALoader(args.dataname, True , 32 , w_vectorizer)

    dataset_opt_path = 'REPLACE_WITH_LOCAL_PATH'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)



    # for i in range(20):
    #     div,fid,r_prec, mm_dist = compute_metrics( test_loader , tokenizer, GPT, h_token_model , args.device  , eval_wrapper)

    
    for i in range(20):
            div, fid, r_prec, mm_dist = compute_metrics(test_loader, tokenizer, GPT, h_token_model, args.device, eval_wrapper)
            
            # Format the metrics as a line of text
            metrics_line = f"Iteration {i+10}: div={div}, fid={fid}, r_prec={r_prec}, mm_dist={mm_dist}\n"
            with open('metrics_output.txt', 'a') as file:
                file.write(metrics_line)
            file.close()