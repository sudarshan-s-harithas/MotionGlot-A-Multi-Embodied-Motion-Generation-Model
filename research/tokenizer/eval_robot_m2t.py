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
import evaluate 
from evaluate import load
import random 
from nlgmetricverse import NLGMetricverse, load_metric
from bert_score import score as score_bert
from process_robot_motion.motion_features import MotionDiscriminator
from utils import  eval_trans

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

def GPT_model( model_path ):
    model = GPT2LMHeadModel.from_pretrained(model_path ).to(args.device)
    model.eval()
    return model

def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path )
    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

def read_raw_txt(file_path , readlines):
    f = open(file_path, "r")
    if(readlines == True ):
        content  = f.readlines()
    else:
        content = f.read()
    return content

def sort_index(lst, rev=True):
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def robot_motion_captioning( motion_idx ):
    
    num_samples = motion_idx.size(0)
    all_prompts = []
    for k in range( num_samples ):
        robot_motion_string = ''.join([f"<robot_motion_id_{i}>" for i in motion_idx[k]   ])
        prompt_rm2t = "give text description: " + robot_motion_string + "<eop>"  +  "<rmtt>"  #+ " \n "
        prompt_token2 =  tokenizer(prompt_rm2t, return_tensors="pt"  )

        all_prompts.append( prompt_token2 )
    
    return all_prompts

class StopOnWord:
    def __init__(self, word):
        self.word = word

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token

        return 51688 in input_ids

def translate_motion( motion_tkn ):

    output= GPT.generate(motion_tkn.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
            do_sample = True,   num_return_sequences=1, stopping_criteria = [StopOnWord("<eog>")],
                temperature= 1, attention_mask =motion_tkn.attention_mask.to(args.device) )
    completed_text = tokenizer.decode(output[0], skip_special_tokens=True )


    val = completed_text.split("<rmtt>")[1].split("\n")[0]
    
    return val 

def eval_bleu( pred, gt_vals):
    
    bleu = evaluate.load("bleu")
    bleu_4 = bleu.compute(predictions=pred, references=gt_vals, max_order =4 )['bleu']

    bleu_1 = bleu.compute(predictions=pred, references=gt_vals, max_order =1 )['bleu']

    P, R, F1 = score_bert( pred ,gt_vals , lang='en',
                              rescale_with_baseline=True,
                              idf=True,
                              verbose=False)
    scores = nlg_evaluator(predictions=pred ,
                                        references= gt_vals )

    return bleu_4 , bleu_1 , F1.mean() , scores['rouge']['rougeL'] , scores['cider']['score']

def measure_r_precision( back_translation , input_prmt ):
    
    negative_samples = list( np.random.choice(all_prompts  , 31 ) )
    all_sample_set = negative_samples.append( input_prmt )


    all_bleu_score = []
    for i in range( len( negative_samples ) ):

        scores = nlg_evaluator(predictions=[back_translation] ,
                                            references= [negative_samples[i] ] )
        all_bleu_score.append( scores['bleu_1']['score'] )

    top_3_id = sort_index( all_bleu_score )[:3]

    if top_3_id[0] == len(negative_samples)-1:
        return np.asarray( [1,1,1] )
    if top_3_id[1] ==len(negative_samples)-1:
        return np.asarray([0,1,1])
    if top_3_id[2] ==len(negative_samples)-1:
        return np.asarray([0,0,1])
    else:
        return np.asarray([0,0,0])
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")

    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )
    parser.add_argument("--test_prompt_list", help=" path to folder with tokenizer " , default= "test_prompts.txt", type= str )
    parser.add_argument("--robot_prompt_folder", help="path to folder with prompts ", default= "REPLACE_WITH_LOCAL_PATH" ,type =str)
    parser.add_argument("--m2t_traj_dir", help="path to folder with prompts ", default= "REPLACE_WITH_LOCAL_PATH" ,type =str)
    parser.add_argument("--compute_metrics", help="path to folder with prompts ", default= 1 ,type =int)

    args = parser.parse_args()

    print(args)

    bertscore = load("bertscore")

    metrics = [
                    load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                    load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                    load_metric("rouge"),
                    load_metric("cider"),
                ]
    nlg_evaluator = NLGMetricverse(metrics)

    r_token_model = define_robot_token_model()
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)
    GPT = GPT_model(args.fine_tune_model_path)

    f = open( args.test_prompt_list , "r")
    all_lines = f.readlines()

    all_files = os.listdir( args.robot_prompt_folder )
    all_prompts = []

    for file in all_files:
        data = read_raw_txt( os.path.join( args.robot_prompt_folder , file ) , True  )
        select_line = data[ random.randint(0, len(data)-1 ) ]
        all_prompts.append(select_line )
    
    all_traj= []

    for i  in range( len(all_lines) ):
        data =  torch.load( os.path.join( args.m2t_traj_dir , str(i) +".pt" ) )
        all_traj.append(data.unsqueeze(0))
    
    all_traj = torch.cat( all_traj, dim =0 )

    traj_token = r_token_model.encode( all_traj )

    if args.compute_metrics == 0:

        all_input_prompts = robot_motion_captioning( traj_token )
        global all_outputs
        all_outputs = []
        R_precision = 0

        for iter in range(20):

            for i in range(len(all_input_prompts)):

                back_trans = translate_motion(  all_input_prompts[i] )
                prec = measure_r_precision(back_trans , all_lines[i] )
                R_precision += prec 
                
                all_outputs.append(back_trans)
            
            path = "REPLACE_WITH_LOCAL_PATH" + str(iter) +".txt"
            
            f3 = open(path  , "a")
            f3.writelines(all_outputs)
            f3.close()

            
            b4, b1, bscore , rougel, cider = eval_bleu(all_outputs , all_lines )
            r_prec = R_precision / len(all_input_prompts) 

            metrics_line = f"Iteration {iter}: r_prec={r_prec}, blue_1 ={b1} , blue_4 = {b4}, bert_score={bscore} , rougel ={rougel}, cider ={cider}\n"

            f2 = open("robot_translation.txt" , "a")
            f2.writelines(metrics_line)
            f2.close()
            all_outputs = []
            R_precision = 0 
    
    if args.compute_metrics == 1:

        vocab_path = "process_robot_motion/t2mt_vocab_new.pkl"
        with open( vocab_path, "rb") as f:
            VOCAB = pickle.load(f)

        start_idx = len(VOCAB) +1 
        end_idx = start_idx +1 
        vocab_size = 128 
        text_data = []
        max_len = 12

        traj_dir = "REPLACE_WITH_LOCAL_PATH"

        text_file = "REPLACE_WITH_LOCAL_PATH"

        all_lines = open( text_file , "r").readlines()


        model = MotionDiscriminator( 3,128,3,  "cuda" , 29).to("cuda")
        ckpt = torch.load( "robot_token_model/robot_text_mot_match_new.pth" , map_location='cuda')
        model.load_state_dict( ckpt  )
        model.eval()
        model.to("cuda")

        traj_embds , text_embds = [] , []

        for i in range( len( all_lines) ):

            txt_des = all_lines[i]
            traj = torch.load( traj_dir + str(i) +".pt" )
            traj = torch.cumsum( traj , dim =0 ).unsqueeze(0)

            mot_pred = model.inference_forward( traj  , "motion")#[:, -1, : ]
            prmpt    = all_lines[i].split(".\n")[0]
            
            text_token = post_process.tokenize_text( prmpt , VOCAB , max_len , end_idx )
            text_token = torch.cat( text_token , dim =0 ).unsqueeze(0).to("cuda")
            text_pred = model.inference_forward(text_token , "text" )

            text_embds.append( text_pred )
            traj_embds.append( mot_pred )

        motion = torch.cat( traj_embds , dim =0 ).detach().to('cpu').numpy()
        text = torch.cat( text_embds  , dim = 0).detach().to("cpu").numpy()
        temp_R, temp_match = eval_trans.calculate_R_precision( motion , text , 3 , True )
        div = eval_trans.calculate_diversity( motion , 10 )

        print( temp_R/19 , div )
