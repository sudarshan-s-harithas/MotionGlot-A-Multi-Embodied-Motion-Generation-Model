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
from tqdm import tqdm 
import evaluate 
import spacy
import nltk
import string
from utils.word_vectorizer import WordVectorizer
from dataset import dataset_Egpt
from evaluate import load
import numpy as np 
from bert_score import score as score_bert
from nlgmetricverse import NLGMetricverse, load_metric
from nlgeval import NLGEval
from utils.eval_trans import compute_m2t_metrics
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import post_process 


nltk.download('stopwords')
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")



def get_tokenizer( path ):
    tokenizer = GPT2Tokenizer.from_pretrained(path)
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
    human_token_model.eval()

    return human_token_model.to(args.device)

    
    num_batches = len(robot_motion_data)
    all_tokens = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i][:, 0:192, : ].to("cuda")

        tokens = r_token_model.encode(batch_data.to(args.device))[0, :]
        all_tokens.append( tokens )  

    return all_tokens 

def create_human_motion_strings( motion_idx ):
    
    num_seqs = len(motion_idx )
    motion_string = []

    for i in range(num_seqs):
        
        data = motion_idx[i]
        if data == []:
            string_out = []
        else:

            data2 = ''.join( [f'<human_motion_id_{int(j)}>' for j in motion_idx[i] ] )
            # ids, mask = tokenize_string(data2)
            string_out = "give text description: " +  data2 + "<eop>" + "<mtt>" 

        motion_string.append( string_out )

    return motion_string  

def extract_text( output_text ):
    
    text = output_text.split("<mtt>")[1]
    return text

class StopOnWord_hm2t:
    def __init__(self, word):
        self.word = word
    def __call__(self, input_ids, scores, **kwargs):
        return eog_id in input_ids

def remove_special_chars( output_idx1 ):
    
    try: 

        output_motion = []

        for out in output_idx1:
            mtt_idx = (out == mtt_id).nonzero(as_tuple=True)[0][0].item()   
            eog_idx = (out == eog_id).nonzero(as_tuple=True)[0][0].item() 
            
            motion = out[ mtt_idx+1: eog_idx]
            output_motion.append(motion)

        return output_motion 
    except:
        print("EOG not found")
        return output_idx1

def generate_text( motion_data ):

    motion_tokens = h_token_model.encode(motion_data.to(args.device))
    train_h_motion_string   = create_human_motion_strings( motion_tokens )
    all_outputs = []

    for k in range( len(train_h_motion_string) ):

        motion_prompt = train_h_motion_string[k]
        hm2t_prompt_token =  tokenizer(motion_prompt , return_tensors="pt"  )

        output_text  = GPT.generate(hm2t_prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 15,
        do_sample = True,  num_return_sequences=1, stopping_criteria = [StopOnWord_hm2t("<eog>")],
            temperature=1, attention_mask =hm2t_prompt_token.attention_mask.to(args.device) )[0]

        output_text = remove_special_chars([output_text])[0]
        motion_text = tokenizer.decode(output_text, skip_special_tokens=True)

        all_outputs.append( motion_text.lower() )
    
    return all_outputs

def eval_bleu( pred, gt_vals):

    # bleu = evaluate.load("bleu")
    # results = bleu.compute(predictions=pred, references=gt_vals, max_order =4 )
    # print("Bleu@4 results")
    # print(results)
    # print("Bleu@1 results")
    # results = bleu.compute(predictions=pred, references=gt_vals, max_order =1 )
    # print(results)
    # # results = bertscore.compute(predictions=pred, references=gt_vals, lang="en")
    P, R, F1 = score_bert( pred ,gt_vals , lang='en',
                              rescale_with_baseline=True,
                              idf=True,
                              verbose=False)
    print( F1.mean() )


    scores = nlg_evaluator(predictions=pred ,
                                        references= gt_vals )

    print(scores['bleu_1']['score'] , scores['bleu_4']['score'], scores['rouge']['rougeL'] , scores['cider']['score'] )

def subsititute_name( prompt ):
    
    substituted_word = "a person"

    prompt = prompt.replace( "give human motion: ", "" )
    prompt = prompt.replace("Give me a motion that captures the meaning of  Input:,", "")
    prompt = prompt.replace('A person',  substituted_word )
    prompt = prompt.replace('The person', substituted_word )
    prompt = prompt.replace('a figure', substituted_word )
    prompt = prompt.replace('A figure', substituted_word )
    prompt = prompt.replace('The figure', substituted_word )
    prompt = prompt.replace('The sketch', substituted_word )
    prompt = prompt.replace('A sketch', substituted_word )
    prompt = prompt.replace('a sketch', substituted_word )
    prompt = prompt.replace('a man', substituted_word )
    prompt = prompt.replace('The man', substituted_word )
    prompt = prompt.replace('A man', substituted_word )
    prompt = prompt.replace('stick figure', substituted_word )
    prompt = prompt.replace('this person', substituted_word )
    prompt = prompt.replace('the man', substituted_word )

    return prompt

def process_gt( gt_text ):
    all_text = []

    for text in gt_text:
        # print("-------------------")
        # print( text )
        # print(subsititute_name( text[0].lower() ) )
        
        # print("-------------------")
        all_text.append( subsititute_name( text[0].lower() ) )
        # all_text.append(  text[0].lower()  )

    return all_text
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )
    parser.add_argument("--test_file" , help="path to humanml test set" , default="REPLACE_WITH_LOCAL_PATH") 
    parser.add_argument("--text_folder", help="point to the folder with original text prompts" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--gt_data_folder" , help="path to the GT motion dataset" ,default="REPLACE_WITH_LOCAL_PATH" )
    parser.add_argument("--dataname" , help="set to 't2m" ,default="t2m" )

    args = parser.parse_args()

    print(args)
    config = GenerationConfig()
    config.stop_strings = "<eog>"

    bertscore = load("bertscore")

    # mtt_id , eog_id = 51685 , 51689
    mtt_id , eog_id = 51685 , 51691

    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)
    h_token_model = define_human_token_model()
    GPT = GPT_model(args.fine_tune_model_path)

    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    test_loader = dataset_Egpt.TestDataloader(args.dataname, True , 1 , w_vectorizer)

    all_gt = []
    all_pred = []
    cntr = 0
    metrics = [
                    load_metric("bleu", resulting_name="bleu_1", compute_kwargs={"max_order": 1}),
                    load_metric("bleu", resulting_name="bleu_4", compute_kwargs={"max_order": 4}),
                    load_metric("rouge"),
                    load_metric("cider"),
                ]
    nlg_evaluator = NLGMetricverse(metrics)

    nlg_eval = NLGEval(
            metrics_to_omit=['METEOR',
                            'EmbeddingAverageCosineSimilarity' ,
                            'SkipThoughtCS',
                            'VectorExtremaCosineSimilarity',
                            'GreedyMatchingScore']
        )

    dataset_opt_path = 'REPLACE_WITH_LOCAL_PATH'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    # compute_m2t_metrics( test_loader , tokenizer , GPT , h_token_model,
    #  "cuda" , eval_wrapper , mtt_id , eog_id ,
    #  nlg_evaluator, score_bert,wrapper_opt , w_vectorizer )
    

    mean , std = torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") )  , torch.from_numpy( np.load("REPLACE_WITH_LOCAL_PATH") ) 


    for batch in tqdm( test_loader ):
        motion, caption = batch
        poses  = motion*std + mean 
        poses = poses.float()
        joint_coords_m2t = post_process.recover_from_ric( poses , 22 ).squeeze(0).to("cpu").detach()

        torch.save( joint_coords_m2t , "0.pt")
        caption = process_gt(caption)

        motion_text = generate_text( motion )[0]
        print(motion_text)
        print(caption)
        break 


        all_gt.append( caption )
        all_pred.append( subsititute_name( motion_text.lower()) )


        # # assert len(caption) == len(motion_text)
        # # for k in range( len(caption) ):
        # #     all_gt.append(  caption[k] ) 
        # #     all_pred.append( motion_text[k]  )
        
        # # print( all_gt , all_pred )

        if cntr % 200 == 0:
            
            eval_bleu(all_pred, all_gt)
        
        cntr +=1 

   