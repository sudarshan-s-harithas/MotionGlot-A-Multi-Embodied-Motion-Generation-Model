from utils.word_vectorizer import WordVectorizer
import os
import json
import torch
import torch.optim as optim
import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
import warnings
warnings.filterwarnings('ignore')
import numpy as np  
import argparse
from tqdm import tqdm 
import random
import pickle 
import options.option_transformer as option_trans 
import post_process 
import wandb
from matplotlib import pyplot as plt 
import torch.distributed as dist
os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL" 

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from process_robot_motion import t2mt_baseline_model
from torch.nn import functional as F
from random import randrange

def split_list(input_list, n):
    # Determine the length of each sublist
    k, m = divmod(len(input_list), n)
    return [input_list[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]

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

def T2MT_Model( model_path ):
    model = t2mt_baseline_model.T2MT( GPTConfig , device_id  )
    ckpt = torch.load( model_path, map_location=args.device )
    model.load_state_dict(ckpt  )
    model.eval()
    return model.to(args.device)

class GPTConfig:
    block_size: int = 1024
    text_vocab_size: int = 128  
    traj_vocab_size: int =256
    n_layer: int = 10
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

def get_tokens():

    tokens = []

    for word in prompt.split(" "):
        idx = vocab.index(word)
        tokens.append( torch.tensor(idx).unsqueeze(0) )
    
    pad_val = []
    for  i in range( text_block_size - len(tokens) ):
        pad_val.append( torch.tensor(pad_id).unsqueeze(0) )
    
    tokens =  pad_val + tokens
    
    return torch.cat( tokens , dim = 0)
    
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

def pad_sequence( idx ):
    
    pad_len = longest_line - len(idx)-1
    pad_val = []
    for i in range(pad_len):
        pad_val.append( torch.tensor(pad_token_idx).unsqueeze(0) )
    
    idx =  [ torch.tensor( bot_id).unsqueeze(0) ] + pad_val + idx 


    return torch.cat( idx , dim =0 )

def generate():

    encoder_output = t2mt_model.m2t_encoder.forward(traj_tokens , None  )
    target_max_seq_len= 12
    beam_size = 3
    topk = 5
    print_process =True
    x = torch.tensor( [  bot_id ] ).unsqueeze(0).to(args.device )
    # Start decoding
    motion_tokens = []
    for _  in range( target_max_seq_len ):
        x_mask = t2mt_model.make_target_mask(x).to(args.device)
        pred = t2mt_model.m2t_decoder.forward(x , encoder_output, None , x_mask )
        pred = F.softmax(t2mt_model.m2t_final_linear(pred), dim=-1)
        pred = pred[:, -1, :].view(-1)
        top_k_scores, top_k_tokens = pred.topk(topk)
        print(top_k_scores )
        random_indices = torch.randint(0, len(top_k_scores), (1 ,))
        next_x = top_k_tokens[random_indices].unsqueeze(0)
        x = torch.cat( [ x , next_x ] , dim = -1 )

    text_output = x[ x < pad_token_idx ]
    result_text = ""
    for t in text_output:
        result_text += str( vocab[t] ) +" "
    
    print(result_text)

    # vel_commands = robot_tokenizer.forward_decoder(x[:, 1:].to(args.device)).squeeze(0)
    # poses = torch.cumsum( vel_commands  , dim =0 )
    # translation , angles = poses[:, 0:2] , poses[:, 2 ]
    # all_translation, all_angles = [translation.detach().to("cpu").numpy()], [angles.detach().to("cpu") ]
    # plot_motion( all_translation , all_angles )

def split_data_per_device( all_traj, all_text, device_id ):
    
    assert len(all_traj) == len(all_text)
    train_traj , valid_traj = all_traj[0:-100 ] , all_traj[-100:  ]
    train_text, valid_text = all_text[0:-100 ] ,all_text[-100:  ]

    num_train = args.batch_size* len(train_traj) //args.batch_size

    train_traj , train_text = train_traj[ 0:num_train ] , train_text[ 0:num_train ]

    start_idx = np.arange( 0, len(train_traj) , args.batch_size )
    end_idx = start_idx + args.batch_size

    batch_split_traj = [] 
    batch_split_text = []

    for i in range( len(start_idx)):
        st = start_idx[i]
        ed = end_idx[i]

        batch_data = train_traj[st:ed , : ].unsqueeze(0)
        tokens = r_token_model.encode(batch_data.to(args.device).squeeze(0)[:, 0:192, : ] )
        bom_tensor = torch.tensor( [ bom_id] ).unsqueeze(0).unsqueeze(0).to( "cuda:" + str(device_id))
        bom_tensor = bom_tensor.repeat( 1 , tokens.size(0) ,1)
        motion_val = torch.cat( [ bom_tensor , tokens.unsqueeze(0) ] , dim =-1  )
        batch_split_traj.append( motion_val )
        batch_split_text.append( train_text[st:ed , : ].unsqueeze(0) )

    
    device_split_data = split_list( batch_split_traj , 8 )
    device_split_text_data = split_list( batch_split_text , 8 )

        
    valid_tokens =  r_token_model.encode(valid_traj.to(args.device).squeeze(0)[:, 0:192, : ] )
    # device_traj_data = torch.cat(device_split_data[device_id ] , dim =0 )
    # device_text_data = torch.cat(device_split_text_data[device_id ] , dim =0 )


    device_traj_data = device_split_data[device_id ] 
    device_text_data = device_split_text_data[device_id ] 


    return device_traj_data , device_text_data , valid_tokens, valid_text

def read_raw_txt(file_path , readlines):
    f = open(file_path, "r")
    if(readlines == True ):
        content  = f.readlines()
    else:
        content = f.read()
    return content

def augment_data( all_traj, all_text ):
    all_traj2 = []
    all_text2  = []

    assert len(all_traj) == len(all_text)

    cntr =0 


    for i in range( len(all_traj) ):
        for j in range( len( all_text[i][0] ) ):
            all_traj2.append( all_traj[i].unsqueeze(0) )
            all_text2.append( all_text[i][0][j].unsqueeze(0) )
            cntr +=1
    
    return  torch.cat(all_traj2 , dim =0 ), torch.cat( all_text2 , dim =0 )

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser("generate t2mt ")
    parser.add_argument("--device", help=" set device  " , default= "cuda:0", type= str )
    parser.add_argument("--vocab_file", help=" set path t " , default= "process_robot_motion/t2mt_vocab.pkl" , type= str )
    parser.add_argument("--fine_tune_model_path", help=" path to robot tokenizer model " , default= "process_robot_motion/t2mt.pth" , type= str )
    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--text_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--batch_size", help="set the batch size to load " , default= 512 , type =int )

    args = parser.parse_args()
    r_token_model = define_robot_token_model()

    text_block_size = 11

    with open(args.vocab_file, "rb") as f:
        vocab = pickle.load(f)
    
    pad_token_idx = len(vocab) +1
    bot_id = len(vocab) + 2
    bom_id = 128
    vocab_size = len(vocab) + 3


    files = os.listdir( args.text_folder )
    longest_line = 12

    all_tokens = []
    for file in tqdm( files):
        data_path = os.path.join(  args.text_folder , file )
        lines = read_raw_txt( data_path , True )
        prmpt_tokens = []
        cntr =0 
        for line in lines:
            if cntr >= 7:
                break 
            data = line.split(".\n")[0].lower().split(" ")
            per_prompt_tokens = []
            cntr += 1 
            for word in data:
                idx = vocab.index( word )
                per_prompt_tokens.append( torch.tensor(idx).unsqueeze(0) )
            prmpt_tokens.append( pad_sequence(per_prompt_tokens ).unsqueeze(0) )
        
        prmpt_tokens = torch.cat( prmpt_tokens , dim =0   )
        all_tokens.append(  prmpt_tokens.unsqueeze(0) )
    
    all_trajectories = []

    for file in tqdm(files):
        path =os.path.join( args.pose_folder , file.split(".")[0]+ ".pt" )
        traj = torch.load( path).unsqueeze(0)

        traj2 = torch.zeros(traj.shape )

        traj2[:, :,  0] = traj[:, :, 1]
        traj2[:, :,  1] = traj[: , :,  0]
        traj2[:, :,  2] = traj[:  , :,  2]

        all_trajectories.append(traj2 )
    
    all_trajectories = torch.cat( all_trajectories , dim =0  )
    all_trajectories , all_tokens =augment_data(all_trajectories , all_tokens )

    traj_data, text_data , valid_traj, valid_text = split_data_per_device( all_trajectories , all_tokens , 0 )

    device_id = 0
    traj_tokens = valid_traj[0, : ].unsqueeze(0)
    text_gt_tokens =  valid_text[0, :] 
    text_gt_tokens = list( text_gt_tokens[ text_gt_tokens < pad_token_idx] )
    result_text = ""
    for t in text_gt_tokens:
        result_text += str( vocab[int(t)] ) +" "

    print(result_text)

    # prompt = 'a robot walks in a clockwise circle'
    # pad_id = len(vocab) + 1 
    # text_tokens = get_tokens().unsqueeze(0).to(args.device)
    # print(text_tokens)

    t2mt_model = T2MT_Model(args.fine_tune_model_path )

    generate()