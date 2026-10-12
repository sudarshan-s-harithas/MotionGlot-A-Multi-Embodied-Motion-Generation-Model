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
from utils.word_vectorizer import WordVectorizer
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

    return robot_token_model

def save_results( reconstruction , input_data ):

    ouptut , input = reconstruction.detach().to("cpu") , input_data.detach().to("cpu")

    num_samples = len(ouptut)

    for i in range(num_samples):

        d_o, d_i = ouptut[i, :, 0:2] , input[i, :, 0:2 ]

        traj_out , traj_in = torch.cumsum(d_o , dim=0).numpy() , torch.cumsum(d_i , dim = 0 ).numpy()
        
        plt.figure(figsize=(8, 6))

        # Create the grid
        plt.grid(True)

        # Set the limits for the x and y axes
        plt.xlim(-5, 7)
        plt.ylim(-5, 5)

        plt.plot( traj_out[:, 0] , traj_out[:, 1 ] , '--' , color='b' )
        plt.plot( traj_in[:, 0] , traj_in[:, 1 ] , color='g' )
        save_path = "analysis/" +  str(i) +".png"

        plt.savefig(save_path)
        plt.clf()

def translation_loss( batch_data , reconstruction ):

    return torch.cumsum( batch_data  , dim =1 ) , torch.cumsum(reconstruction , dim =1 )

def train_tokenizer( traj_data , robot_tokenizer):
    
    optimizer = optim.Adam(robot_tokenizer.parameters(), lr=args.lr)
    num_batches = len(traj_data)
    criterion = torch.nn.L1Loss()

    robot_tokenizer , optimizer = accelerator.prepare(robot_tokenizer , optimizer  )
    
    if(accelerator.is_main_process):
        num_batches = num_batches 

    for eph in tqdm(range(args.num_epochs)):
        TotalLoss = 0

        for i in range(num_batches ):
            optimizer.zero_grad()

            input_data = traj_data[i][:, 0:192, : ].to(args.device)
            batch_data = torch.zeros( input_data.shape ).to(args.device)
            
            batch_data[:, :, 0] = input_data[:, :, 1]
            batch_data[:, :, 1] = input_data[:, :, 0]
            batch_data[:, :, 2]  =input_data[:, :, 2]

            reconstruction , q_loss, _ = robot_tokenizer.forward(batch_data )
            trans_in, trans_out = translation_loss( batch_data , reconstruction )
            loss = criterion(reconstruction , batch_data) + q_loss  + criterion(trans_in , trans_out)
            
            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            optimizer.step()  
            optimizer.zero_grad()
            accelerator.wait_for_everyone()

            # loss.backward()
            # optimizer.step()  
            
            TotalLoss += loss 
        accelerator.log({"loss":TotalLoss })
        
        if(accelerator.is_main_process and eph %100 ==0 ):
    
            input_data = valid_traj[:, 0:192, :].to("cuda")

            valid_data = torch.zeros( input_data.shape ).to(args.device)
            
            valid_data[:, :, 0] = input_data[:, :, 1]
            valid_data[:, :, 1] = input_data[:, :, 0]
            valid_data[:, :, 2]  =input_data[:, :, 2]

            
            robot_tokenizer.eval()
            with torch.no_grad():
                
                reconstruction , q_loss, _ = robot_tokenizer.forward(valid_data )

                save_results(reconstruction , valid_data )
                model_save_path = "robot_token_model/latest_augment.pth"
                model = accelerator.get_state_dict(robot_tokenizer)
                torch.save(model,model_save_path)

                valid_loss = criterion(reconstruction , valid_data) + q_loss 
                accelerator.log({"valid_loss":valid_loss })  
            robot_tokenizer.train()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data loader for robot trajectory tokenizers ")
    parser.add_argument("--pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--num_time_samples", help="num time samples" , default= 260 , type =int )
    parser.add_argument("--batch_size", help="set the batch size to load " , default= 512 , type =int )
    parser.add_argument("--num_epochs", help="set the number of epochs" , default= 10000 , type =int )
    parser.add_argument("--lr", help="learning rate" , default= 1e-5 , type=float)
    parser.add_argument("--meta_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--sim_pose_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--start_wandb", help="start wandb " , default=1 , type =int )

    args = parser.parse_args()

    print(args)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    if( args.start_wandb == 1):
        print("Initilized Wandb ")
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs ] , log_with="wandb" )
        accelerator.init_trackers(
            "QUANT_LAM",
            config={
            "learning_rate": args.lr,
            "architecture": "Robot VQVAE TRAINING",
            }
        )
    else:
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs ])
    
    args.device = accelerator.device

    robot_tokenizer = define_robot_token_model()
    # run = wandb.init()

    # traj_data , _  = post_process.load_spot_data( args.pose_folder  , args.meta_path )

    # post_process.load_and_augment_spot_data( args.pose_folder , args.meta_path )

    device_id = int (str(args.device).split(":")[1])

    # traj_data = post_process.load_sim_spot_data( args.sim_pose_path ,
    #  args.batch_size , device_id )

    traj_data , valid_traj = post_process.load_robot_dataset( args.pose_folder ,args.batch_size , device_id )
    


    train_tokenizer( traj_data , robot_tokenizer  )

