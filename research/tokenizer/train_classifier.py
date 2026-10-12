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
from process_robot_motion.motion_features import MotionDiscriminator


def motion_model():

    MD = MotionDiscriminator( 3,128,3,  accelerator.device , 29)
    return MD.to(accelerator.device)


def train(model , train_data , train_label , valid_data , valid_label , neg_samples ):
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    num_batches = len(train_data)
    criterion = torch.nn.L1Loss() #torch.nn.CrossEntropyLoss()

    torch.manual_seed(458)

    print(train_data.shape )

    model , optimizer = accelerator.prepare(model , optimizer  )

    for eph in tqdm(range(args.num_epochs)):
        TotalLoss1 , TotalLoss2 = 0,0 
        for i in range(num_batches ):
            optimizer.zero_grad()
            
            input_data = train_data[i].to(accelerator.device )            
            input_data2 = torch.zeros( input_data.shape ).to(accelerator.device)
            
            input_data2[:, :, 0] = input_data[:, :, 1]
            input_data2[:, :, 1] = input_data[:, :, 0]
            input_data2[:, :, 2] = input_data[:, :, 2]

            input_data2 = torch.cumsum( input_data2 , dim = 1 )

            targets = train_label[i].to(accelerator.device)

            concept_id = torch.argwhere( targets )

            mot_pred = model.forward( input_data2[:, 0:180, : ] , "motion")

            pos_pred = model.forward( train_label[i].to(accelerator.device) , "text" ) 


            neg_id = random.randint(0, 14)
            neg_pred = model.forward( neg_samples[i][:, neg_id , : ].to(accelerator.device) , "text" ) 



            loss_pos =   torch.linalg.norm(mot_pred - pos_pred  , dim =-1 ) 
            loss_neg =   torch.linalg.norm( mot_pred - neg_pred , dim =-1 ) 


            loss = torch.mean( torch.maximum( loss_pos + 0.75 -  loss_neg , torch.tensor(0.0) ) )


            accelerator.backward(loss)
            accelerator.wait_for_everyone()
            optimizer.step()  
            optimizer.zero_grad()
            accelerator.wait_for_everyone()

            TotalLoss1 += loss

            # loss1 = criterion( pred, input_data )
            # loss2 = criterion( pred[concept_id] , targets[concept_id] )

            # loss = loss1 #+ loss2 

            # accelerator.backward(loss)
            # # for name, param in model.named_parameters():
            # #     if param.grad is None:
            # #         print(name)
            # accelerator.wait_for_everyone()
            # optimizer.step()  
            # optimizer.zero_grad()
            # accelerator.wait_for_everyone()

            # TotalLoss1 += loss1
            # TotalLoss2 += loss2

        accelerator.log({"loss1":TotalLoss1 / num_batches })
        # accelerator.log({"loss2":TotalLoss2 / num_batches })
        if(accelerator.is_main_process and eph %100 ==0 ):
            model.eval()
            with torch.no_grad():
                model_save_path = "robot_token_model/robot_text_mot_match_new2.pth"
                model_1 = accelerator.get_state_dict(model)
                torch.save(model_1,model_save_path)
            model.train()


            # input_data = valid_data.to("cuda")
            # input_data2 = torch.zeros( input_data.shape ).to(accelerator.device)
            
            # input_data2[:, :, 0] = input_data[:, :, 1]
            # input_data2[:, :, 1] = input_data[:, :, 0]
            # input_data2[:, :, 2] = input_data[:, :, 2]


            # model.eval()
            # with torch.no_grad():
            #     logits = model.forward( input_data2 )
            #     valid_loss = criterion( logits, input_data2.to("cuda") )
            #     model_save_path = "robot_token_model/classifier.pth"
            #     model_1 = accelerator.get_state_dict(model)
            #     torch.save(model_1,model_save_path)
            #     accelerator.log({"valid_loss":valid_loss })
            # model.train()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data loader for robot trajectory tokenizers ")
    parser.add_argument("--robot_pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--num_time_samples", help="num time samples" , default= 260 , type =int )
    parser.add_argument("--batch_size", help="set the batch size to load " , default= 512 , type =int )
    parser.add_argument("--num_epochs", help="set the number of epochs" , default= 8000 , type =int )
    parser.add_argument("--lr", help="learning rate" , default= 3e-3 , type=float)
    parser.add_argument("--start_wandb", help="start wandb " , default=1 , type =int )
    parser.add_argument("--robot_prompt_folder", help="path to folder with prompts ", default= "REPLACE_WITH_LOCAL_PATH" ,type =str)

    args = parser.parse_args()

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
    
    args.device = str(accelerator.device )

    model = motion_model() 

    if args.device == "cuda":
        device_id = 0 
    else:
        device_id = int(str( args.device).split(":")[1] )


    train_data, train_text , train_neg_samples,  valid_data , valid_text = post_process.load_data_action_classifier( args.robot_pose_folder , args.robot_prompt_folder , args.batch_size , device_id )


    train_data = train_data[:,  :, 0:192, :]
    # valid_data = valid_data[ :, 0:192, : ]

    # print(train_data.shape, valid_data.shape )


    train(model ,train_data , train_text ,  valid_data , valid_text , train_neg_samples)
