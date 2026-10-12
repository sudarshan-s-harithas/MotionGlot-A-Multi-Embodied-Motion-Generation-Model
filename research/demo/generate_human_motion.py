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
import cv2 
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio
import io
import gradio as gr


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

class StopOnWord:
    def __init__(self, word):
        self.word = word

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token

        return 51689 in input_ids

def extract_motion_from_string( motion_string ):
    
    idx = re.findall(r"\d+", motion_string)
    int_idx = []
    for i in range(len(idx)):
        int_idx.append( int( idx[i ]) )
    
    idx_tensor = torch.tensor( int_idx )
    poses = h_token_model.forward_decoder(idx_tensor.to(args.device)).squeeze(0)
    
    poses  = poses*std + mean 
    poses = poses.float()
    joint_coords = post_process.recover_from_ric( poses , 22 ).squeeze(0).to("cpu").detach()

    return joint_coords.numpy()

def plot_3d_motion(joints, out_name=None, title=None,
                   figsize=(10,10), fps=120, radius=4):
    # reshape and normalize
    data = joints.copy().reshape(len(joints), -1, 3)
    MINS = data.min(axis=(0,1))
    MAXS = data.max(axis=(0,1))
    data[...,1] -= MINS[1]               # put ground at y=0
    data[...,0] -= data[:,0:1,0]        # center on root x
    data[...,2] -= data[:,0:1,2]        # center on root z
    trajec = data[:,0,[0,2]]
    
    # define chains & colors
    chains = [
      [0,2,5,8,11], [0,1,4,7,10],
      [0,3,6,9,12,15],[9,14,17,19,21],[9,13,16,18,20]
    ]
    colors = ['red','blue','black','red','blue']
    
    # one‐time fig + axis setup
    fig = plt.figure(figsize=figsize, dpi=96)
    ax  = fig.add_subplot(111, projection='3d')
    if title:
        # wrap long titles
        from textwrap import wrap
        fig.suptitle('\n'.join(wrap(title, 40)), fontsize=16)
    
    def draw_plane():
        # floor plane in XZ
        verts = [
            [MINS[0]-radius, 0, MINS[2]-radius],
            [MINS[0]-radius, 0, MAXS[2]+radius],
            [MAXS[0]+radius, 0, MAXS[2]+radius],
            [MAXS[0]+radius, 0, MINS[2]-radius],
        ]
        plane = Poly3DCollection([verts], alpha=0.3)
        plane.set_facecolor((0.5,0.5,0.5,0.5))
        ax.add_collection3d(plane)
    
    def update(i):
        ax.cla()
        # axis limits
        ax.set_xlim(-radius, radius)
        ax.set_ylim(0, radius*1.5)
        ax.set_zlim(-radius, radius)
        ax.axis('off')
        
        # camera
        ax.view_init(elev=110, azim=-90)
        
        # floor + trajectory so far
        draw_plane()
        if i > 0:
            past = trajec[:i]
            ax.plot3D(past[:,0], np.zeros(i), past[:,1], linewidth=1, color='blue')
        
        # draw each limb chain
        for chain, col in zip(chains, colors):
            xs = data[i,chain,0]
            ys = data[i,chain,1]
            zs = data[i,chain,2]
            ax.plot3D(xs, ys, zs, lw=3, color=col)
        
        # save or return
        if out_name:
            fig.savefig(f"{out_name}_{i:03d}.png", dpi=96)
        else:
            fig.canvas.draw()
            h, w = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            return buf.reshape(h, w, 3)
    
    # collect frames
    frames = []
    for i in range(len(data)):
        img = update(i)
        if img is not None:
            frames.append(img)
    plt.close(fig)
    return np.stack(frames, axis=0)


def on_submit( text ):
    input_text = text 
    
    prompt= "give human motion: "+  str(text) + "<eop>" + "<toh>" + " \n "

    prompt_token =  tokenizer(prompt, return_tensors="pt"  )

    # output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
    #     do_sample = False , num_beams=4,  num_return_sequences=1, stopping_criteria = [StopOnWord("<eog>")],
    #         temperature=1, attention_mask =prompt_token.attention_mask.to(args.device) )[0]

    output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
        do_sample = True , num_beams=4,  num_return_sequences=1, 
            temperature=1, attention_mask =prompt_token.attention_mask.to(args.device) )[0]
    

    completed_text = tokenizer.decode(output_text, skip_special_tokens=False)

    poses = extract_motion_from_string(completed_text)
    # torch.save( torch.tensor(poses) , "REPLACE_WITH_LOCAL_PATH" )
    video_frames = plot_3d_motion(  poses  ,None , input_text )
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (960, 960))

    for i in range(video_frames.shape[0]):
        frame = video_frames[i]
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert from RGBA to BGR
        out.write(frame_bgr)
    
    out.release()
    # cv2.destroyAllWindows()
    return "REPLACE_WITH_LOCAL_PATH"

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")

    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "pretrained/VQVAE/net_best_fid.pth" , type= str )
    parser.add_argument("--device", help=" set device  " , default= "cuda", type= str )
    parser.add_argument("--num_time_samples", help=" set a dummy integer " , default= 64 , type= int )
    parser.add_argument("--fine_tune_model_path", help=" set path to fine tune model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--fine_tune_tokenizer_path", help=" path to folder with tokenizer " , default= "REPLACE_WITH_LOCAL_PATH", type= str )


    args = parser.parse_args()

    print(args)

    config = GenerationConfig()
    config.stop_strings = "<eog>"
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)
    config = GenerationConfig()
    config.stop_strings = "<eog>"
    tokenizer = get_tokenizer(args.fine_tune_tokenizer_path)
    h_token_model = define_human_token_model()


    mean , std = torch.from_numpy( np.load("checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy") ).to(args.device) , torch.from_numpy( np.load("checkpoints/t2m/Comp_v6_KLD005/meta/std.npy") ).to(args.device)
    
    GPT = GPT_model(args.fine_tune_model_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (960, 960))

    with gr.Blocks() as demo:

        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 36px; font-weight: bold;">Text to Human Motion Generation Demo</h1>
            </div>
            """
        )

        gr.Markdown("Here are a few sample prompts \n ")

        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 20px; font-weight: bold;"> Q&A with Human Motion </h1>
            </div>
            """
        )

        gr.Markdown("1. Show me a move in a gymnastics class \n  2.  What are the best exercise for building muscle? \n  3. Show an exercise to improve muscle strength?. \n 4. How does a person greet their friend? " )

        gr.Markdown(
            """
            <div style="text-align: center;">
                <h1 style="font-size: 20px; font-weight: bold;"> Text-2-Human Motion </h1>
            </div>
            """
        )
        gr.Markdown("1. a person is lying on the ground stands up. \n  2. a person is doing pushups \n  3. a person is walking like a chiclen. \n 4. a person lifts an object from the floor " )


        # gr.Markdown("Text to Human Motion Generation Demo")
        textbox = gr.Textbox(label="Enter prompt to generate human motion here..")
        # video_output = gr.Video(label="Output Video")

        textbox.submit(on_submit, inputs=[textbox] , outputs= [gr.Video(label="Video")] )
    
    demo.launch(share=True)



    # prompt= "give human motion: "+ "Show me a motion that captures the essence of  Input: a person lying on the ground stands up and walks in a circle.." + "<eop>" + "<toh>" + " \n "
    # prompt_token =  tokenizer(prompt, return_tensors="pt"  )

    # output_text  = GPT.generate(prompt_token.input_ids.to(args.device) , max_new_tokens=50 , min_new_tokens = 10,
    # do_sample = False , num_beams=4,  num_return_sequences=1, stopping_criteria = [StopOnWord("<eog>")],
    #     temperature=1, attention_mask =prompt_token.attention_mask.to(args.device) )[0]

    
    # h_token_model = define_human_token_model()
    # completed_text = tokenizer.decode(output_text, skip_special_tokens=False)

    # poses = extract_motion_from_string(completed_text)
    # video_frames = plot_3d_motion(  poses  ,None , prompt )

    # for i in range(video_frames.shape[0]):
    #     frame = video_frames[i]
    #     frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)  # Convert from RGBA to BGR
    #     out.write(frame_bgr)
    
    # out.release()


    # cv2.destroyAllWindows()