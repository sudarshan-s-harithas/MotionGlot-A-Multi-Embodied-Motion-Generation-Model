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
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
import torch 
import numpy as np  
import argparse
import os 
from tqdm import tqdm 
import random
import pickle 
from transformers import GPT2Tokenizer
import configparser 
from os.path import join as pjoin
from dataset import dataset_tokenize
import options.option_transformer as option_trans 
import post_process

def read_raw_txt(file_path , readlines):
    f = open(file_path, "r")
    if(readlines == True ):
        content  = f.readlines()
    else:
        content = f.read()
    return content

def get_all_file_ids():
    
    file_ids =[]

    file_id_idx_corresp = {}

    data = read_raw_txt(args.all_file_names , True )
    for i in range(len(data)):
        ### extract the file_id and remove the preceeding white space 
        d2 = data[i].split(".")[1].replace(" ", "")
        file_ids.append(d2)
        file_id_idx_corresp[d2] = i

    return file_ids , file_id_idx_corresp

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

    print ('loading checkpoint from {}'.format(args.human_model_path))
    ckpt = torch.load(args.human_model_path, map_location='cuda')
    human_token_model.load_state_dict(ckpt['net'], strict=True)
    human_token_model.eval()
    human_token_model.to(args.device)

    return human_token_model.to("cuda")

def define_robot_token_model():
    
    args2 = option_vq.get_args_parser()
    args2.dataname = "spot"
    args2.nb_code = 32
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
    ckpt = torch.load(args.robot_model_path, map_location='cuda')
    robot_token_model.load_state_dict(ckpt  )
    robot_token_model.eval()
    robot_token_model.to(args.device)

    return robot_token_model.to("cuda")

def tokenize_human_motion(data_loader  ):
    
    motion_tokens =[]
    all_names = []
    
    for batch in tqdm(data_loader ):
        pose, name = batch
        tokens = h_token_model.encode(pose.to(args.device))
        motion_tokens.append( tokens.squeeze(0))
        all_names.append( name[0])

    return motion_tokens , all_names

def define_vocabulary():
    
    config = configparser.ConfigParser()
    config.read("quantizer_config.ini")
    
    num_human_motion_tokens = config.getint("CodeBookParms" , "num_human_codes" )
    num_robot_motion_tokens = config.getint("CodeBookParms" , "num_robot_codes" )
    num_goals = 600

    tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")

    ### add motion tokens 
    tokenizer.add_tokens(
            [f"<human_motion_id_{i}>" for i in range(num_human_motion_tokens  ) ] )
    tokenizer.add_tokens(
            [f"<robot_motion_id_{i}>" for i in range(num_robot_motion_tokens  ) ] )
    tokenizer.add_tokens(
            [f"<goal_id_{i}>" for i in range(num_goals  ) ] )

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_bos_token = True
    
    ## add meta tokens

    tokenizer.add_tokens( "<eop>" , special_tokens=True ) ## end of prompt token
    tokenizer.add_tokens( "<toh>" , special_tokens=True ) ## translate to human token
    tokenizer.add_tokens( "<tor>" , special_tokens=True ) ## translate to robot token
    tokenizer.add_tokens( "<tog>" , special_tokens=True ) ## translate to goal
    tokenizer.add_tokens( "<eog>" , special_tokens=True ) ## end of generation token

    VOCAB = {}

    VOCAB['language_id_range'] = [ 0, tokenizer.vocab_size ]
    VOCAB['eos_id'] = tokenizer.eos_token # - 1 

    VOCAB['total_vocab_size'] = len(tokenizer) #VOCAB['motion_id_range'][1] +1 
    VOCAB['block_size'] = config.getint("EncoderParms" , "block_size" )
    VOCAB['eop_id'] = tokenizer.convert_tokens_to_ids("<eop>" )
    VOCAB['eop_char'] = "<eop>"
    VOCAB['toh_char'] = "<toh>"
    VOCAB['tor_char'] = "<tor>"
    VOCAB['tog_char'] = "<tog>"
    VOCAB['eog_char'] = "<eog>"

    VOCAB['num_h_codes'] = num_human_motion_tokens
    VOCAB['h_feat_dim'] = config.getint("EncoderParms" , "human_feat_dim" )

    return tokenizer , VOCAB 

def create_human_motion_strings( motion_idx ):
    
    num_seqs = len(motion_idx )
    motion_string = []

    for i in tqdm(range(num_seqs)):
        
        data = motion_idx[i]
        if data == []:
            string_out = []
        else:

            data2 = ''.join( [f'<human_motion_id_{int(j)}>' for j in motion_idx[i] ] )
            # ids, mask = tokenize_string(data2)
            string_out = data2

        motion_string.append( string_out )

    return motion_string  

def read_human_text_prompts( file_ids , toh , corresp  , train_test):
    
    all_text_files = os.listdir( os.path.join( args.text_folder ) )
    text_prompts = []

    gpt_4_1 = load_gpt_prompts( args.new_prompts_file )
    gpt_4_2 = load_gpt_prompts( args.gpt_prompts2 )

    all_text_prompts = []

    prefixes = read_raw_txt( args.prefix_path , True )

    command = "give human motion: "

    total_new_prmpt_counter = 0

    if( train_test == "train" ):
    
        for i in tqdm( range( len(file_ids) ) ):
            id = file_ids[i]
            txt_file_name = file_ids[i] +".txt"
            file_path = os.path.join( args.text_folder , txt_file_name )
            lines = read_raw_txt(file_path , True ) 
            
            # prompt = data.split("#")#[0]
            per_file_prompts = []
            for line in lines:
                prompt = line.split("#")[0]
                # per_file_prompts.append( command + prompt)
                
                for k in range(args.augment_time):
                    new_prompt = subsitute_prefix( prompt , prefixes )
                    per_file_prompts.append( command + new_prompt )

            line_num = get_line_number(id , corresp)

            if line_num != None: 

                if line_num < len( gpt_4_1 ):
                    prompt = gpt_4_1[line_num].split("\n")[0]
                    per_file_prompts.append( command + prompt )
                    total_new_prmpt_counter +=1 
                
                if line_num < len( gpt_4_2):
                    prompt = gpt_4_2[line_num].split("\n")[0]
                    per_file_prompts.append( command + prompt)
                    total_new_prmpt_counter +=1 
            
            all_text_prompts.append(per_file_prompts)
    
    if( train_test == "test" ):

        for i in tqdm( range( len(file_ids) ) ):
            per_file_prompts = []    
            id = file_ids[i]
            txt_file_name = file_ids[i] +".txt"
            file_path = os.path.join( args.text_folder , txt_file_name )
            lines = read_raw_txt(file_path , True ) 
            prompt = lines[0].split("#")[0]
            prompt = subsitute_prefix(prompt , prefixes )
            per_file_prompts.append(command + prompt)
            all_text_prompts.append(per_file_prompts)    
        
    return all_text_prompts

def load_gpt_prompts(file_name):
    
    data = read_raw_txt( file_name , True )
    GPT_prompts = []
    for i in tqdm( range(len(data))) :
        new_prompt = data[i]
        GPT_prompts.append(new_prompt)
    
    return GPT_prompts

def subsitute_prefix( prompt, all_prefix ):
    
    prefix = random.choice( all_prefix )

    prefix = prefix.split("\n")[0]

    prfix_updated = prefix.replace("<Caption_Placeholder>",  prompt)
    prfix_updated = prfix_updated.replace("..", ".")

    return prfix_updated  

def get_line_number( key, dict ):
    
    return dict.get(key, None)

def create_unified_stream( motion_strings, text_strings , first_ouput_token , eog_token ):
    
    assert len(motion_strings) == len(text_strings)

    correspondance_data = []

    for i in tqdm( range( len(motion_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_output = motion_strings[i]

            num_text_prompts = len(  text_strings[i] )
            prompts = text_strings[i]
            for k in range( num_text_prompts):
                text_input = prompts[k]

                data = text_input + vocab_parms['eop_char'] + first_ouput_token + ' \n ' + motion_output + eog_token
                correspondance_data.append( data )
        
    return correspondance_data

def create_unified_m2t_stream( motion_strings, text_strings , first_ouput_token , eog_token):
    
    assert len(motion_strings) == len(text_strings)


    correspondance_data = []

    for i in tqdm( range( len(motion_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_input = motion_strings[i]
            text_output = text_strings[i]

            data = "give text description: " + motion_input + vocab_parms['eop_char'] + first_ouput_token +  ' \n ' + text_output + eog_token

            correspondance_data.append(data)

    return correspondance_data

def tokenize_string( data ):
    tokenization_output=  tokenizer(data, return_tensors="pt", 
    padding="max_length", max_length= vocab_parms['block_size']  )
    
    return tokenization_output['input_ids']  , tokenization_output['attention_mask']

def tokenize_unified_stream( unified_stream ):
    
    num_data_samples = len(unified_stream)

    train_data = []
    train_mask = []

    for i in tqdm(range(num_data_samples)):
        data_string = unified_stream[i]
        ids, mask = tokenize_string( data_string )
        
        train_data.append( ids )
        train_mask.append( mask )
    
    return torch.cat( train_data , dim =0 ) , torch.cat( train_mask, dim =0  )

def tokenize_robot_trajectories( robot_motion_data ):

    num_batches = len(robot_motion_data)
    all_tokens = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i][:, 0:192, : ].to("cuda")

        input_data = torch.zeros( batch_data.shape ).to("cuda")
        
        input_data[:, :, 0] = batch_data[:, :, 1]
        input_data[:, :, 1] = batch_data[:, :, 0]
        input_data[:, :, 2]  =batch_data[:, :, 2]

        tokens = r_token_model.encode(input_data.to(args.device))
        all_tokens.append( tokens )   
    return all_tokens 

def check_prompt_fitness( prompts_folders ):
    
    num_folders = len(prompts_folders)
    
    files = []
    cnt =0 
    for i in range(num_folders):
        all_prompts = os.listdir(prompts_folders[i]  )#.sort()

        if cnt > 0 :
            if all_prompts != past_prompt:
                return False
        
        past_prompt = all_prompts
    
    return True

def subsititute_embodiment_name( prompt ):
    
    substituted_word = "A robot"

    prompt = prompt.replace( "give human motion: ", "" )

    prompt = prompt.replace( 'a robot' , substituted_word )
    prompt = prompt.replace( 'The robot' , substituted_word)
    prompt = prompt.replace('a quadruped' , substituted_word )
    prompt = prompt.replace('A quadruped' ,substituted_word )
    prompt = prompt.replace('The quadruped', substituted_word )
    prompt = prompt.replace('the quadruped', substituted_word )

    prompt = prompt.replace('A four-legged robot', substituted_word )
    prompt = prompt.replace('The four-legged robot', substituted_word )
    prompt = prompt.replace('the four-legged robot', substituted_word )
    prompt = prompt.replace('An entity', substituted_word )
    prompt = prompt.replace('The entity', substituted_word )
    prompt = prompt.replace('A entity', substituted_word )
    prompt = prompt.replace('It', substituted_word )

    prompt = prompt.replace('moves', "walks" )
    prompt = prompt.replace('jogs', "walks" )
    prompt = prompt.replace('steps', "walks" )

    return prompt

def read_robot_text_prompts( prompts_folders , tor ):
    
    if check_prompt_fitness(prompts_folders ):

        file_names = os.listdir(prompts_folders[0]  )#.sort()

        all_prompts = []
        for file_name in file_names:
            file_id = file_name.split(".")[0]

            data = []
            for folder in robot_prompts_folder:
                path = os.path.join( folder , file_name )
                d = read_raw_txt(path , True)
                
                if file_id in exceptions:
                    data_len = len(d)
                else:
                    data_len = len(d) -1 
                
                data.append(d )

            for k in range( data_len ): 
                per_motion_prompt = []
                for j in range( len(data) ):
                    if data[j][k] != '\n':
                        edited_prompt = subsititute_embodiment_name(  data[j][k] )
                        text = "give robot motion: " + edited_prompt
                        per_motion_prompt.append(text)
                
                all_prompts.append(per_motion_prompt)
            
            
        return all_prompts

def create_robot_motion_strings( motion_idx ):
    num_seqs = len(motion_idx )
    motion_string = []

    for i in tqdm(range(num_seqs)):
        
        data = motion_idx[i]
        if data == []:
            string_out = []
        else:

            data2 = ''.join( [f'<robot_motion_id_{int(j)}>' for j in motion_idx[i] ] )
            # ids, mask = tokenize_string(data2)
            string_out = data2

        motion_string.append( string_out )

    return motion_string  

def create_unified_robot_strings( motion_strings, text_strings , first_ouput_token , eog_token ):

    correspondance_data = []

    for i in tqdm( range( len(text_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_output = motion_strings[i]

            num_text_prompts = len(  text_strings[i] )
            prompts = text_strings[i]
            for k in range( num_text_prompts):
                text_input = prompts[k]

                data = text_input + vocab_parms['eop_char'] + first_ouput_token + ' \n ' + motion_output + eog_token
                correspondance_data.append( data )
        
    return correspondance_data

def get_pixel_id( data , centers  ):

    data =  data.unsqueeze(1)
    centers = centers.unsqueeze(0)
    id =   torch.argmin( torch.abs( data - centers ) , dim =-1)
    return id 

def get_goal_tokens( robot_motion_data ):
    num_batches = len(robot_motion_data)
    all_xy = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i][:, 0:192, : ].to("cuda")

        input_data = torch.zeros( batch_data.shape ).to("cuda")
        
        input_data[:, :, 0] = batch_data[:, :, 1]
        input_data[:, :, 1] = batch_data[:, :, 0]
        input_data[:, :, 2]  =batch_data[:, :, 2]
        
        reconstruction, _, _ = r_token_model.forward(input_data)
        translation = torch.cumsum( reconstruction , dim =1 )[:, :, 0:2 ]
        all_xy.append(translation)
    
    xy = torch.cat( all_xy , dim =0 )
    goals = xy[:, -1 , : ].detach().to("cpu").numpy()

    N_c = ( grid_shape[1] - grid_shape[0] - cell_size ) // (cell_size ) 

    K = np.arange(0 , N_c+1 , 1)

    x_c = grid_shape[0] + cell_size /2  + K* (cell_size  )
    y_c = grid_shape[1] - cell_size /2  -  K* (cell_size   )

    idx = get_pixel_id( torch.from_numpy( goals[:, 0] ) , torch.from_numpy( x_c )  )
    idy = get_pixel_id( torch.from_numpy( goals[:, 1 ] ) , torch.from_numpy( y_c )  )

    idx = idx * (N_c+1) + idy 

    return idx 

def create_goal_strings( goal_idx ):
    num_goals = len(goal_idx )
    goal_string = []

    for i in tqdm(range(num_goals)):
        
        data = goal_idx[i]
        if data == []:
            string_out = []
        else:

            data2 =  '<goal_id_' + str( int( goal_idx[i] )) + '>' 
            # ids, mask = tokenize_string(data2)
            string_out = data2

        goal_string.append( string_out )

    return goal_string  

def create_unified_motion_to_goal_string( motion_strings, goal_string, first_ouput_token , eog_token ):

    correspondance_data = []

    for i in tqdm( range( len(goal_string) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_output = motion_strings[i]

            num_text_prompts = len(  goal_string[i] )
            prompts = goal_string[i]
            data = prompts + vocab_parms['eop_char'] + first_ouput_token + ' \n ' + motion_output + eog_token

            correspondance_data.append( data )
        
    return correspondance_data

def tokenize_unified_stream_common( unified_stream ):
    
    num_data_samples = len(unified_stream)
    
    data = []
    
    for i in tqdm(range(num_data_samples)):
        data_string = unified_stream[i]
        tokenization_output=  tokenizer(data_string, return_tensors="pt",  
        padding="max_length", max_length= vocab_parms['block_size']  )  
        data.append( tokenization_output )
    
    return data

def get_robot_train_test_data( r_data, r2g_data ):

    train_data = []
    test_data = []

    for i in range( len(r_data) -30) :
        train_data.append( r_data[i] )
    
    for i in range( len(r2g_data) -30) :
        train_data.append( r2g_data[i] )

    for i in range( len(r_data)-30 , len(r_data) ):
        test_data.append( r_data[i] )

    for i in range( len(r2g_data)-30 , len(r2g_data) ):
        test_data.append( r2g_data[i] )
    
    return train_data , test_data 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser("data pre-process for body movements ")
    parser.add_argument("--all_file_names", help="point to the file with the name of all files", default="REPLACE_WITH_LOCAL_PATH", type=str)
    parser.add_argument("--text_folder", help="point to the folder with original text prompts" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--human_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--robot_model_path", help=" path to human tokenizer model " , default= "REPLACE_WITH_LOCAL_PATH" , type= str )
    parser.add_argument("--new_prompts_file", help="point to the file with GPT4 prompts" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--gpt_prompts2", help="point to the file with GPT4 prompts" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--humanml_train" , help="path to humanml train set " , default="REPLACE_WITH_LOCAL_PATH" )
    parser.add_argument("--humanml_test" , help="path to humanml test set" , default="REPLACE_WITH_LOCAL_PATH") 
    parser.add_argument("--meta_dir" , help=" meta dir to load mean and std" , default= 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta' ) 
    parser.add_argument("--device" , help="set to device id" , default= "cuda") 
    parser.add_argument("--prefix_path" , help="set to device id" , default= "prefix_template.txt") 
    parser.add_argument("--augment_time" , help="set a value for data augmentation" , default= 5 ) 
    parser.add_argument("--meta_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--robot_pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--robot_prompt_folder", help="path to folder with prompts ", default="REPLACE_WITH_LOCAL_PATH" ,type =str)

    args = parser.parse_args()

    cell_size = 0.5 
    grid_shape =[ -6, 6 ]

    print(args)

    print("Defining Token Model")
    
    h_token_model = define_human_token_model()
    r_token_model = define_robot_token_model()

    file_ids , file_id_idx_corresp = get_all_file_ids()
    tokenizer , vocab_parms = define_vocabulary()

    print("Load Robot Trajectories")
    robot_traj_data , robot_file_order = post_process.load_spot_data( args.robot_pose_folder  , args.meta_path )

    print("Get Goal tokens ")
    goal_token_idx = get_goal_tokens( robot_traj_data )
    goal_strings = create_goal_strings( goal_token_idx )

    print("Tokenize Robot Trajectories ")

    robot_traj_tokens = tokenize_robot_trajectories( robot_traj_data )
    robot_traj_tokens2 = []

    for i in range( len( robot_traj_tokens  ) ):
        for j in range( len( robot_traj_tokens[i] ) ):
            robot_traj_tokens2.append( robot_traj_tokens[i][j] )
    
    robot_token_sequence = robot_traj_tokens2

    r_motion_string   = create_robot_motion_strings( robot_token_sequence )

    goal_reaching_stream = create_unified_motion_to_goal_string( r_motion_string , goal_strings , vocab_parms['tog_char'] , vocab_parms['eog_char']  )
    
    print(" Read robot motion prompts ")

    robot_prompts_folder = [ os.path.join( args.robot_prompt_folder , p) for p in  os.listdir( args.robot_prompt_folder ) ] #[ "REPLACE_WITH_LOCAL_PATH" , "REPLACE_WITH_LOCAL_PATH" ]    
    exceptions = [ 'J9' ,  'W6' ]

    
    robot_text_prompts = read_robot_text_prompts(robot_prompts_folder ,  vocab_parms['tor_char'])


    robot_t2m_stream = create_unified_robot_strings( r_motion_string  , robot_text_prompts ,vocab_parms['tor_char'] , vocab_parms['eog_char'] )

    
    print("load Human Dataset ")
    
    args2 = option_trans.get_args_parser()
    args2.dataname = "t2m"
    train_data_loader = dataset_tokenize.DATALoader(args2.dataname, False,  1, unit_length=2**args2.down_t)
    test_data_loader = dataset_tokenize.DATALoader(args2.dataname, True ,  1, unit_length=2**args2.down_t)

    print("Tokenize Human Poses  ")

    train_h_token_sequence , train_file_names = tokenize_human_motion(   train_data_loader  )
    test_h_token_sequence ,  test_file_names = tokenize_human_motion(   test_data_loader  )

    print("Crearing Human Motion Strings ")

    train_h_motion_string   = create_human_motion_strings( train_h_token_sequence )
    test_h_motion_string = create_human_motion_strings( test_h_token_sequence )


    print("Reading Human Motion Prompts")

    train_h_text_prompts = read_human_text_prompts(train_file_names , vocab_parms['toh_char'] , file_id_idx_corresp , "train" )
    test_h_text_prompts = read_human_text_prompts(test_file_names , vocab_parms['toh_char'] , file_id_idx_corresp , "test")

    print("Create a Unified text 2 Human motion sequence ... ")

    train_unified_human_stream = create_unified_stream(  train_h_motion_string , train_h_text_prompts ,  vocab_parms['toh_char'] , vocab_parms['eog_char'] )
    test_unified_human_stream = create_unified_stream( test_h_motion_string , test_h_text_prompts , vocab_parms['toh_char'] , vocab_parms['eog_char'] )

    train_h_data = tokenize_unified_stream_common(train_unified_human_stream)
    test_h_data = tokenize_unified_stream_common(test_unified_human_stream)

    train_r_data = tokenize_unified_stream_common( robot_t2m_stream )
    train_r2g_data = tokenize_unified_stream_common( goal_reaching_stream )

    robot_train_data , robot_test_data = get_robot_train_test_data( train_r_data , train_r2g_data )


    DATA = {}

    DATA['human_train_data'] = train_h_data
    DATA['robot_train_data'] = robot_train_data
    DATA['human_test_data']  = test_h_data
    DATA['robot_test_data']  = robot_test_data

    # train_h_tokens, train_h_mask = tokenize_unified_stream( train_unified_human_stream )
    # test_h_tokens, test_h_mask = tokenize_unified_stream( test_unified_human_stream )
    
    # train_r_tokens, train_r_mask = tokenize_unified_stream( robot_t2m_stream )
    # train_r2g_tokens, train_r2g_mask = tokenize_unified_stream( goal_reaching_stream )


    # test_m2t_tokens, test_m2t_mask = tokenize_unified_stream(test_m2t_unified_stream  )

    # train_data =  torch.cat([ train_h_tokens , train_r_tokens[0:-30 , : ] , train_r2g_tokens[0:-30 , : ]    ])
    # train_mask = torch.cat([ train_h_mask  , train_r_mask[0:-30 , : ] , train_r2g_mask[0:-30 , : ]  ]) 

    # valid_data= torch.cat([ test_h_tokens[0:500, : ]  , train_r_tokens[-30:, : ] ,  train_r2g_tokens[-30:, : ]  ]) 
    # valid_mask =  torch.cat([  test_h_mask[0:500, : ] , train_r_mask[-30:, : ]  , train_r2g_mask[-30:, : ]  ]) 

    # train_data =  torch.cat([   train_r2g_tokens[0:-30 , : ]  ])
    # train_mask = torch.cat([ train_r2g_mask[0:-30 , : ]  ]) 

    # valid_data= torch.cat([   train_r2g_tokens[-30:, : ]  ]) 
    # valid_mask =  torch.cat([   train_r2g_mask[-30:, : ]   ]) 

    # idx = torch.randperm(train_data.shape[0])
    # train_data , train_mask = train_data[idx] , train_mask[idx]

    '''
    DATA = {}

    # DATA['train_data']  = train_data
    # DATA['valid_data']  = valid_data
    # DATA['train_mask']  = train_mask
    # DATA['valid_mask']  = valid_mask


    human_train_data =   train_h_tokens 
    human_train_mask = train_h_mask
    robot_train_data =  torch.cat([ train_r_tokens[0:-30 , : ] , train_r2g_tokens[0:-30 , : ]    ])
    robot_train_mask = torch.cat([  train_r_mask[0:-30 , : ] , train_r2g_mask[0:-30 , : ]  ]) 

    robot_valid_data = torch.cat([  train_r_tokens[-30:, : ] ,  train_r2g_tokens[-30:, : ]  ]) 
    robot_valid_mask = torch.cat([  train_r_mask[-30:, : ]  , train_r2g_mask[-30:, : ]  ]) 


    DATA['human_train_data'] = human_train_data
    DATA['human_train_mask'] = human_train_mask
    DATA['robot_train_data'] = robot_train_data
    DATA['robot_train_mask'] = robot_train_mask

    DATA['human_valid_data'] = test_h_tokens[0:500, : ] 
    DATA['human_valid_mask'] = test_h_mask[0:500, : ]
    DATA['robot_valid_data'] = robot_valid_data
    DATA['robot_valid_mask'] = robot_valid_mask
    
    DATA['vocab_parms'] = vocab_parms
    '''

    tokenizer.save_pretrained("motion_tokenizers/common_motion_tokenizer_256_small_all")

    with open("train_data/human_robot_seperate_data.pkl", "wb") as f:
        pickle.dump(DATA , f)
