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
from transformers import AutoTokenizer
import random 

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
    args2.nb_code = robot_codebooksize
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

def get_all_file_ids():
    
    file_ids =[]

    file_id_idx_corresp = {}

    data = read_raw_txt(args.all_file_names , True )
    for i in range(len(data)):
        ### extract the file_id and remove the preceeding white space 
        d2 = data[i].split(".")[1].replace(" ", "")
        file_ids.append(d2)
        file_id_idx_corresp[d2] = i 
    
    with open("file_id_map.pkl", "wb") as f:
        pickle.dump( file_id_idx_corresp , f)

    return file_ids , file_id_idx_corresp

def read_raw_txt(file_path , readlines):
    f = open(file_path, "r")
    if(readlines == True ):
        content  = f.readlines()
    else:
        content = f.read()
    return content

def define_vocabulary():
    
    config = configparser.ConfigParser()
    config.read("quantizer_config.ini")
    
    num_human_motion_tokens = config.getint("CodeBookParms" , "num_human_codes" )
    num_robot_motion_tokens =  robot_codebooksize #config.getint("CodeBookParms" , "num_robot_codes" )
    num_goals =  int( (grid_shape[1] - grid_shape[0] )**2 / ( cell_size**2) )

    # tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained( args.model_type )

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
    tokenizer.add_tokens( "<mtt>" , special_tokens=True ) ## motion to text token
    tokenizer.add_tokens( "<rmtt>" , special_tokens=True ) ## motion to text token

    tokenizer.add_tokens( "<trott>" , special_tokens=True ) ## indicates trotting gait
    tokenizer.add_tokens( "<gallop>" , special_tokens=True ) ## indicates gallop gait

    tokenizer.add_tokens( "<eog_tthm>" , special_tokens=True ) ## end of text to human motion token
    tokenizer.add_tokens( "<eog_ttrm>" , special_tokens=True ) ## end of text to robot motion token
    tokenizer.add_tokens( "<eog_hmtt>" , special_tokens=True ) ## end of human motion to text token
    tokenizer.add_tokens( "<eog_rmtt>" , special_tokens=True ) ## end of robot motion to text token
    tokenizer.add_tokens( "<eog_rmg>" , special_tokens=True ) ## end of robot motion for goal reaching


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
    VOCAB['mtt_char'] = "<mtt>"
    VOCAB['rmtt_char'] = "<rmtt>"

    VOCAB['trott_char'] = "<trott>"
    VOCAB['gallop_char'] = "<gallop>"


    VOCAB['eog_tthm_char'] = "<eog_tthm>"
    VOCAB['eog_ttrm_char'] = "<eog_ttrm>"
    VOCAB['eog_hmtt_char'] = "<eog_hmtt>"
    VOCAB['eog_rmtt_char'] = "<eog_rmtt>"
    VOCAB['eog_rmg_char'] = "<eog_rmg>"

    return tokenizer , VOCAB 

def load_robot_data( pose_folder , text_folder ):

    all_files = os.listdir(pose_folder)
    all_robot_traj = []
    all_robot_text = []

    for file in tqdm(all_files):
        file_name = file.split(".")[0]
        txt_path =  os.path.join(   text_folder  , file_name + ".txt" )
        txt_description = read_raw_txt( txt_path , True )

        traj = torch.load( os.path.join( pose_folder, file )).unsqueeze(0)[:, 0:192, : ]


        traj2 = torch.zeros(traj.shape )

        traj2[:, :,  0] = traj[:, :, 1]
        traj2[:, :,  1] = traj[: , :,  0]
        traj2[:, :,  2] = traj[:  , :,  2]

        all_robot_traj.append(traj2)
        all_robot_text.append( txt_description )
    
    return all_robot_traj , all_robot_text

def get_goal_tokens( robot_motion_data ):
    num_batches = len(robot_motion_data)
    all_xy = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i].to("cuda")
        
        reconstruction, _, _ = r_token_model.forward(batch_data)
        translation = torch.cumsum( reconstruction , dim =1 )[:, :, 0:2 ].detach().to("cpu")
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

def get_pixel_id( data , centers  ):
    
    data =  data.unsqueeze(1)
    centers = centers.unsqueeze(0)
    id =   torch.argmin( torch.abs( data - centers ) , dim =-1)
    return id 

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

def tokenize_robot_trajectories( robot_motion_data ):
    
    num_batches = len(robot_motion_data)
    all_tokens = []

    for i in tqdm( range(num_batches) ):
        batch_data = robot_motion_data[i][:, 0:192, : ].to("cuda")
        tokens = r_token_model.encode(batch_data.to(args.device))[0, :]
        all_tokens.append( tokens )  

    return all_tokens 

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

def get_positive_text(text_input):

    positive_words = [
        "joyfully",
        "blissfully",
        "cheerfully",
        "radiantly",
        "delightfully",
        "jubilantly",
        "serenely",
        "elatedly",
        "sunnily",
        "gratefully",
        "gleefully",
        "upliftedly",
        "contentedly",
        "vibrantly",
        "playfully",
        "merrily",
        "optimistically",
        "thrillingly",
        "enchantedly",
        "livelily"
    ]


    word = random.sample( positive_words , 1)
    start_phrase = "A robot"
    replace_text = "A robot " + word[0] 
    new_string = text_input.replace(start_phrase , replace_text)


    return new_string

def create_unified_robot_strings( motion_strings, text_strings , first_ouput_token , eog_token ):
    
    correspondance_data = []

    assert len(motion_strings) == len(text_strings)

    for i in tqdm( range( len(text_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_output = motion_strings[i]

            num_text_prompts = len(  text_strings[i] )
            prompts = text_strings[i]
            for k in range( num_text_prompts):
                text_input = prompts[k]

                data_neutral = "give robot motion: " +  text_input + vocab_parms['eop_char'] +  first_ouput_token + ' \n ' +  vocab_parms['trott_char']  +  motion_output + eog_token
                correspondance_data.append( data_neutral )
                data_positive = "give robot motion: " +  get_positive_text(text_input) + vocab_parms['eop_char'] + first_ouput_token + ' \n ' +  vocab_parms['gallop_char'] + motion_output + eog_token
                correspondance_data.append( data_positive )

    return correspondance_data

def tokenize_human_motion(data_loader , is_wham ):

    if is_wham == False:
    
        motion_tokens =[]
        all_names = []
        
        for batch in tqdm(data_loader ):
            pose, name = batch
            tokens = h_token_model.encode(pose.to(args.device))
            motion_tokens.append( tokens.squeeze(0))
            all_names.append( name[0])
    
    if is_wham == True:

        motion_tokens= []
        all_names = []

        for sample in tqdm( data_loader ) :
            tokens = h_token_model.encode( torch.tensor( sample).unsqueeze(0).to(args.device))
            motion_tokens.append( tokens.squeeze(0))

    return motion_tokens , all_names

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

def subsitute_prefix( prompt, all_prefix ):
    
    prefix = random.choice( all_prefix )

    prefix = prefix.split("\n")[0]

    prfix_updated = prefix.replace("<Caption_Placeholder>",  prompt)
    prfix_updated = prfix_updated.replace("..", ".")

    return prfix_updated  

def get_line_number( key, dict ):
    
    return dict.get(key, None)

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
                per_file_prompts.append( command + prompt)
                
                for k in range(args.augment_time):
                    new_prompt = subsitute_prefix( prompt , prefixes )
                    per_file_prompts.append( command + new_prompt )
            
            if (args.include_questions == 1 ):
                print("Including questions ")

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

def tokenize_string( data ):
    tokenization_output=  tokenizer(data, return_tensors="pt", 
    padding="max_length", max_length= vocab_parms['block_size']  )
    
    return tokenization_output['input_ids']  , tokenization_output['attention_mask']

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

def get_instruction_llama_template( start_description, user_inputs , system_outputs , first_ouput_token , eog_token  ):

    assert len(user_inputs) == len(system_outputs)

    correspondance_data = []

    for i in range( len( user_inputs ) ):

        output_string = system_outputs[i]
        prompts = user_inputs[i]

        for k in range(len(prompts) ):
            text_input = prompts[k]

            system_instruction = "<s>[INST] <<SYS>> you are a motion generator read the instruction and generate motion <</SYS>>" 

            if start_description == None:
                data = system_instruction + text_input + vocab_parms['eop_char'] + "[/INST]" +  output_string + eog_token + "</s>"
            else:
                data = system_instruction + start_description + text_input + vocab_parms['eop_char'] + "[/INST]" +  output_string + eog_token + "</s>"
            
            correspondance_data.append(data)
    
    return correspondance_data

def get_instruction_llama_template_goal_reaching( start_description, goal_string , motion_strings , first_ouput_token , eog_token  ):

    correspondance_data = []

    for i in tqdm( range( len(goal_string) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_output = motion_strings[i]

            prompts = goal_string[i]
            system_instruction = "<s>[INST] <<SYS>> you are a motion generator read the instruction and generate motion <</SYS>>"

            data = system_instruction + prompts + vocab_parms['eop_char'] + "[/INST]" +  motion_output + eog_token + "</s>" 

            correspondance_data.append( data )
        
    return correspondance_data

def remove_starter( prompt ):
    prompt = prompt.replace( "give human motion: ", "" )

    return prompt
    
def create_unified_m2t_stream( motion_strings, text_strings , first_ouput_token , eog_token):
    
    assert len(motion_strings) == len(text_strings)

    print("CREATING A UNIFIED Motion 2 text stream ")

    correspondance_data = []

    for i in tqdm( range( len(motion_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_input = motion_strings[i]

            prompts = text_strings[i]
            num_text_prompts = len(  text_strings[i] )
            for k in range( num_text_prompts):

                text_output = remove_starter( prompts[k] )

                data = "give text description: " + motion_input + vocab_parms['eop_char'] + first_ouput_token + text_output + eog_token

                correspondance_data.append(data)

    return correspondance_data

def create_unified_rm2t_stream(  motion_strings , text_strings , first_ouput_token, eog_token ):

    assert len(motion_strings) == len(text_strings)

    print("CREATING A UNIFIED robot Motion 2 text stream ")

    correspondance_data = []

    for i in tqdm( range( len(motion_strings) ) ): #

        if motion_strings[i] == [] :
            continue 

        if motion_strings[i] != []:

            motion_input = motion_strings[i]
            text_output = remove_starter( text_strings[i][0] )

            data = "give text description: " + motion_input + vocab_parms['eop_char'] + first_ouput_token + text_output + eog_token

            correspondance_data.append(data)

    return correspondance_data

def read_pickle(path):
    
    with open( path, "rb") as f:
        data = pickle.load(f)
    
    return data 

def load_wham_data( wham_path ):

    wham_data = read_pickle(wham_path)
    wham_motion, wham_prompts = wham_data[0] , wham_data[1]

    text_templates = json.load( open(args.txt_template ) )

    assert len(wham_motion) == len(wham_prompts)
    command = "give human motion: "
    
    all_wham_prompts= []
    for prompt in wham_prompts:
        K = prompt.split(".")[0]
        if K  == "walk forward":
            K = "walks forward"
        if K  == "spins on the spot":
            K = "Spins on the spot"
        if K == 'walks backward in a counter clockwise circle':
            K = 'walks backward in a counter-clockwise circle'
        if K == 'walk backward and turns left':
            K ='walks backward and turns left'
        if K  == 'walks backward and steps to the left':
            K = "walks forward and then steps to the left"
        if K  == 'walks forward and steps to the left':
            K ='walks forward and then steps to the left'
        if K  == 'walks forward and steps to the right':
            K = 'walks forward and then steps to the right'
        if K == 'walk backward and turns right':
            K = 'walks backward and turns right'
        if K == 'walks backward and steps to the right' :
            K = "walks backward and then steps to the right"
        
        prmpt = []
        prmpt.append( command + "a person " + K  )
        new_prompts= text_templates[K]
 
        for k in range( len( new_prompts ) ):
            prmpt.append( command + "a person " + new_prompts[k].lower() )
        
        all_wham_prompts.append( prmpt )
    
    
    return wham_motion , all_wham_prompts
    
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
    parser.add_argument("--augment_time" , help="set a value for data augmentation" , default= 1 ) 
    parser.add_argument("--meta_path", help="learning rate" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--robot_pose_folder", help="point to the folder with original poses" , default="REPLACE_WITH_LOCAL_PATH", type =str)
    parser.add_argument("--robot_prompt_folder", help="path to folder with prompts ", default= "REPLACE_WITH_LOCAL_PATH" ,type =str)
    parser.add_argument("--model_type", help="path to folder with prompts ", default= "openai-community/gpt2"  ,type =str ) # "openai-community/gpt2"
    parser.add_argument("--wham_data", help="path to folder with prompts ", default= "REPLACE_WITH_LOCAL_PATH"  ,type =str ) # "openai-community/gpt2"
    parser.add_argument("--txt_template", help="path to folder with prompts ", default= "process_robot_motion/text_templates.json"  ,type =str ) # "openai-community/gpt2"
    parser.add_argument("--include_questions", help="set to 1 to incldue questions ", default= 0 ,type =int ) # "openai-community/gpt2"

    args = parser.parse_args()

    cell_size = 0.5 
    grid_shape =[ -7, 7 ]

    print(args)

    print("Defining Token Model")

    robot_codebooksize = 128

    h_token_model = define_human_token_model()
    r_token_model = define_robot_token_model()

    file_ids , file_id_idx_corresp = get_all_file_ids()
    tokenizer , vocab_parms = define_vocabulary()

    # print("load WHAM data ")
    # wham_motion, wham_prompts = load_wham_data( args.wham_data)

    print("Load Robot Trajectories")
    robot_traj , robot_text_prompts  = load_robot_data( args.robot_pose_folder , args.robot_prompt_folder )

    goal_token_idx = get_goal_tokens( robot_traj )
    goal_strings = create_goal_strings( goal_token_idx )

    print("Tokenize Robot Trajectories ")
    robot_traj_tokens = tokenize_robot_trajectories( robot_traj )
    r_motion_string   = create_robot_motion_strings( robot_traj_tokens )
    
    print("load Human Dataset ")
    
    args2 = option_trans.get_args_parser()
    args2.dataname = "t2m"
    train_data_loader = dataset_tokenize.DATALoader(args2.dataname, False,  1, unit_length=2**args2.down_t)
    test_data_loader = dataset_tokenize.DATALoader(args2.dataname, True ,  1, unit_length=2**args2.down_t)

    print("Tokenize Human Poses  ")

    train_h_token_sequence , train_file_names = tokenize_human_motion(   train_data_loader , is_wham=False  )
    test_h_token_sequence ,  test_file_names = tokenize_human_motion(   test_data_loader  , is_wham=False )

    # wham_token_sequence , _ = tokenize_human_motion( wham_motion , is_wham=True)
    
    print("Crearing Human Motion Strings ")

    train_h_motion_string   = create_human_motion_strings( train_h_token_sequence )
    test_h_motion_string = create_human_motion_strings( test_h_token_sequence )

    # wham_motion_string = create_human_motion_strings( wham_token_sequence )

    print("Reading Human Motion Prompts")

    train_h_text_prompts = read_human_text_prompts(train_file_names , vocab_parms['toh_char'] , file_id_idx_corresp , "train" )
    test_h_text_prompts = read_human_text_prompts(test_file_names , vocab_parms['toh_char'] , file_id_idx_corresp , "test")

    print("Create a Unified sequences ... ")

    if (args.model_type == "openai-community/gpt2" or args.model_type == "sarvamai/sarvam-2b-v0.5" ):

        # unified_wham_human_stream = create_unified_stream(  wham_motion_string , wham_prompts ,  vocab_parms['toh_char'] , vocab_parms['eog_tthm_char'] )

        goal_reaching_stream = create_unified_motion_to_goal_string( r_motion_string , goal_strings , vocab_parms['tog_char'] , vocab_parms['eog_rmg_char']  )
        robot_t2m_stream = create_unified_robot_strings( r_motion_string  , robot_text_prompts ,vocab_parms['tor_char'] , vocab_parms['eog_ttrm_char'] )

        train_unified_human_stream = create_unified_stream(  train_h_motion_string , train_h_text_prompts ,  vocab_parms['toh_char'] , vocab_parms['eog_tthm_char'] )
        test_unified_human_stream = create_unified_stream( test_h_motion_string , test_h_text_prompts , vocab_parms['toh_char'] , vocab_parms['eog_tthm_char'] )
        

        train_human_m2t_stream = create_unified_m2t_stream( train_h_motion_string , train_h_text_prompts , vocab_parms['mtt_char'], vocab_parms['eog_hmtt_char'] )
        # test_human_m2t_stream = create_unified_m2t_stream( train_h_motion_string , train_h_text_prompts , vocab_parms['mtt_char'], vocab_parms['eog_char'] )

        train_robot_m2t_stream = create_unified_rm2t_stream(  r_motion_string ,robot_text_prompts , vocab_parms['rmtt_char'] , vocab_parms['eog_rmtt_char'] )


        train_h_tokens, train_h_mask = tokenize_unified_stream(train_unified_human_stream)
        test_h_tokens, test_h_mask = tokenize_unified_stream(test_unified_human_stream)

        train_m2t_tokens, train_m2t_mask = tokenize_unified_stream( train_human_m2t_stream )

        train_r_tokens, train_r_mask = tokenize_unified_stream( robot_t2m_stream )
        train_r2g_tokens, train_r2g_mask = tokenize_unified_stream( goal_reaching_stream )

        train_rmtt_tokens, train_rmtt_mask = tokenize_unified_stream( train_robot_m2t_stream )

        # train_wham_tokens , train_wham_mask = tokenize_unified_stream( unified_wham_human_stream )


    
    if( args.model_type  == "meta-llama/Llama-2-7b-chat-hf" ):


        goal_reaching_stream = get_instruction_llama_template_goal_reaching(None , goal_strings, r_motion_string, vocab_parms['tog_char'] , vocab_parms['eog_char'] )
        robot_t2m_stream = get_instruction_llama_template( "give robot motion" ,robot_text_prompts , r_motion_string , vocab_parms['tor_char'] , vocab_parms['eog_char'] )
        train_unified_human_stream = get_instruction_llama_template(None, train_h_text_prompts , train_h_motion_string , vocab_parms['toh_char'] , vocab_parms['eog_char'] )
        test_unified_human_stream = get_instruction_llama_template( None , test_h_text_prompts , test_h_motion_string ,   vocab_parms['toh_char'] , vocab_parms['eog_char'] )


        train_h_tokens, train_h_mask = tokenize_unified_stream(train_unified_human_stream)
        test_h_tokens, test_h_mask = tokenize_unified_stream(test_unified_human_stream)

        train_r_tokens, train_r_mask = tokenize_unified_stream( robot_t2m_stream )
        train_r2g_tokens, train_r2g_mask = tokenize_unified_stream( goal_reaching_stream )


    DATA= {}


    train_data =  torch.cat([ train_h_tokens , train_r_tokens[0:-30 , : ] , train_r2g_tokens[0:-30 , : ]  , 
    train_m2t_tokens , train_rmtt_tokens   ])
    train_mask = torch.cat([ train_h_mask  , train_r_mask[0:-30 , : ] , train_r2g_mask[0:-30 , : ] , 
    train_m2t_mask , train_rmtt_mask  ]) 

    valid_data= torch.cat([ test_h_tokens[0:500, : ]  , train_r_tokens[-30:, : ] ,  train_r2g_tokens[-30:, : ]  ]) 
    valid_mask =  torch.cat([  test_h_mask[0:500, : ] , train_r_mask[-30:, : ]  , train_r2g_mask[-30:, : ]  ]) 

    idx = torch.randperm(train_data.shape[0])
    train_data , train_mask = train_data[idx] , train_mask[idx]


    DATA['train_data']  = train_data
    DATA['valid_data']  = valid_data
    DATA['train_mask']  = train_mask
    DATA['valid_mask']  = valid_mask

    print(train_data.shape, train_mask.shape )

    # tokenizer.save_pretrained("motion_tokenizers/all_tasks_gait")

    # with open("train_data/all_tasks_gait_m2t_augment_without_qa.pkl", "wb") as f:
    #     pickle.dump(DATA , f)