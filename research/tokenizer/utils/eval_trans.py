import os

import clip
import numpy as np
import torch
from scipy import linalg
import spacy

import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from tqdm import tqdm 
import re



def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


            pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose

            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in val_loader:
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger


@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in val_loader:

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            
            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], True)
                except:
                    index_motion = torch.ones(1,1).cuda().long()

                pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def get_prompts( caption_in ):
    
    # return "give human motion: " + "Show me a motion that captures the essence of  Input:"+ caption_in + "<eop>" + "<toh>" +" \n "
    return "give human motion: " +  caption_in + "<eop>" + "<toh>" +" \n "

def generate_motion(prompt_enc, GPT ):

    
    # output_text=  GPT.generate(prompt_enc.input_ids  , max_new_tokens=50 , min_new_tokens = 10,
    # do_sample = False , num_beams=4,  num_return_sequences=1, stopping_criteria = [StopOnWord("<eog>")],
    #     temperature=1, attention_mask =prompt_enc.attention_mask  )[0]

    output_text=  GPT.generate(prompt_enc.input_ids  , max_new_tokens=50 , min_new_tokens = 10,
    do_sample =True, num_beams=1,  num_return_sequences=1,
     stopping_criteria = [StopOnWord("<eog>")],
        temperature=0.6, attention_mask =prompt_enc.attention_mask  )[0]

    return output_text
    

def extract_motion_from_string( motion_string  , h_token_model , device ):

    idx = re.findall(r"\d+", motion_string)
    # idx = re.findall(r"(?<=<human_motion_id_)\d+", motion_string)
    int_idx = []
    for i in range(len(idx)):
        if int( idx[i] ) < 512:
            int_idx.append( int( idx[i ]) )
        
    idx_tensor = torch.tensor( int_idx )
    idx_tensor = idx_tensor.unsqueeze(0)
    poses = h_token_model.forward_decoder(idx_tensor.to( device)).squeeze(0)    
    poses = poses.float()


    return poses 

class StopOnWord:
    def __init__(self, word):
        self.word = word

    def __call__(self, input_ids, scores, **kwargs):
        # Decode the last generated token
        return 51689 in input_ids

def subsititute_name( prompt ):
    
    substituted_word = "a person"

    # prompt = prompt.replace( "give human motion: ", "" )
    # prompt = prompt.replace("Give me a motion that captures the meaning of  Input:,", "")
    # prompt = prompt.replace('A person',  substituted_word )
    # prompt = prompt.replace('The person', substituted_word )
    # prompt = prompt.replace('a figure', substituted_word )
    # prompt = prompt.replace('A figure', substituted_word )
    # prompt = prompt.replace('The figure', substituted_word )
    # prompt = prompt.replace('The sketch', substituted_word )
    # prompt = prompt.replace('A sketch', substituted_word )
    # prompt = prompt.replace('a sketch', substituted_word )
    # prompt = prompt.replace('a man', substituted_word )
    # prompt = prompt.replace('The man', substituted_word )
    # prompt = prompt.replace('A man', substituted_word )
    # prompt = prompt.replace('stick figure', substituted_word )
    # prompt = prompt.replace('this person', substituted_word )
    # prompt = prompt.replace('the man', substituted_word )

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


@torch.no_grad() 
def compute_m2t_metrics( test_loader , tokenizer, GPT ,
 h_token_model , device , eval_wrapper , mtt_id , eog_id ,
  nlg_evaluator, score_bert, wrapper_opt, w_vectorizer ):
    
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    cntr = 0 
    nb_sample = 0
    opt = wrapper_opt

    all_gt = []
    all_pred= []

    nlp = spacy.load('en_core_web_sm')
    

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

        motion_tokens = h_token_model.encode(motion_data.to(device))
        train_h_motion_string   = create_human_motion_strings( motion_tokens )
        all_outputs = []

        for k in range( len(train_h_motion_string) ):

            motion_prompt = train_h_motion_string[k]
            hm2t_prompt_token =  tokenizer(motion_prompt , return_tensors="pt"  )

            output_text  = GPT.generate(hm2t_prompt_token.input_ids.to(device) , max_new_tokens=50 , min_new_tokens = 15,
            do_sample = False, num_beams=1,  num_return_sequences=1, stopping_criteria = [StopOnWord_hm2t("<eog>")],
                temperature=1, attention_mask =hm2t_prompt_token.attention_mask.to(device) )[0]

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

    def process_text( sentence):
        sentence = sentence.replace('-', '')
        doc = nlp(sentence)
        word_list = []
        pos_list = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
                word_list.append(token.lemma_)
            else:
                word_list.append(word)
            pos_list.append(token.pos_)
        return word_list, pos_list

    def get_embds( t_tokens  ):

        if len(t_tokens) < opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + t_tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * ( opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = t_tokens[: opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh  = w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = torch.tensor( np.concatenate(pos_one_hots, axis=0) ) 
        word_embeddings = torch.tensor( np.concatenate(word_embeddings, axis=0) )

        return pos_one_hots , word_embeddings ,sent_len


    m_lens = torch.ones(32,).long()
    sent_lens = torch.ones(32,).long()
    k =0 
    word_tensor = []
    pos_one_hots_tensor = []
    all_motions = torch.zeros( (32, 196 , 263 ) ).cuda()

    global_R_val = 0
    all_size = 0


    for batch in tqdm( test_loader ):
        motion, caption = batch
        original_cap = caption
        m_length = motion.size(1)

        all_motions[k:k+1 , :motion.size(1) , :] = motion 


        caption = process_gt(caption)
        motion_text = generate_text( motion )[0]

        generate_original = motion_text

        # print("*****************")
        # print(original_cap[0][0])
        # print( generate_original )
        # print("---------------")

        motion_text = subsititute_name( generate_text(motion)[0].lower() )

        word_list , pos_list = process_text(generate_original  )
        t_tokens = ['%s/%s' % (word_list[i], pos_list[i]) for i in range(len(word_list))]
        pos_one_hots , word_embeddings ,sent_len = get_embds(t_tokens)
        m_lens[k] = motion.size(1)
        sent_lens[k] = sent_len


        k += 1
        
        all_gt.append( caption )
        all_pred.append( motion_text )
        word_tensor.append( word_embeddings.unsqueeze(0) )
        pos_one_hots_tensor.append( pos_one_hots.unsqueeze(0) )

        cntr +=1

        if k % 32 == 0 and k >1 :
            len_val,order_idx  = torch.sort( sent_lens , descending =True)

            # order_idx = sorted(range(len(sent_lens)), key=lambda i: sent_lens[i], reverse=True)
            word_tensor = [word_tensor[i] for i in order_idx ]
            pos_one_hots_tensor = [pos_one_hots_tensor[i] for i in order_idx ] 
            all_motions2 = all_motions[ order_idx , :,  : ]
            # sent_lens= [sent_lens[i] for i in order_idx ]
            m_lens2 = m_lens[order_idx]

            # print(len_val)


            text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(
                        word_embs= torch.cat( word_tensor , dim =0 )  ,
                        pos_ohot=torch.cat( pos_one_hots_tensor , dim =0 )  ,
                        cap_lens= len_val,
                        motions= all_motions2,
                        m_lens= m_lens2 
                    )
            temp_R, temp_match = calculate_R_precision( 
            motion_embeddings.cpu().numpy(), text_embeddings.cpu().numpy() , top_k=3, sum_all=True)
            

            global_R_val += temp_R 
            all_size += text_embeddings.shape[0]

            print( global_R_val / (all_size  ) )

            m_lens = torch.ones(32,).long()
            sent_lens = torch.ones(32,).long()
            k =0 
            word_tensor = []
            pos_one_hots_tensor = []
            all_motions = torch.zeros( (32, 196 , 263 ) ).cuda()


        if cntr % 200 == 0:
                eval_bleu(all_pred, all_gt)
            

def interpolate_poses( pred_pose, gt_length ):

    output = torch.nn.functional.interpolate( pred_pose.unsqueeze(0).permute(0,2,1) ,scale_factor = gt_length / pred_pose.size(0) )
    return output.squeeze(0).permute(1,0) 


def compute_multimodality( prompt_enc, GPT , h_token_model , seq , eval_wrapper , 
tokenizer , device):

    repeat_times = 30

    output_text=  GPT.generate(prompt_enc.input_ids  , max_new_tokens=50 , min_new_tokens = 10,
    do_sample =True,  num_return_sequences=repeat_times , stopping_criteria = [StopOnWord("<eog>")],
        temperature=0.6, attention_mask =prompt_enc.attention_mask  ) 

    pred_pose_eval = torch.zeros((repeat_times, seq, 263 )).cuda()
    pred_len = torch.ones(repeat_times).long()

    for j in range( len(output_text) ):
        
        completed_text = tokenizer.decode(output_text[j] , skip_special_tokens=False )
        pred_pose = extract_motion_from_string(completed_text , h_token_model , device)
        
        cur_len = pred_pose.shape[0]

        pred_len[j] = min(cur_len, seq)
        pred_pose_eval[j:j+1, :cur_len] = pred_pose[  :seq, : ]
    
    em_embds = eval_wrapper.get_motion_embeddings( pred_pose_eval , pred_len )

    multi_modal_val = calculate_multimodality( em_embds.unsqueeze(0).to("cpu") , 10 )

    


    return multi_modal_val 

@torch.no_grad()   
def compute_metrics( test_loader , tokenizer, GPT_model , h_token_model , device , eval_wrapper):
    
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    cntr = 0 
    nb_sample = 0
    mm_dist = 0
    all_multimodal_val = 0
    
    for batch in tqdm( test_loader ):
        
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token , name = batch
        bs, seq = motion.shape[:2]
        nb_sample += bs 
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()
        pred_len = torch.ones(bs).long()
        
        for k in range(bs):
            # pred_pose , _, _ = h_token_model.forward( motion[k, :m_length[k] ].unsqueeze(0).to(device) )

            try:
                # print(caption[k])

                prompt = get_prompts(caption[k] )
                prompt_enc =  tokenizer(prompt, return_tensors="pt" ).to(device)

                output = generate_motion(prompt_enc , GPT_model )
                # print(output)
                # print("++++++++++++++++++++++")
                completed_text = tokenizer.decode(output, skip_special_tokens=False )
                pred_pose = extract_motion_from_string(completed_text , h_token_model , device)
                # pred_pose = interpolate_poses(pred_pose , int( m_length[k] ) ) 

                # pred_denorm = test_loader.dataset.inv_transform(int_pred_pose.detach().cpu().numpy())
                # int_pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), 22)

                # pred_denorm = test_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                # pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), 22)


                # torch.save(pred_xyz , "0.pt")
                # torch.save(int_pred_xyz , "int_0.pt")

                if torch.isnan(pred_pose).any():
                    print("NAN decoded")
                    print("-------------------------")
                    continue
            
                # print(pred_pose.shape )
                cur_len = pred_pose.shape[0]

                multi_modal_value = 0 #compute_multimodality(  prompt_enc , GPT_model , 
                # h_token_model , seq , eval_wrapper , tokenizer , device )
                all_multimodal_val += multi_modal_value

                val = min(cur_len, seq)

                if val <= 0:
                    continue 

                pred_len[k] = val
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[  :seq, : ]
            except Exception as e:
                print(e)
        
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len )
        all_dist = torch.linalg.norm(  et_pred -  em_pred  , dim =-1  ) 
        mm_dist += torch.sum( all_dist )
        motion_pred_list.append(em_pred)
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)

        R_precision += temp_R

        gt_motion = motion.to(device).float()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, gt_motion , m_length )
        motion_annotation_list.append( em )
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R


        cntr +=1 

        if(cntr% 20 == 0 ):
            
            motion_annotation_np_temp = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
            motion_pred_np_temp = torch.cat(motion_pred_list, dim=0).cpu().numpy()

            # num_sample = motion_annotation_np_temp
            print( motion_annotation_np_temp.shape  )
            
            gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np_temp)
            mu, cov= calculate_activation_statistics(motion_pred_np_temp)

            diversity_real = calculate_diversity(motion_annotation_np_temp , 300 if nb_sample > 300 else 100)
            diversity = calculate_diversity(motion_pred_np_temp, 300 if nb_sample > 300 else 100)

            fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
            
            print( diversity , fid , R_precision / nb_sample , R_precision_real /nb_sample , mm_dist / nb_sample , all_multimodal_val/ nb_sample )


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)


    return diversity , fid , R_precision / nb_sample , mm_dist / nb_sample


def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist