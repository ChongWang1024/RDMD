import os.path
import cv2
import logging

import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import OrderedDict
import hdf5storage
import argparse

from utils import utils_model
from utils import utils_logger
from utils import utils_sisr as sr
from utils import utils_image as util
from utils.utils_resizer import Resizer
from functools import partial
from utils import utils_restoration_model
from utils import utils_pnp as pnp
from utils.utils_dps_measurement import get_noise, get_operator
from scipy import ndimage

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
)

torch.manual_seed(1)

def run_deblur(args):

    # ----------------------------------------
    # Preparation
    # ----------------------------------------

    noise_level_img         = args.noise_level_img       # set AWGN noise level for LR image, default: 0
    noise_level_model       = noise_level_img   # set noise level of model, default: 0
    model_name              = 'diffusion_ffhq_10m' if 'ffhq' in args.testset else '256x256_diffusion_uncond'  # diffusion_ffhq_10m, 256x256_diffusion_uncond; set diffusino model
    testset_name            = args.testset    # set testing set,  'imagenet_val' | 'ffhq_val'
    num_train_timesteps     = 1000
    iter_num                = 100                # set number of sampling iterations
    iter_num_U              = 1                 # set number of inner iterations, default: 1
    skip                    = num_train_timesteps//iter_num     # skip interval

    show_img                = False             # default: False
    save_L                  = False              # save LR image
    save_E                  = args.save_img             # save estimated image
    save_LEH                = False             # save zoomed LR, E and H images
    save_progressive        = False              # save generation process
    border                  = 0

    sigma                   = max(0.001,noise_level_img)  # noise level associated with condition y
    lambda_                 = args.lambda_                 # key parameter lambda
    sub_1_analytic          = False              # use analytical solution

    log_process             = False
    ddim_sample             = False             # sampling method
    model_output_type       = 'pred_xstart'     # model output type: pred_x_prev; pred_xstart; epsilon; score
    skip_type               = 'quad'            # uniform, quad
    eta                     = 0.                # eta for ddim sampling
    zeta                    = args.zeta               
    guidance_scale          = 1.0
    tau                     = args.tau   

    # restoration prior parameter
    psi                     = 0.               # weight for diffusion model(1.0) and restoration model(0.0)
    modelSigma1             = 49                # set sigma_1, default: 49

    calc_LPIPS              = True
    use_DIY_kernel          = True
    blur_mode               = args.blur_mode          # Gaussian; motion      
    kernel_size             = 61
    kernel_std              = 3.0 if blur_mode == 'Gaussian' else 0.5

    sf                      = 1
    task_current            = 'deblur' 
    n_channels              = 3                 # fixed
    cwd                     = '' 
    model_zoo               = os.path.join(cwd, 'model_zoo')    # fixed
    testsets                = os.path.join(cwd, 'testsets')     # fixed
    # results                 = os.path.join(cwd, 'results', str(iter_num) + 'iter')      # fixed
    results                 = os.path.join(cwd, args.save_results_dir)
    result_name             = f'{testset_name}_{task_current}_{model_name}_sigma{noise_level_img}_NFE{iter_num}_eta{eta}_zeta{zeta}_lambda{lambda_}_blurmode{blur_mode}_tau{args.tau}'
    model_path              = os.path.join(model_zoo, model_name+'.pt')
    restoration_model_path  = os.path.join(model_zoo, restoration_model_name+'.pth')
    device                  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    
    # noise schedule 
    beta_start              = 0.1 / 1000
    beta_end                = 20 / 1000
    betas                   = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    betas                   = torch.from_numpy(betas).to(device)
    alphas                  = 1.0 - betas
    alphas_cumprod          = np.cumprod(alphas.cpu(), axis=0)
    sqrt_alphas_cumprod     = torch.sqrt(alphas_cumprod)
    sqrt_1m_alphas_cumprod  = torch.sqrt(1. - alphas_cumprod)
    reduced_alpha_cumprod   = torch.div(sqrt_1m_alphas_cumprod, sqrt_alphas_cumprod)        # equivalent noise sigma on image

    noise_model_t           = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_model)
    noise_model_t           = 0

    noise_inti_img          = 50 / 255
    t_start                 = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_inti_img) # start timestep of the diffusion process
    t_start                 = num_train_timesteps - 1   

    # ----------------------------------------
    # L_path, E_path, H_path
    # ----------------------------------------

    L_path = os.path.join(testsets, testset_name) # L_path, for Low-quality images
    E_path = os.path.join(results, result_name)   # E_path, for Estimated images
    util.mkdir(E_path)

    logger_name = result_name
    utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    logger = logging.getLogger(logger_name)

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model_config = dict(
            model_path=model_path,
            num_channels=128,
            num_res_blocks=1,
            attention_resolutions="16",
        ) if model_name == 'diffusion_ffhq_10m' \
        else dict(
            model_path=model_path,
            num_channels=256,
            num_res_blocks=2,
            attention_resolutions="8,16,32",
        )
    args = utils_model.create_argparser(model_config).parse_args([])
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model.eval()

    model = model.to(device)

    logger.info('model_name:{}, image sigma:{:.3f}, model sigma:{:.3f}'.format(model_name, noise_level_img, noise_level_model))
    logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, tau:{:.3f}, guidance_scale:{:.2f} '.format(eta, zeta, lambda_, tau, guidance_scale))
    logger.info('start step:{}, skip_type:{}, skip interval:{}, skipstep analytic steps:{}'.format(t_start, skip_type, skip, noise_model_t))
    logger.info('use_DIY_kernel:{}, blur mode:{}'.format(use_DIY_kernel, blur_mode))
    logger.info('Model path: {:s}'.format(model_path))
    logger.info(L_path)
    L_paths = util.get_image_paths(L_path)

    # --------------------------------
    # load kernel
    # --------------------------------

    if calc_LPIPS:
        import lpips
        loss_fn_vgg = lpips.LPIPS(net='vgg').to(device)
    

    def test_rho(lambda_=lambda_, zeta=zeta, model_output_type=model_output_type, tau=tau):
        logger.info('eta:{:.3f}, zeta:{:.3f}, lambda:{:.3f}, tau:{:.2f}, guidance_scale:{:.2f}'.format(eta, zeta, lambda_, tau, guidance_scale))
        test_results = OrderedDict()
        test_results['psnr'] = []
        test_results['ssim'] = []

        
        if calc_LPIPS:
            test_results['lpips'] = []
        for idx, img in enumerate(L_paths):
            
            model_out_type = model_output_type

            img_name, ext = os.path.splitext(os.path.basename(img))
            img_H = util.imread_uint(img, n_channels=n_channels)
            img_H = util.modcrop(img_H, 8)  # modcrop

            # --------------------------------
            # (1) get img_L
            # --------------------------------
            if blur_mode == 'Gaussian':
                degradation = get_operator(device=device, name='gaussian_blur', kernel_size=kernel_size, intensity=kernel_std)
            elif blur_mode == 'motion':
                degradation = get_operator(device=device, name='motion_blur', kernel_size=kernel_size, intensity=kernel_std)
            else:
                pass

            noiser = get_noise(name='gaussian', sigma=noise_level_img * 2)

            img_name, ext = os.path.splitext(os.path.basename(img))

            # --------------------------------
            # (2) get rhos and sigmas
            # -------------------------------- 

            sigmas = []
            sigma_ks = []
            rhos = []
            modelSigma2 = max(sf, noise_level_model*255.)
            lambda_diff = lambda_ * tau
            for i in range(num_train_timesteps):
                sigmas.append(reduced_alpha_cumprod[num_train_timesteps-1-i])
                if model_out_type == 'pred_xstart':
                    sigma_ks.append((sqrt_1m_alphas_cumprod[i]/sqrt_alphas_cumprod[i]))
                #elif model_out_type == 'pred_x_prev':
                else:
                    sigma_ks.append(torch.sqrt(betas[i]/alphas[i]))
                rhos.append(lambda_diff*(sigma**2)/(sigma_ks[i]**2))
                    
            rhos, sigmas, sigma_ks = torch.tensor(rhos).to(device), torch.tensor(sigmas).to(device), torch.tensor(sigma_ks).to(device)

            # restoration model parameter
            pnp_rhos, pnp_sigmas = pnp.get_rho_sigma(sigma=max(0.255/255., noise_level_model), iter_num=iter_num, modelSigma1=modelSigma1, modelSigma2=modelSigma2, w=1)
            pnp_sigmas = torch.tensor(pnp_sigmas).to(device)
            pnp_rhos = torch.tensor(pnp_rhos).to(device)
            
            # --------------------------------
            # (3) initialize x, and pre-calculation
            # -------------------------------

            x_gt = util.uint2tensor4(img_H).to(device)
            x_gt = x_gt * 2 - 1

            y_ = degradation.forward(x_gt)
            y = noiser(y_) / 2 + 0.5        # convert to [0, 1] for consistency

            img_L = util.tensor2single(y)
            # initial restoration model input
            x0 = (2*y-1).clone()
            x_prev_diff = (2*y-1).clone()
            x_diff = (2*y-1).clone()

            # initial diffusion model input
            t_y = utils_model.find_nearest(reduced_alpha_cumprod, 2 * noise_level_img)
            sqrt_alpha_effective = sqrt_alphas_cumprod[t_start] / sqrt_alphas_cumprod[t_y]
            x = sqrt_alpha_effective * (2*y-1) + torch.sqrt(sqrt_1m_alphas_cumprod[t_start]**2 - \
                    sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_y]**2) * torch.randn_like(y)


            # --------------------------------
            # (4) main iterations
            # --------------------------------

            progress_img = []
            # create sequence of timestep for sampling
            skip = num_train_timesteps//iter_num
            if skip_type == 'uniform':
                seq = [i*skip for i in range(iter_num)]
                if skip > 1:
                    seq.append(num_train_timesteps-1)
            elif skip_type == "quad":
                seq = np.sqrt(np.linspace(0, num_train_timesteps**2, iter_num))
                seq = [int(s) for s in list(seq)]
                seq[-1] = seq[-1] - 1
            progress_seq = seq[::max(len(seq)//10,1)]
            if progress_seq[-1] != seq[-1]:
                progress_seq.append(seq[-1])
            
            # reverse diffusion for one image from random noise
            for i in range(len(seq)):
                curr_sigma = sigmas[seq[i]].cpu().numpy()
                # time step associated with the noise level sigmas[i]
                t_i = utils_model.find_nearest(reduced_alpha_cumprod,curr_sigma)
                # skip iters
                if t_i > t_start:
                    continue
                # repeat for semantic consistence: from repaint
                for u in range(iter_num_U):

                    t_restore = utils_model.find_nearest(reduced_alpha_cumprod, pnp_sigmas[i].cpu().numpy())
                    x_restore_in = x0.clone()
                    x_restored = utils_model.model_fn(x_restore_in, noise_level=255*2*pnp_sigmas[i].cpu().numpy(), model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)

                    # calculate diffusion model results
                    x_prev_diff = x_diff
                    x_diff = utils_model.model_fn(x, noise_level=curr_sigma*255, model_out_type=model_out_type, \
                            model_diffusion=model, diffusion=diffusion, ddim_sample=ddim_sample, alphas_cumprod=alphas_cumprod)
                        
                        

                    # --------------------------------
                    # step 2, FFT
                    # --------------------------------
                    if seq[i] != seq[-1]:
                        x0 = x0.requires_grad_()

                        norm_grad, norm = utils_model.grad_and_value(operator=degradation.forward,x=x0, x_hat=x0, measurement=y*2-1)

                        lambda_red = lambda_ * (1 - tau)
                        step_size = 1 / (1 + lambda_red * noise_level_img**2 + rhos[t_i])
                        x0 = x0 - step_size * (norm_grad * norm + lambda_red * noise_level_img**2 * (x0 - (x_restored)) + (x0 - x_diff) * rhos[t_i])

                        x0 = x0.detach_()
      
                    # add noise back to t=i-1
                    if model_out_type == 'pred_xstart' and not (seq[i] == seq[-1] and u == iter_num_U-1):
                        
                        t_im1 = utils_model.find_nearest(reduced_alpha_cumprod,sigmas[seq[i+1]].cpu().numpy())
                        eps = (x - sqrt_alphas_cumprod[t_i] * x0) / sqrt_1m_alphas_cumprod[t_i]
                        # calculate \hat{\eposilon}
                        eta_sigma = eta * sqrt_1m_alphas_cumprod[t_im1] / sqrt_1m_alphas_cumprod[t_i] * torch.sqrt(betas[t_i])
                        x = sqrt_alphas_cumprod[t_im1] * x0 + np.sqrt(1-zeta) * (torch.sqrt(sqrt_1m_alphas_cumprod[t_im1]**2 - eta_sigma**2) * eps \
                                    + eta_sigma * torch.randn_like(x)) + np.sqrt(zeta) * sqrt_1m_alphas_cumprod[t_im1] * torch.randn_like(x)
                    else:
                        x = x0
                        pass
                        
                    # set back to x_t from x_{t-1}
                    if u < iter_num_U-1 and seq[i] != seq[-1]:
                        sqrt_alpha_effective = sqrt_alphas_cumprod[t_i] / sqrt_alphas_cumprod[t_im1]
                        x = sqrt_alpha_effective * x + torch.sqrt(sqrt_1m_alphas_cumprod[t_i]**2 - \
                                sqrt_alpha_effective**2 * sqrt_1m_alphas_cumprod[t_im1]**2) * torch.randn_like(x)
                            

                # save the process
                x_0 = (x_diff/2+0.5)
                if save_progressive and (seq[i] in progress_seq):
                    x_show = x_0.clone().detach().cpu().numpy()       #[0,1]
                    x_show = np.squeeze(x_show)
                    if x_show.ndim == 3:
                        x_show = np.transpose(x_show, (1, 2, 0))
                    progress_img.append(x_show)
                    if log_process:
                        logger.info('{:>4d}, steps: {:>4d}, np.max(x_show): {:.4f}, np.min(x_show): {:.4f}'.format(seq[i], t_i, np.max(x_show), np.min(x_show)))
                    
                    if show_img:
                        util.imshow(x_show)

            # --------------------------------
            # (3) img_E
            # --------------------------------

            img_E = util.tensor2uint(x_0)

            psnr = util.calculate_psnr(img_E, img_H, border=border)
            ssim = util.calculate_ssim(img_E, img_H, border=border)
            
            test_results['psnr'].append(psnr)
            test_results['ssim'].append(ssim)
            
            if calc_LPIPS:
                img_H_tensor = np.transpose(img_H, (2, 0, 1))
                img_H_tensor = torch.from_numpy(img_H_tensor)[None,:,:,:].to(device)
                img_H_tensor = img_H_tensor / 255 * 2 -1
                lpips_score = loss_fn_vgg(x_0.detach()*2-1, img_H_tensor)
                lpips_score = lpips_score.cpu().detach().numpy()[0][0][0][0]
                test_results['lpips'].append(lpips_score)
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB SSIM: {:.4f} LPIPS: {:.4f} ave LPIPS: {:.4f}'.format(idx+1, img_name+ext, psnr, ssim, lpips_score, sum(test_results['lpips']) / len(test_results['lpips'])))
            else:
                logger.info('{:->4d}--> {:>10s} PSNR: {:.4f}dB'.format(idx+1, img_name+ext, psnr))

            if n_channels == 1:
                img_H = img_H.squeeze()

            if save_E:
                util.imsave(img_E, os.path.join(E_path, img_name+'_'+model_name+ext))

            if save_progressive:
                now = datetime.now()
                current_time = now.strftime("%Y_%m_%d_%H_%M_%S")
                img_total = cv2.hconcat(progress_img)
                if show_img:
                    util.imshow(img_total,figsize=(80,4))
                util.imsave(img_total*255., os.path.join(E_path, img_name+'_sigma_{:.3f}_process_lambda_{:.3f}_{}_psnr_{:.4f}{}'.format(noise_level_img,lambda_,current_time,psnr,ext)))
                                                                            
            # --------------------------------
            # (4) img_LEH
            # --------------------------------

            if save_LEH:
                # img_L = util.single2uint(img_L)
                img_L = util.tensor2uint(y)
                img_I = cv2.resize(img_L, (sf*img_L.shape[1], sf*img_L.shape[0]), interpolation=cv2.INTER_NEAREST)
                img_I[:img_L.shape[0], :img_L.shape[1], :] = img_L
                util.imshow(np.concatenate([img_I, img_E, img_H], axis=1), title='LR / Recovered / Ground-truth') if show_img else None
                util.imsave(np.concatenate([img_I, img_E, img_H], axis=1), os.path.join(E_path, img_name+'_LEH'+ext))

            if save_L:
                util.imsave(util.single2uint(img_L), os.path.join('baseline_results/measurements/deblur_motion', img_name+'_LR'+ext))
        
        # --------------------------------
        # Average PSNR and LPIPS
        # --------------------------------

        ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
        logger.info('------> Average PSNR of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_psnr))

        ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
        logger.info('------> Average SSIM of ({}), sigma: ({:.3f}): {:.4f} dB'.format(testset_name, noise_level_model, ave_ssim))


        if calc_LPIPS:
            ave_lpips = sum(test_results['lpips']) / len(test_results['lpips'])
            logger.info('------> Average LPIPS of ({}) sigma: ({:.3f}): {:.4f}'.format(testset_name, noise_level_model, ave_lpips))

        return ave_psnr, ave_ssim, ave_lpips
   

    # experiments
    ave_psnr, ave_ssim, ave_lpips = test_rho(lambda_, zeta=zeta, model_output_type=model_output_type, tau=tau)

    return ave_psnr, ave_ssim, ave_lpips

    # ---------------------------------------
    # Average PSNR and LPIPS for all sf and kernels
    # ---------------------------------------

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--noise_level_img', type=float, default=12.75/255.0, help='noise level on the measurement')
    parser.add_argument('--blur_mode', type=str, default='Gaussian', help='Gaussian or motion')

    parser.add_argument('--lambda_', type=float, default=20, help='weight parameter for regularizer')
    parser.add_argument('--tau', type=float, default=0.5, help='weighting parameter between two regularizer, 1 for generative, 0 for discriminative')

    parser.add_argument('--zeta', type=float, default=0.3, help='stochastic noise for ddim')

    parser.add_argument('--save_results_dir', default='results/rdmd_deblur', type=str, help='directory to save results')

    parser.add_argument('--save_img', action='store_true', default=False)
    parser.add_argument('--testset', type=str, default='ffhq_val_100')

    args = parser.parse_args()

    run_deblur(args)
