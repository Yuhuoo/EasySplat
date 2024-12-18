import os
import shutil
import torch
import numpy as np
import argparse
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.sys.path.append(os.path.abspath(os.path.join(BASE_DIR, "submodules", "mast3r")))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from mast3r.model import AsymmetricMASt3R
from dust3r.utils.device import to_numpy
from dust3r.image_pairs import make_pairs
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
from utils.dust3r_utils import load_images, storePly, save_colmap_cameras, save_colmap_images

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=512, choices=[512, 224], help="image size")
    # parser.add_argument("--model_path", type=str, default="./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth", help="path to the model weights")
    parser.add_argument("--weights_path", type=str, default="submodules/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth", help="path to the model weights")
    parser.add_argument("--device", type=str, default='cuda', help="pytorch device")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--schedule", type=str, default='linear')
    parser.add_argument("--lr1", type=float, default=0.07)
    parser.add_argument("--niter1", type=int, default=500)
    parser.add_argument("--lr2", type=float, default=0.014)
    parser.add_argument("--niter2", type=int, default=200)
    parser.add_argument("--matching_conf_thr", type=int, default=5)
    parser.add_argument("--optim_level", type=str, default="refine+depth")
    parser.add_argument("--min_conf_thr", type=float, default=1.0)

    parser.add_argument("--img_base_path", type=str, default="/home/aogao/code/InstantSplat/data/Tanks_for_mast3r/Barn/images")

    return parser

def get_intrinsics(n_imgs, device, focals, principal_points):
    K = torch.zeros((n_imgs, 3, 3), device=device)
    focals = focals.view(n_imgs, -1)
    K[:, 0, 0] = focals[:, 0]
    K[:, 1, 1] = focals[:, -1]
    K[:, :2, 2] = principal_points
    K[:, 2, 2] = 1
    return K
    
if __name__ == '__main__':
    
    parser = get_args_parser()
    args = parser.parse_args()

    device = args.device
    batch_size = args.batch_size
    schedule = args.schedule
    lr1 = args.lr1
    niter1 = args.niter1
    lr2 = args.lr2
    niter2 = args.niter2
    matching_conf_thr = args.matching_conf_thr
    img_base_path = args.img_base_path
    optim_level = args.optim_level
    model = AsymmetricMASt3R.from_pretrained(args.weights_path).to(args.device)
    ##########################################################################################################################################################################################
    
    img_list = sorted(os.listdir(img_base_path))
    img_path_list = [os.path.join(img_base_path, img) for img in img_list]
    images, ori_size = load_images(img_path_list, size=512) 
    print("ori_size", ori_size)
    output_colmap_path = img_base_path.replace("images", "sparse/0")
    os.makedirs(output_colmap_path, exist_ok=True)

    start_time = time.time()
    ##########################################################################################################################################################################################
    pairs = make_pairs(images, scene_graph='oneref', prefilter=None, symmetrize=True)
    if optim_level == 'coarse':
        niter2 = 0
    
    # Sparse GA (forward mast3r -> matching -> 3D optim -> 2D refinement -> triangulation)
    cache_dir = img_base_path.replace("images", "cache")
    if os.path.exists(cache_dir):
        print("clean old cache")
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    scene = sparse_global_alignment(img_path_list, pairs, cache_dir,
                                    model, lr1=lr1, niter1=niter1, lr2=lr2, niter2=niter2, device=device,
                                    opt_depth='depth' in optim_level, shared_intrinsics=True,
                                    matching_conf_thr=matching_conf_thr)

    imgs = [img.reshape(-1, 3) for img in to_numpy(scene.imgs)]
    pp = scene.get_principal_points()
    focals = scene.get_focals()
    intrinsics = to_numpy(get_intrinsics(len(imgs), device, focals, pp))
    poses = to_numpy(scene.get_im_poses())
    pts3d, _, confs = to_numpy(scene.get_dense_pts3d(clean_depth=True))
    confidence_masks = [c.reshape(-1,) for c in to_numpy([c > args.min_conf_thr for c in confs])]

    ##########################################################################################################################################################################################
    end_time = time.time()
    print("Time Cost: ", end_time - start_time)

    # save
    save_colmap_cameras(ori_size, intrinsics, os.path.join(output_colmap_path, 'cameras.txt'))
    save_colmap_images(poses, os.path.join(output_colmap_path, 'images.txt'), img_list)

    pts_4_3dgs = np.concatenate([p[m] for p, m in zip(pts3d, confidence_masks)])
    color_4_3dgs = np.concatenate([i[m] for i, m in zip(imgs, confidence_masks)])
    color_4_3dgs = (color_4_3dgs * 255.0).astype(np.uint8)
    storePly(os.path.join(output_colmap_path, "points3D.ply"), pts_4_3dgs, color_4_3dgs)
    pts_4_3dgs_all = np.array(pts3d).reshape(-1, 3)
    np.save(output_colmap_path + "/pts_4_3dgs_all.npy", pts_4_3dgs_all)
    np.save(output_colmap_path + "/focal.npy", to_numpy(focals))
