import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import cv2
import torch
from graspnetAPI import GraspGroup
import math
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

class GraspNetInfer:
    def __init__(self, checkpoint_path="./checkpoints/checkpoint-rs.tar", num_point=20000, num_view=300, collision_thresh=0.01, voxel_size=0.01\
                , img_resolution = (1280.0,720.0), vertical_fov = 45,factor_depth = 1):
        self.checkpoint_path = checkpoint_path
        self.num_point = num_point
        self.num_view = num_view
        self.collision_thresh = collision_thresh
        self.voxel_size = voxel_size
        self.img_resolution = img_resolution
        self.vertical_fov = vertical_fov
        self.factor_depth = factor_depth
        self.intrinsic_matrix = np.array([[self.img_resolution[1]/(2*math.tan(math.radians(self.vertical_fov/2))),   0.        ,self.img_resolution[0]/2],
                                        [  0.        , self.img_resolution[1]/(2*math.tan(math.radians(self.vertical_fov/2))), self.img_resolution[1]/2],
                                        [  0.        ,   0.        ,   1.        ]])
        self.net = self.get_net()
    def vis_grasps(self):
        # visualize
        cloud = self.cloud
        gg = self.gg
        gg.nms()
        gg.sort_by_score()
        gg = gg[:50]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
        
    def get_grasps(self,end_points):
        # Forward pass
        net = self.net
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        gg = GraspGroup(gg_array)
        return gg
    
    def collision_detection(self,gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=self.voxel_size)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.collision_thresh)
        gg = gg[~collision_mask]
        return gg
    def get_net(self):
        # Init the model
        net = GraspNet(input_feature_dim=0, num_view=self.num_view, num_angle=12, num_depth=4,
                cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        # Load checkpoint
        checkpoint = torch.load(self.checkpoint_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print("-> loaded checkpoint %s (epoch: %d)"%(self.checkpoint_path, start_epoch))
        # set model to eval mode
        net.eval()
        return net
    def process_image(self,rgb_image, depth_image,workspace_mask):
        depth = depth_image
        color = np.array(rgb_image,dtype=np.float32)
        color = color/255.0
        # assign intrinsic matrix and depth factor
        intrinsic = self.intrinsic_matrix
        factor_depth = self.factor_depth
        # generate cloud
        camera = CameraInfo(self.img_resolution[0], self.img_resolution[1], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # sample points
        if len(cloud_masked) >= self.num_point:
            idxs = np.random.choice(len(cloud_masked), self.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), self.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]
        # get cloud
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        # get end_points form sampled
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled
        return end_points, cloud

    def infer(self,rgb_image,depth_image,workspace_mask):
        end_points, cloud = self.process_image(rgb_image,depth_image,workspace_mask)
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        # self.vis_grasps(gg, cloud)
        self.gg = gg
        self.cloud = cloud
        return gg

if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    data_dir = "/home/zsn/Desktop/mjc_grasp/reconstruct_scripts/graspnet_baseline/doc/example_data"
    color = np.array(Image.open(os.path.join(data_dir, 'color1.png')), dtype=np.float32)
    depth = np.array(Image.open(os.path.join(data_dir, 'depth1.png')))/1000
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    graspnet = GraspNetInfer()
    graspnet.infer(color, depth, workspace_mask)
    graspnet.vis_grasps()