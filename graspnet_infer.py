import os
import sys
import numpy as np
import open3d as o3d
import scipy.io as scio
from PIL import Image
import cv2
import torch
from graspnetAPI import GraspGroup
import math
import copy
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
import pickle
class GraspNetInfer:
    def __init__(self, checkpoint_path="./checkpoints/checkpoint-rs.tar", num_point=20000, num_view=300, collision_thresh=0.01, voxel_size=0.01\
                , img_resolution = (960,960), vertical_fov = 45,factor_depth = 1):
        self.checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),checkpoint_path)
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
        self.vis_renderer_init()
    def vis_renderer_init(self):
        self.vis_renderer =o3d.visualization.Visualizer()
        self.vis_renderer.create_window(visible=False)
        # change the viewpoint of vis_renderer
        self.renderer_view = self.vis_renderer.get_view_control()
    def vis_grasps(self):
        # visualize
        cloud = self.cloud
        gg = self.gg
        gg.nms()
        gg.sort_by_score()
        gg = gg[:5]
        grippers = gg.to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])
    def offscreen_vis_grasps(self,gg,cloud,image_path):
        grippers = gg.to_open3d_geometry_list()
        self.vis_renderer.add_geometry(cloud,reset_bounding_box=True)
        for gripper in grippers:
            self.vis_renderer.add_geometry(gripper,reset_bounding_box=False)
        # self.vis_renderer.update_renderer()
        self.renderer_view.set_front([0,-0.707,-0.707])
        self.vis_renderer.capture_screen_image(image_path,do_render=True)
        self.vis_renderer.clear_geometries()
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
    def change_boxes2masks(self,box):
        # input the boxes and return the masks
        # boxes is a list of [x1,y1,x2,y2]
        # masks is a numpy array, 1 is the mask, 0 is the background
        box = np.array(box,dtype=np.int32)
        masks = np.zeros(self.img_resolution,dtype=np.bool_)
        x1,y1,x2,y2 = box
        x1 = max(0,x1-50)
        y1 = max(0,y1-50)
        x2 = min(self.img_resolution[1],x2+50)
        y2 = min(self.img_resolution[0],y2+50)
        masks[y1:y2,x1:x2] = 1
        return masks
    def process_image(self,rgb_image, depth_image,boxes):
        workspace_mask = self.change_boxes2masks(boxes)
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
    def get_cloud(self,rgb_image,depth_image):
        depth = copy.deepcopy(depth_image)
        color = np.array(rgb_image,dtype=np.float32)
        color = color/255.0
        # assign intrinsic matrix and depth factor
        intrinsic = self.intrinsic_matrix
        factor_depth = self.factor_depth
        # generate cloud
        camera = CameraInfo(self.img_resolution[0], self.img_resolution[1], intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
        cloud_from_depth = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask = np.ones(depth.shape, dtype=np.bool_)
        cloud_masked = cloud_from_depth[mask]
        color_masked = color[mask]
        # get cloud
        cloud = o3d.geometry.PointCloud()
        # import the color image and cloud_from_depth to o3d
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        
        return cloud
    def infer(self,rgb_image_raw,depth_image_raw,workspace_mask):
        rgb_image = copy.deepcopy(rgb_image_raw)
        depth_image = copy.deepcopy(depth_image_raw)
        end_points, cloud = self.process_image(rgb_image,depth_image,workspace_mask)
        gg = self.get_grasps(end_points)
        if self.collision_thresh > 0:
            gg = self.collision_detection(gg, np.array(cloud.points))
        # self.vis_grasps(gg, cloud)
        # sorted gg by gg.scores from high to low
        # sorted_idx = np.argsort(-gg.scores)
        # gg = gg[sorted_idx]
        gg.nms()
        gg.sort_by_score()
        self.gg = gg
        self.cloud = cloud
        return gg, cloud

if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    data_dir = "/home/zsn/Desktop/mjc_grasp/reconstruct_scripts/graspnet_baseline/doc/example_data"
    color = np.load('../rgb.npy')
    depth = np.load('../depth.npy')
    # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))
    workspace_boxes = [0,0,960,960]
    graspnet = GraspNetInfer()
    gg,cloud = graspnet.infer(color, depth, workspace_boxes)
    graspnet.offscreen_vis_grasps(gg,cloud,'test.png')