import torch
import torch.utils.data as data
import os
import numpy as np
import json
from PIL import Image
from .utils import load_pc, get_rays, PointCloud
from torchvision import transforms as T



class nerfDataset(data.Dataset):

    def __init__(self, args, split, mode):
    
        self.img_wh = (args.W, args.H)
        self.device = args.device
        self.pc_dir = args.pcdir
        self.mode = mode

        datadir = args.datadir
        self.blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        with open(os.path.join(datadir, 'transforms_' + split + '.json'), 'r') as f:
            self.meta = json.load(f)
        self.transform = T.ToTensor()
        focal = 0.5 * 800 / np.tan(0.5 * self.meta['camera_angle_x']) 
        focal *= self.img_wh[0] / 800
        self.intrinsic = torch.tensor([[focal, 0, self.img_wh[0] / 2], [0, focal, self.img_wh[1] / 2], [0, 0, 1]], dtype=torch.float32)

        self.id_list = [i for i in range(len(self.meta["frames"]))]
        self.img_list = []
        self.w2c_list = []
        self.ray_list = []

        for idx in self.id_list:
            frame = self.meta['frames'][idx] 
            image_path = os.path.join(datadir, f"{frame['file_path']}.png")
            # print(image_path)

        # load img
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).permute(1,2,0)
            self.img_list.append(img[...,:3] * img[...,-1:] + (1 - img[...,-1:])) 
            

        # load pose
            pose = np.array(frame['transform_matrix']) @ self.blender2opencv
            c2w = torch.tensor(pose, dtype=torch.float32)

            # load ray
            if mode == 'render':
                ray = get_rays(args.H, args.W, self.intrinsic, c2w)
                self.ray_list.append(ray)
            else:
                #TODO better way?
                self.ray_list.append(torch.ones([0]))

            pose = np.linalg.inv(pose)
            self.w2c_list.append(torch.tensor(pose, dtype=torch.float32))


    def get_pc(self):
        pc_xyz = load_pc(self.pc_dir, self.device)  # n,3
        pc = PointCloud(pc_xyz, self.intrinsic, self.device, self.img_wh)
        return pc

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (H W C),
                       "img.mask": mask (H,W 1),
                       "camera_mat": camera_mat (4,4)
        """
        idx = idx % self.__len__()
        rgb = self.img_list[idx]
        w2c = self.w2c_list[idx]
        ray = self.ray_list[idx]

        return {"idx": str(idx).rjust(3,'0'),
                "rgb": rgb.to(self.device), 
                "w2c": w2c.to(self.device),
                "ray": ray.to(self.device)}

class DTUDataset(data.Dataset):

    @staticmethod
    def read_cam(path):
        data = []
        f = open(path)
        for i in range(10):
            line = f.readline()
            tmp = line.split()
            data.append(tmp)
        f.close()
        pose = np.array(data[1:5], dtype=np.float32)
        intrinsic = np.array(data[7:10], dtype=np.float32)
        return pose, intrinsic

    def __init__(self, args, split, mode):

        self.id_list = []
        test_ = [7, 12, 17, 22, 27, 32, 37, 42, 47]
        if split == 'train':
            for i in range(64):
                if i in test_:
                    continue
                self.id_list.append(str(i).rjust(2,'0'))
        else:
            self.id_list = [str(i).rjust(2,'0') for i in test_]
        print(split, self.id_list)

        self.transform = T.ToTensor()

        self.img_list = []
        self.w2c_list = []
        self.ray_list = []
        self.mask_list = []

        self.img_wh = (args.W, args.H)
        self.pcdir = args.pcdir
        self.device = args.device
        
        
        self.intrinsic = None

        for idx in self.id_list:
            image_path = os.path.join(args.datadir, 'image', f"0000{idx}.png") 
            img = Image.open(image_path)
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img).permute(1,2,0)

            mask_path = os.path.join(args.datadir, 'mask', f"0{idx}.png") 
            mask = Image.open(mask_path)
            mask = mask.resize(self.img_wh, Image.LANCZOS)
            mask = self.transform(mask).permute(1,2,0)
            self.img_list.append(img * mask + 1 - mask)
            self.mask_list.append(mask) # h w 3
            # exit()
            

        # load pose
            cam_path = os.path.join(args.datadir, 'cams_1', f"000000{idx}_cam.txt") 
            pose, intrinsic = self.read_cam(cam_path)
            w2c = torch.tensor(pose)
            c2w = np.linalg.inv(pose)
            c2w = torch.tensor(c2w)

            if self.intrinsic is None:
                scale1_c = 1600 / args.W
                scale2_c = 1200 / args.H

                intrinsic[0:1, :] = intrinsic[0:1, :] / scale1_c
                intrinsic[1:2, :] = intrinsic[1:2, :] / scale2_c
                self.intrinsic = torch.tensor(intrinsic)
            

            # load ray
            if mode == 'render':
                ray = get_rays(args.H, args.W, self.intrinsic, c2w)
                self.ray_list.append(ray)
            else:
                self.ray_list.append(torch.ones([0]))

            self.w2c_list.append(w2c)


    def get_pc(self):
        pc_xyz = load_pc(self.pcdir, self.device, 1)  # n,3
        pc = PointCloud(pc_xyz, self.intrinsic, self.device, self.img_wh)
        return pc

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (H W C),
                       "img.mask": mask (H,W 1),
                       "camera_mat": camera_mat (4,4)
        """
        idx = idx % self.__len__()
        rgb = self.img_list[idx]
        w2c = self.w2c_list[idx]
        ray = self.ray_list[idx]
        mask = self.mask_list[idx]

        return {"idx": str(idx).rjust(3,'0'),
                "rgb": rgb.to(self.device), 
                "w2c": w2c.to(self.device),
                "ray": ray.to(self.device),
                'mask': mask.to(self.device)}



class ScanDataset(data.Dataset):

    @staticmethod
    def CameraRead(dir, line_num):
        camera_para = []
        f = open(dir)
        for i in range(line_num):
            line = f.readline()
            tmp = line.split()
            camera_para.append(tmp)
        camera_para = np.array(camera_para, dtype=np.float32)
        f.close()
        return torch.tensor(camera_para, dtype=torch.float32)

    def __init__(self, args, split, mode):
        self.img_size = (args.H, args.W)
        self.device = args.device
        self.mode = mode
        self.id_list = []
        self.pc_dir = args.pcdir
        self.args = args
        img_path = os.path.join(args.datadir, 'color_select')
        pose_path = os.path.join(args.datadir, 'pose_select')
        total = len(os.listdir(img_path))
            
        raw = list(np.linspace(10, total - 10, 10, dtype=int))

        if split == 'test':
            self.id_list = [str(k) for k in raw]
        else:
            train_id = []
            for i in range(total):
                if i in raw:
                    continue
                train_id.append(i)
            self.id_list = [str(k) for k in train_id]
        print(self.id_list)

        intrinsic_path = os.path.join(args.datadir, 'intrinsic', 'intrinsic_color.txt')
        intrinsic = self.CameraRead(intrinsic_path, 4)[:3,:3]
        

        scale1_c = (1296 - 1) / (self.img_size[1] - 1)
        scale2_c = (968 - 1) / (self.img_size[0] - 1)

        intrinsic[0:1, :] = intrinsic[0:1, :] / scale1_c
        intrinsic[1:2, :] = intrinsic[1:2, :] / scale2_c
        self.intrinsic = intrinsic

        
        self.transform = T.ToTensor()
        self.img_list = []
        self.w2c_list = []
        self.ray_list = []

        for idx in self.id_list:
            image_path = os.path.join(img_path, idx + '.jpg')

        # load img
            img = Image.open(image_path)
            img = img.resize((self.img_size[1], self.img_size[0]), Image.LANCZOS)
            img = self.transform(img).permute(1,2,0)
            self.img_list.append(img)
            

        # load pose
            c2w = self.CameraRead(os.path.join(pose_path, idx + '.txt'), 4)
            w2c = torch.inverse(c2w)
            self.w2c_list.append(w2c)

            # load ray
            if mode == 'render':
                ray = get_rays(args.H, args.W, self.intrinsic, c2w)
                self.ray_list.append(ray)
            else:
                self.ray_list.append(torch.ones([0]))



    def get_pc(self):
        pc_xyz = load_pc(self.pc_dir, self.device, down=4)  # n,3
        pc = PointCloud(pc_xyz, self.intrinsic, self.device, (self.args.W, self.args.H))
        return pc


    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, idx):
        """
        Returns:
            data dict {"img.rgb": rgb (H W C),
                       "img.mask": mask (H,W 1),
                       "camera_mat": camera_mat (4,4)
        """
        pass
        idx = idx % self.__len__()
        id = self.id_list[idx]
        rgb = self.img_list[idx]
        w2c = self.w2c_list[idx]
        ray = self.ray_list[idx]

        return {"idx": str(idx).rjust(3,'0'),
                "file_id": id,
                "rgb": rgb.to(self.device), 
                "w2c": w2c.to(self.device),
                "ray": ray.to(self.device)}

