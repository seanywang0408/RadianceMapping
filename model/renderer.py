from .net import MLP, UNet
import torch
import torch.nn as nn
from torchvision import transforms as T

class Renderer(nn.Module):
    """
    This class implements radiance mapping and refinement.
    """

    def __init__(self, args):
        super(Renderer, self).__init__()
        self.mlp = MLP(args.dim).to(args.device)
        self.unet = UNet(args).to(args.device)
        self.dim = args.dim

        if args.xyznear:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(args.scale_min, args.scale_max), ratio=(1., 1.))
        else:
            self.randomcrop = T.RandomResizedCrop(args.train_size, scale=(args.scale_min, args.scale_max), ratio=(1., 1.), interpolation=T.InterpolationMode.NEAREST)
        
        self.pad_w = T.Pad(args.pad, 1., 'constant')
        self.pad_b = T.Pad(args.pad, -1., 'constant')
        
        self.xyznear = args.xyznear # bool
        self.mask = args.pix_mask
        self.train_size = args.train_size

    def forward(self, zbuf, ray, gt, mask_gt, isTrain, xyz_o):
        """
        Args:
            zbuf: z-buffer from rasterization (index buffer when xyznear is True)
            ray: ray direction map
            gt: gt image (used in training to maintain consistent cropping and resizing with input)
            mask_gt: gt mask (used in dtu dataset)
            isTrain: train mode or not
            xyz_o: world coordinates of point clouds (used when xyzenar is True)

        Output:
            img: rendered image
            gt: gt image after cropping and resizing
            mask_gt: gt mask after cropping and resizing 
            fea_map: the first three dimensions of the feature map of radiance mapping
        """

        if isTrain:

            o = ray[:self.train_size,:self.train_size,:3] # trainH trainW 3
            dirs = self.pad_w(ray[...,3:6].permute(2,0,1).unsqueeze(0)) # 1 3 H W
            cos = self.pad_w(ray[...,-1:].permute(2,0,1).unsqueeze(0)) # 1 1 H W
            gt = self.pad_w(gt.permute(2,0,1).unsqueeze(0)) # 1 3 H W
            zbuf = self.pad_b(zbuf.permute(2,0,1).unsqueeze(0))

            if mask_gt is not None:
                # never pad
                mask_gt = mask_gt.permute(2,0,1).unsqueeze(0)
                cat_img = torch.cat([dirs, cos, gt, zbuf, mask_gt], dim=1) 
            else:
                cat_img = torch.cat([dirs, cos, gt, zbuf], dim=1) 

            cat_img = self.randomcrop(cat_img)

            _, _, H, W = cat_img.shape
            K = 1

            dirs = cat_img[0,:3].permute(1,2,0)
            cos = cat_img[0,3:4].permute(1,2,0)
            gt = cat_img[0,4:7].permute(1,2,0)
            zbuf = cat_img[0,7:8].permute(1,2,0)
            if mask_gt is not None:
                mask_gt = cat_img[0,8:].permute(1,2,0)

            pix_mask = zbuf > 0.2  # h w k 
            
        else:

            H, W, K = zbuf.shape  # in fact, k always = 1
            o = ray[...,:3] # H W 3
            dirs = ray[...,3:6] # H W 3
            cos = ray[...,-1:] # H W 1

            pix_mask = zbuf > 0  # h w k 

        o = o.unsqueeze(-2).expand(H, W, K, 3)[pix_mask] # occ_point 3
        dirs = dirs.unsqueeze(-2).expand(H, W, K, 3)[pix_mask]  # occ_point 3
        cos = cos.unsqueeze(-2).expand(H, W, K, 1)[pix_mask]  # occ_point 1
        zbuf = zbuf.unsqueeze(-1)[pix_mask]  # occ_point 1

        if self.xyznear:
            xyz_near = o + dirs * zbuf / cos # occ_point 3
        else:
            xyz_near = xyz_o[zbuf.squeeze(-1).long()]

        feature = self.mlp(xyz_near, dirs) # occ_point 3

        feature_map = torch.zeros([H, W, K, self.dim], device=zbuf.device)
        feature_map[pix_mask] = feature


        # Unet
        # sigma H, W, 1, 1
        # color H, W, 1, 8
        # gt h w 3
        pix_mask = pix_mask.int().unsqueeze(-1).permute(2,3,0,1)# h w 1 1
        feature_map_view  = feature_map.clone().squeeze(-2)[...,:3]
        feature_map = self.unet(feature_map.permute(2,3,0,1))

        if self.mask:
            feature_map = feature_map * pix_mask + (1 - pix_mask) # 1 3 h w
        img = feature_map.squeeze(0).permute(1,2,0)


        return {'img':img, 'gt':gt, 'mask_gt':mask_gt, 'fea_map':feature_map_view}