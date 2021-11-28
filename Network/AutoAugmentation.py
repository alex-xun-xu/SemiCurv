import numpy as np
import torch
from torch import nn
import GeoTform

## Auto Augmentation Module
#
#   This module provides each augmentation operation as differentiable tensor operations
#   Supported operations include geometric and pixel transformations.
#   Geometric transformation: rotation, translation, scaling, shearing, etc.
#
#

class AutoAug(nn.Module):

    def __init__(self,aug_para,device):

        super().__init__()


        self.aug_para = aug_para
        self.device = device

        self.operations = []
        self.para_learnable = []
        self.para_learnable_names = []

        if 'rotation' in self.aug_para:
            opRotation = self.Rotation(para=self.aug_para['rotation']['para'],
                          distribution=self.aug_para['rotation']['distribution'],device=self.device)
            self.operations.append(opRotation)
            self.para_learnable += [v for i, (k, v) in enumerate(opRotation.para_learnable.items())]
            self.para_learnable_names += [k for i, (k,v) in enumerate(opRotation.para_learnable.items())]

        if 'translation' in self.aug_para:
            opTranslation = self.Translation(para=self.aug_para['translation']['para'],
                                       distribution=self.aug_para['translation']['distribution'], device=self.device)
            self.operations.append(opTranslation)
            self.para_learnable += [v for i, (k,v) in enumerate(opTranslation.para_learnable.items())]
            self.para_learnable_names += [k for i, (k,v) in enumerate(opTranslation.para_learnable.items())]

        if 'scaling' in self.aug_para:
            opScaling = self.Scaling(para=self.aug_para['scaling']['para'],
                                             distribution=self.aug_para['scaling']['distribution'], device=self.device)
            self.operations.append(opScaling)
            self.para_learnable += [v for i, (k, v) in enumerate(opScaling.para_learnable.items())]
            self.para_learnable_names += [k for i, (k, v) in enumerate(opScaling.para_learnable.items())]

    def forward(self,X,Y):
        '''
        Forward pass
        X: input images ~ B*H*W*C
        Y: input ground-truth ~ B*H*W*C
        '''

        Ts = [] # all transformation matrices

        for op in self.operations:
            X, Y, T = op(X,Y)
            Ts.append(T)

        return X, Y, Ts



    class Rotation(nn.Module):

        def __init__(self,para,device,distribution='uniform'):
            '''
            distribution: prior distribution for rotation.
                candidates include 'uniform' - i.e. rotation angle (theta) is sampled from a uniform distribution
            '''
            super().__init__()

            self.GeometricTransform = GeoTform.GeometricTransform()
            self.distribution = distribution
            self.para_init = para
            self.para_learnable = {}
            ## Append parameters as learnable
            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['theta_max'] = nn.Parameter(torch.tensor(self.para_init['theta_max'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['theta_min'] = nn.Parameter(torch.tensor(self.para_init['theta_min'],dtype=torch.float32,device=device),requires_grad=True)

            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['theta_mean'] = nn.Parameter(torch.tensor(self.para_init['theta_mean'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['theta_sigma'] = nn.Parameter(torch.tensor(self.para_init['theta_sigma'],dtype=torch.float32,device=device),requires_grad=True)


        def forward(self, X, Y):
            '''
            X: input image B*H*W*C
            '''

            B,H,W,C = X.shape
            device = X.device

            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                theta = torch.rand(B,device=device)
                theta = (self.para_learnable['theta_max']-self.para_learnable['theta_min'])*theta + self.para_learnable['theta_min']
                theta /= 3.14159*180
            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                theta = torch.normal(0,1,size=(B),device=device)
                theta = self.para_learnable['rot_sigma']*theta + self.para_learnable['rot_mean']
                pass

            ## Apply rotation to image
            T = self.GeometricTransform.rotate_mat_ch(theta)
            X_o = self.GeometricTransform.transform_image_tensor(X,T)
            Y_o = self.GeometricTransform.transform_image_tensor(Y,T)

            return X_o, Y_o, T

    class Translation(nn.Module):

        def __init__(self,para,device,distribution='uniform'):
            '''
            distribution: prior distribution for translation.
                candidates include 'uniform' - i.e. translation (dx, dy) are sampled from a uniform distribution
            '''
            super().__init__()

            self.GeometricTransform = GeoTform.GeometricTransform()
            self.distribution = distribution
            self.para_init = para
            self.para_learnable = {}
            ## Append parameters as learnable
            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['dx_max'] = nn.Parameter(torch.tensor(self.para_init['dx_max'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dx_min'] = nn.Parameter(torch.tensor(self.para_init['dx_min'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dy_max'] = nn.Parameter(torch.tensor(self.para_init['dy_max'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dy_min'] = nn.Parameter(torch.tensor(self.para_init['dy_min'],dtype=torch.float32,device=device),requires_grad=True)


            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['dx_mean'] = nn.Parameter(torch.tensor(self.para_init['dx_mean'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dx_sigma'] = nn.Parameter(torch.tensor(self.para_init['dx_sigma'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dy_mean'] = nn.Parameter(torch.tensor(self.para_init['dy_mean'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['dy_sigma'] = nn.Parameter(torch.tensor(self.para_init['dy_sigma'],dtype=torch.float32,device=device),requires_grad=True)


        def forward(self, X, Y):
            '''
            X: input image B*H*W*C
            '''

            B,H,W,C = X.shape
            device = X.device

            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                dx = torch.rand(B,device=device)
                dy = torch.rand(B,device=device)
                dx = (self.para_learnable['dx_max']-self.para_learnable['dx_min'])*dx + self.para_learnable['dx_min']
                dy = (self.para_learnable['dy_max']-self.para_learnable['dy_min'])*dy + self.para_learnable['dy_min']
                # unnormalize hyperpara
                dx *= 50
                dy *= 50

            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                dx = torch.normal(0,1,size=(B),device=device)
                dy = torch.normal(0,1,size=(B),device=device)
                dx = self.para_learnable['dx_sigma']*dx + self.para_learnable['dx_mean']
                dy = self.para_learnable['dy_sigma']*dy + self.para_learnable['dy_mean']
                pass

            ## Apply rotation to image
            T = self.GeometricTransform.translate_mat_ch(dx,dy)
            X_o = self.GeometricTransform.transform_image_tensor(X,T)
            Y_o = self.GeometricTransform.transform_image_tensor(Y,T)

            return X_o, Y_o, T

    class Scaling(nn.Module):

        def __init__(self,para,device,distribution='uniform'):
            '''
            distribution: prior distribution for translation.
                candidates include 'uniform' - i.e. translation (dx, dy) are sampled from a uniform distribution
            '''
            super().__init__()

            self.GeometricTransform = GeoTform.GeometricTransform()
            self.distribution = distribution
            self.para_init = para
            self.para_learnable = {}
            ## Append parameters as learnable
            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['sx_max'] = nn.Parameter(torch.tensor(self.para_init['sx_max'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sx_min'] = nn.Parameter(torch.tensor(self.para_init['sx_min'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sy_max'] = nn.Parameter(torch.tensor(self.para_init['sy_max'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sy_min'] = nn.Parameter(torch.tensor(self.para_init['sy_min'],dtype=torch.float32,device=device),requires_grad=True)


            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                self.para_learnable['sx_mean'] = nn.Parameter(torch.tensor(self.para_init['sx_mean'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sx_sigma'] = nn.Parameter(torch.tensor(self.para_init['sx_sigma'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sy_mean'] = nn.Parameter(torch.tensor(self.para_init['sy_mean'],dtype=torch.float32,device=device),requires_grad=True)
                self.para_learnable['sy_sigma'] = nn.Parameter(torch.tensor(self.para_init['sy_sigma'],dtype=torch.float32,device=device),requires_grad=True)


        def forward(self, X, Y):
            '''
            X: input image B*H*W*C
            '''

            B,H,W,C = X.shape
            device = X.device

            if self.distribution == 'uniform':
                # sample rotation parameter theta from prior distribution
                sx = torch.rand(B,device=device)
                sy = torch.rand(B,device=device)
                sx = (self.para_learnable['sx_max']-self.para_learnable['sx_min'])*sx + self.para_learnable['sx_min']
                sy = (self.para_learnable['sy_max']-self.para_learnable['sy_min'])*sy + self.para_learnable['sy_min']

                sx += 1
                sy += 1

            elif self.distribution == 'gaussian':
                # sample rotation parameter theta from prior distribution
                dx = torch.normal(0,1,size=(B),device=device)
                dy = torch.normal(0,1,size=(B),device=device)
                dx = self.para_learnable['dx_sigma']*dx + self.para_learnable['dx_mean']
                dy = self.para_learnable['dy_sigma']*dy + self.para_learnable['dy_mean']
                pass

            ## Apply rotation to image
            T = self.GeometricTransform.scaling_mat_ch(sx,sy)
            X_o = self.GeometricTransform.transform_image_tensor(X,T)
            Y_o = self.GeometricTransform.transform_image_tensor(Y,T)

            return X_o, Y_o, T