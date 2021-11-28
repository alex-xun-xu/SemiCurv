#currently support affine only

from enum import Enum, auto
import numpy as np
from skimage.transform import warp
from timeit import default_timer as timer
import torch
import cv2
import torchgeometry as tg
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt

pi = 3.141592653

class Op(Enum):
    FLIP_LR = auto()
    FLIP_UD = auto()
    ROTATE = auto()
    TRANSLATE_X = auto()
    TRANSLATE_Y = auto()

CV2_EXTRAPOLATION = {
    "constant": cv2.BORDER_CONSTANT,
    "edge": cv2.BORDER_REPLICATE,
    "symmetric": cv2.BORDER_REFLECT,
    "reflect": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP
}

CV2_INTERPOLATION = {
    "nearest" : cv2.INTER_NEAREST,
    "linear" : cv2.INTER_LINEAR,
    "area" : cv2.INTER_AREA,
    "cubic": cv2.INTER_CUBIC
}

SK_INTERPOLATION = {
    "nearest" : 0,
    "linear" : 1,
    "quadratic" : 2,
    "cubic" : 3,
    "quartic" : 4,
    "quintic" : 5
}

class MockTransform():
    def __init__(self):
        pass

    def construct_random_transform(self, count, shuffle_order=True):
        pass

    def transform_images(self, images, mode='constant', interpolation='linear'):
        return images

    def inv_transform_images(self, images, mode='constant', interpolation='linear'):
        return images

    def inv_transform_tensors(self, t_images):
        return t_images

class GeometricTransform():
    def __init__(self):
        self.operations = {}
        self.random_values = []
        self.matrix_templates = {}
        self.fliplr_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.flipud_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
        self.imgcount = 0

    def rotate_mat(self,theta):
        '''
        Generate Rotation Matrix given theta (in degrees)
        :param theta:
        :return:
        '''
        return np.array([[np.cos(theta/180*np.pi), -np.sin(theta/180*np.pi), 0],
                         [np.sin(theta/180*np.pi), np.cos(theta/180*np.pi), 0],
                         [0, 0, 1]])

    def translate_mat(self,dx=0,dy=0):
        '''
        Generate translation matrix given dx and/or dy
        :param dx:
        :param dy:
        :return:
        '''
        return np.array([[1,0,dx],[0,1,dy],[0,0,1]])

    def shear_mat(self,shearx=0, sheary=0):
        '''
        Generate shearing Matrix given shearx and/or sheary
        :param shearx:
        :param sheary:
        :return:
        '''
        return np.array([[1, shearx, 0],
                         [sheary, 1, 0],
                         [0, 0, 1]])

    def rotate_mat_ch(self,theta):
        '''
        Generate Rotation Matrix given theta (in degree)
        :param theta:
        :return:
        '''

        device = theta.device
        B = theta.shape
        # T = torch.stack([torch.tensor([[torch.cos(theta_i/180*pi), -torch.sin(theta_i/180*pi), 0],
        #                  [torch.sin(theta_i/180*pi), torch.cos(theta_i/180*pi), 0],
        #                  [0, 0, 1]],device=theta.device,requires_grad=True) for theta_i in theta],0)
        #I = torch.tensor(I,device=theta.device)
        T = torch.stack([torch.stack([torch.stack([torch.cos(theta_i / 180 * pi), -torch.sin(theta_i / 180 * pi), torch.zeros(1,dtype=torch.float,device=device)[0]]),
                                      torch.stack([torch.sin(theta_i / 180 * pi), torch.cos(theta_i / 180 * pi), torch.zeros(1,dtype=torch.float,device=device)[0]]),
                                      torch.stack([torch.zeros(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0], torch.ones(1,dtype=torch.float,device=device)[0]])]) for theta_i in
                         theta], 0)

        return T

    def translate_mat_ch(self,dx,dy):
        '''
        Generate Translation Matrix given dx and dy
        :param dx:
        :param dx:
        :return:
        '''

        device = dx.device
        B = dx.shape
        T = torch.stack([torch.stack([torch.stack([torch.ones(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0], dx_i]),
                                      torch.stack([torch.zeros(1,dtype=torch.float,device=device)[0], torch.ones(1,dtype=torch.float,device=device)[0], dy_i]),
                                      torch.stack([torch.zeros(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0], torch.ones(1,dtype=torch.float,device=device)[0]])]) for dx_i, dy_i in zip(dx, dy)])

        return T

    def scaling_mat_ch(self,sx,sy):
        '''
        Generate Scaling Matrix given sx and sy
        :param sx:
        :param sy:
        :return:
        '''

        device = sx.device
        B = sx.shape
        T = torch.stack([torch.stack([torch.stack([sx_i*torch.ones(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0]]),
                                      torch.stack([torch.zeros(1,dtype=torch.float,device=device)[0], sy_i*torch.ones(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0]]),
                                      torch.stack([torch.zeros(1,dtype=torch.float,device=device)[0], torch.zeros(1,dtype=torch.float,device=device)[0], torch.ones(1,dtype=torch.float,device=device)[0]])]) for sx_i, sy_i in zip(sx, sy)])

        return T


    def add_fliplr(self, probability=0.5):
        self.operations.update({'FLIP_LR': {'prob':probability}})

    def add_flipud(self, probability=0.5):
        self.operations.update({'FLIP_UD': {'prob':probability}})

    def add_shearx(self, range, probability=0.2):
        self.operations.update({'SHEAR_X': {'range':range, 'prob':probability}})

    def add_sheary(self, range, probability=0.2):
        self.operations.update({'SHEAR_Y': {'range':range, 'prob':probability}})

    def add_rotate(self, range, probability=1):
        self.operations.update({'ROTATE': {'range':range, 'prob':probability}})

    def add_translate_x(self, range):
        self.operations.update({'TRANSLATE_X': {'range':range}})

    def add_translate_y(self, range):
        self.operations.update({'TRANSLATE_Y': {'range':range}})

    def add_elastic(self,para):
        '''Add elastic deformation of image'''
        self.operations.update({'ElasticDeform': {'para':para}})


    def construct_random_transform(self, images):
        '''
        Construct random similarity transform (with reflective) by randomly sampling flipping, rotation and translation
        :param images: images to be transformed B*H*W*C
        :return:
        '''
        batchsize, H, W, _ = images.shape
        ## Sequentially Generate Transformation
        self.Tform = [np.identity(3)]*batchsize
        self.ElasticDeformFlow = {'dx':None,'dy':None}
        if 'ElasticDeform' in self.operations:
            self.ElasticDeformFlow = [np.zeros([2,H,W])]*batchsize
            # self.ElasticDeformFlow['dy'] = [np.zeros([H,W])]*batchsize

        for b_i in range(batchsize):
            ## Flip Left-Right
            if 'FLIP_LR' in self.operations:
                if np.random.random() < self.operations['FLIP_LR']['prob']:
                    self.Tform[b_i] = self.fliplr_matrix @ self.Tform[b_i]
            ## Flip Up-Down
            if 'FLIP_UD' in self.operations:
                if np.random.random() < self.operations['FLIP_UD']['prob']:
                    self.Tform[b_i] = self.fliplr_matrix @ self.Tform[b_i]
            ## Shear X
            if 'SHEAR_X' in self.operations:
                if np.random.random() < self.operations['SHEAR_X']['prob']:
                    shearx = np.random.choice(self.operations['SHEAR_X']['range'],1)[0]
                    self.Tform[b_i] = self.shear_mat(shearx=shearx) @ self.Tform[b_i]
            ## Shear Y
            if 'SHEAR_Y' in self.operations:
                if np.random.random() < self.operations['SHEAR_Y']['prob']:
                    sheary = np.random.choice(self.operations['SHEAR_Y']['range'], 1)[0]
                    self.Tform[b_i] = self.shear_mat(sheary=sheary) @ self.Tform[b_i]
            ## Rotate
            if 'ROTATE' in self.operations:
                # if np.random.random() < self.operations['ROTATE']['prob']:
                theta = np.random.choice(self.operations['ROTATE']['range'],1)[0]
                self.Tform[b_i] = self.rotate_mat(theta) @ self.Tform[b_i]
            ## Translate X
            if 'TRANSLATE_X' in self.operations:
                dx = W*np.random.choice(self.operations['TRANSLATE_X']['range'],1)[0]
                self.Tform[b_i] = self.translate_mat(dx=dx) @ self.Tform[b_i]
            ## Translate Y
            if 'TRANSLATE_Y' in self.operations:
                dy = H*np.random.choice(self.operations['TRANSLATE_Y']['range'],1)[0]
                self.Tform[b_i] = self.translate_mat(dy=dy) @ self.Tform[b_i]

            ## Generate Elastic Deformation

            if 'ElasticDeform' in self.operations:
                shape = [H,W]
                random_state = np.random.RandomState(None)

                dx = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                                     self.operations['ElasticDeform']['para']['sigma'], mode="constant", cval=0) * \
                                    self.operations['ElasticDeform']['para']['alpha']
                dy = gaussian_filter((random_state.rand(*shape) * 2 - 1),
                                     self.operations['ElasticDeform']['para']['sigma'], mode="constant", cval=0) * \
                                    self.operations['ElasticDeform']['para']['alpha']

                self.ElasticDeformFlow[b_i] = np.stack([dx, dy])
                # x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
                # indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    def transform_images_int(self, images, extrapolation='constant', interpolation='linear'):
        '''
        Apply transformation to images or ground-truth masks
        :param images: raw image or gt masks    B*H*W*C
        :param extrapolation: extrapolation mode
        :param interpolation: interpolation mode
        :return:
        '''
        batchsize, H, W, _ = images.shape
        T_norm = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])  # normalize image center to origin
        T_unnorm = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]])  # unnormalize image back to image center
        img_transformed = np.zeros_like(images) # transformed images
        for img_idx, image in enumerate(images):
            # Apply elastic deformation
            if 'ElasticDeform' in self.operations:
                image = self.elastic_deform(image,self.ElasticDeformFlow[img_idx])
            # Apply transform to each image. extrapolation mode controls the way to extrapolate images.
            image_warped = cv2.warpAffine(image, (T_unnorm@self.Tform[img_idx]@T_norm)[:2],
                dsize=(W,H), flags=CV2_INTERPOLATION[interpolation],
                borderMode=CV2_EXTRAPOLATION[extrapolation], borderValue=0)

            # Check if transforming ground-truth map
            if img_transformed.dtype == np.uint8 and image_warped.max()<1.1:
                image_warped = (image_warped>0.5).astype(np.uint8)
                pass

            # Accumulate transformed images/ground-truths
            if len(image_warped.shape) <3:
                img_transformed[img_idx] = image_warped[...,np.newaxis]
            else:
                img_transformed[img_idx] = image_warped
        return img_transformed

    def transform_images_float(self, images, extrapolation='constant', interpolation='linear'):
        '''
        Apply transformation to images or ground-truth masks
        :param images: raw image or gt masks    B*H*W*C
        :param extrapolation: extrapolation mode
        :param interpolation: interpolation mode
        :return:
        '''
        batchsize, H, W, _ = images.shape
        T_norm = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])  # normalize image center to origin
        T_unnorm = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]])  # unnormalize image back to image center
        img_transformed = np.zeros_like(images,dtype=np.float32) # transformed images
        for img_idx, image in enumerate(images):
            # Apply elastic deformation
            if 'ElasticDeform' in self.operations:
                image = self.elastic_deform(image,self.ElasticDeformFlow[img_idx])
            # Apply transform to each image. extrapolation mode controls the way to extrapolate images.
            image_warped = cv2.warpAffine(image, (T_unnorm@self.Tform[img_idx]@T_norm)[:2],
                dsize=(W,H), flags=CV2_INTERPOLATION[interpolation],
                borderMode=CV2_EXTRAPOLATION[extrapolation], borderValue=0)

            # Check if transforming ground-truth map
            if img_transformed.dtype == np.uint8 and image_warped.max()<1.1:
                # image_warped = (image_warped>0.5).astype(np.uint8)
                pass

            # Accumulate transformed images/ground-truths
            if len(image_warped.shape) <3:
                img_transformed[img_idx] = image_warped[...,np.newaxis]
            else:
                img_transformed[img_idx] = image_warped
        return img_transformed

    def transform_images(self, images, extrapolation='constant', interpolation='linear'):

        return self.transform_images_float(images, extrapolation, interpolation)

    def invtransform_images(self, images, extrapolation='constant', interpolation='linear'):
        '''
        Apply inverse transformation to images or ground-truth masks
        :param images: raw image or gt masks    B*H*W*C
        :param extrapolation: extrapolation mode
        :param interpolation: interpolation mode
        :return:
        '''
        batchsize, H, W, _ = images.shape
        T_norm = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])  # normalize image center to origin
        T_unnorm = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]])  # unnormalize image back to image center
        img_transformed = np.zeros_like(images) # transformed images
        for img_idx, image in enumerate(images):
            # Apply transform to each image. extrapolation mode controls the way to extrapolate images.
            image_warped = cv2.warpAffine(image, (np.linalg.inv(T_unnorm@self.Tform[img_idx]@T_norm))[:2],
                dsize=(W,H), flags=CV2_INTERPOLATION[interpolation],
                borderMode=CV2_EXTRAPOLATION[extrapolation], borderValue=0)
            if len(image_warped.shape) <3:
                img_transformed[img_idx] = image_warped[...,np.newaxis]
            else:
                img_transformed[img_idx] = image_warped
        return img_transformed


    def transform_image_tensor(self, images, Tforms):
        '''
        Transform pytorch tensor image
        :param img: B*H*W*C input image tensors
        :param T:   B*3*3   transformation matrices
        :return: img_t: transformed images B*H*W*C
        '''

        batchsize, H, W, _ = images.shape
        T_norm = torch.tensor([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]],dtype=torch.float32,device=images.device)  # normalize image center to origin
        T_unnorm = torch.tensor([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]],dtype=torch.float32,device=images.device)  # unnormalize image back to image center
        image_tform = []
        # transform each input image
        for image, T in zip(images,Tforms):
            # mask1_ts = torch.tensor(img, dtype=torch.float32)[None, ...].permute([0, 3, 1, 2])
            image = image.permute(2,0,1)[None,...]
            # image_tform.append(tg.warp_affine(image, torch.tensor((T_unnorm @ T.cpu().detach().numpy() @ T_norm)[0:2, :], dtype=torch.float32, device=image.device)[None, ...], (H, W)))
            image_tform.append(self.affine_tensor(image, T_unnorm @ T @ T_norm))
        return torch.cat(image_tform,dim=0)

    def invtransform_image_tensor(self, images, Tforms):
        '''
        Inverse transform pytorch tensor image
        :param img: B*C*H*W input image tensors
        :param T:   B*3*3   transformation matrices
        :return: img_t: transformed images B*C*H*W
        '''

        batchsize, _, H, W = images.shape
        T_norm = np.array([[1, 0, -W / 2], [0, 1, -H / 2], [0, 0, 1]])  # normalize image center to origin
        T_unnorm = np.array([[1, 0, W / 2], [0, 1, H / 2], [0, 0, 1]])  # unnormalize image back to image center
        image_tform = []
        # transform each input image
        for img_idx, (image, T) in enumerate(zip(images,Tforms)):
            if not isinstance(T,np.ndarray):
                T = T.detach().cpu().numpy()
            # mask1_ts = torch.tensor(img, dtype=torch.float32)[None, ...].permute([0, 3, 1, 2])
            image = tg.warp_affine(image[None,...], torch.tensor(np.linalg.inv(T_unnorm @ T @ T_norm)[0:2, :],
                                                                  device=images.device,dtype=torch.float32)[None, ...], (H, W))

            # Apply elastic deformation backward
            if 'ElasticDeform' in self.operations:
                image = self.elastic_deform_tensor(image, -self.ElasticDeformFlow[img_idx])

            image_tform.append(image)

        return torch.cat(image_tform,dim=0)

    def elastic_deform_v1(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a tensor
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        ElasticDeformFlow = np.einsum('ij->ji',np.reshape(ElasticDeformFlow,[2,-1]))
        H, W, C = image.shape
        img_vec = np.reshape(image,[-1,C])
        img_vec_aug = np.concatenate([np.zeros([1,C]), img_vec], axis=0)  # N+1*C
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = np.arange(0, H)
        y = np.arange(0, W)
        coord_x_hat, coord_y_hat = np.meshgrid(x, y)
        p_hat = np.stack([np.reshape(coord_x_hat, [-1]), np.reshape(coord_y_hat, [-1])], axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        h_gap = np.stack([np.ceil(p[:, 0]) - p[:, 0], p[:, 0] - np.floor(p[:, 0])], axis=-1)  # HW*2
        w_gap = np.stack([np.ceil(p[:, 1]) - p[:, 1], p[:, 1] - np.floor(p[:, 1])], axis=-1)  # HW*2

        I_nn = np.zeros([H * W, 2, 2, C])  # neighbors for interpolation   HW*2*2*3

        h_fl = np.floor(p[:, 0])
        h_cl = np.floor(p[:, 0])
        w_fl = np.floor(p[:, 1])
        w_cl = np.floor(p[:, 1])

        h_fl_validx = (h_fl >= 0) * (h_fl < H)
        h_cl_validx = (h_cl >= 0) * (h_cl < H)
        w_fl_validx = (w_fl >= 0) * (w_fl < W)
        w_cl_validx = (w_cl >= 0) * (w_cl < W)

        idx_lu = (h_fl * W + w_fl + 1) * h_fl_validx * w_fl_validx
        idx_lb = (h_fl * W + w_cl + 1) * h_fl_validx * w_cl_validx
        idx_ru = (h_cl * W + w_fl + 1) * h_cl_validx * w_fl_validx
        idx_rb = (h_cl * W + w_cl + 1) * h_cl_validx * w_cl_validx

        I_nn_lu = img_vec_aug[idx_lu.astype(int)]
        I_nn_lb = img_vec_aug[idx_lb.astype(int)]
        I_nn_ru = img_vec_aug[idx_ru.astype(int)]
        I_nn_rb = img_vec_aug[idx_rb.astype(int)]

        I_nn[:, 0, 0, :] = I_nn_lu
        I_nn[:, 0, 1, :] = I_nn_ru
        I_nn[:, 1, 0, :] = I_nn_lb
        I_nn[:, 1, 1, :] = I_nn_rb

        # interpolate deformed image
        img_hat = np.einsum('ij,ijkl->ikl', h_gap, I_nn)  # HW*2*3
        img_hat = np.einsum('ijk,ij->ik', img_hat, w_gap)

        img_hat = np.reshape(img_hat, [H, W, C])

        return img_hat

    def elastic_deform_v2(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a tensor
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        ElasticDeformFlow = np.einsum('ij->ji',np.reshape(ElasticDeformFlow,[2,-1]))
        H, W, C = image.shape
        # img_vec = np.reshape(image,[-1,C])
        # img_vec_aug = np.concatenate([np.zeros([1,C]), img_vec], axis=0)  # N+1*C
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = np.arange(0, H)
        y = np.arange(0, W)
        coord_x_hat, coord_y_hat = np.meshgrid(x, y)
        p_hat = np.stack([np.reshape(coord_x_hat, [-1]), np.reshape(coord_y_hat, [-1])], axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        h_gap = np.stack([np.ceil(p[:, 0]) - p[:, 0], p[:, 0] - np.floor(p[:, 0])], axis=-1)  # HW*2
        w_gap = np.stack([np.ceil(p[:, 1]) - p[:, 1], p[:, 1] - np.floor(p[:, 1])], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = int

        x_f = np.floor(p[:, 0]).astype(dtype_long)
        x_c = x_f + 1

        y_f = np.floor(p[:, 1]).astype(dtype_long)
        y_c = y_f + 1

        x_f = np.clip(x_f, 0, H - 1)
        x_c = np.clip(x_c, 0, H - 1)

        y_f = np.clip(y_f, 0, W - 1)
        y_c = np.clip(y_c, 0, W - 1)

        Ia = image[x_f, y_f]
        Ib = image[x_f, y_c]
        Ic = image[x_c, y_f]
        Id = image[x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_hat = Ia * wa[:,np.newaxis] + Ib * wb[:,np.newaxis] + Ic * wc[:,np.newaxis] + Id * wd[:,np.newaxis]

        img_hat = np.reshape(img_hat,[H,W,-1])

        return img_hat

    def elastic_deform_v3(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a numpy array image.
        borderMode ~ 'reflective'
        'reflective' means ...abcdcbabcdcbabcd...
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        image = torch.tensor(image)
        ElasticDeformFlow = torch.tensor(ElasticDeformFlow)
        ElasticDeformFlow = torch.t(ElasticDeformFlow.view([2,-1]))

        H, W, C = image.shape
        # img_vec = img.view([-1, 3])
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H))
        y = torch.tensor(np.arange(0, W))
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        # import pdb; pdb.set_trace()
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - (torch.ceil(p[:, 0])-1.)], axis=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - (torch.ceil(p[:, 1])-1.)], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = torch.LongTensor

        x_f = torch.floor(p[:, 0]).type(dtype_long)
        x_c = x_f + 1

        y_f = torch.floor(p[:, 1]).type(dtype_long)
        y_c = y_f + 1

        x_f = torch.clamp(x_f, 0, H - 1)
        x_c = torch.clamp(x_c, 0, H - 1)

        y_f = torch.clamp(y_f, 0, W - 1)
        y_c = torch.clamp(y_c, 0, W - 1)

        Ia = image[x_f, y_f]
        Ib = image[x_f, y_c]
        Ic = image[x_c, y_f]
        Id = image[x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_new = torch.t((torch.t(Ia) * wa)) + torch.t((torch.t(Ib) * wb)) + torch.t((torch.t(Ic) * wc)) + torch.t(
            (torch.t(Id) * wd))

        img_hat = np.reshape(img_new.numpy(), [H, W, C])

        return img_hat

    def elastic_deform_v4(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a numpy array image
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        image = torch.tensor(image)
        ElasticDeformFlow = torch.tensor(ElasticDeformFlow)
        ElasticDeformFlow = torch.t(ElasticDeformFlow.view([2,-1]))

        H, W, C = image.shape
        # img_vec = img.view([-1, 3])
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H))
        y = torch.tensor(np.arange(0, W))
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        # import pdb; pdb.set_trace()
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - (torch.ceil(p[:, 0])-1.)], axis=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - (torch.ceil(p[:, 1])-1.)], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = torch.LongTensor

        x_f = torch.floor(p[:, 0]).type(dtype_long)
        x_c = x_f + 1

        y_f = torch.floor(p[:, 1]).type(dtype_long)
        y_c = y_f + 1

        x_f = torch.clamp(x_f, 0, H - 1)
        x_c = torch.clamp(x_c, 0, H - 1)

        y_f = torch.clamp(y_f, 0, W - 1)
        y_c = torch.clamp(y_c, 0, W - 1)

        Ia = image[x_f, y_f]
        Ib = image[x_f, y_c]
        Ic = image[x_c, y_f]
        Id = image[x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_new = torch.t((torch.t(Ia) * wa)) + torch.t((torch.t(Ib) * wb)) + torch.t((torch.t(Ic) * wc)) + torch.t(
            (torch.t(Id) * wd))

        img_hat = np.reshape(img_new.numpy(), [H, W, C])

        return img_hat

    def elastic_deform_v5(self, image, ElasticDeformFlow):
        '''
        Elastic Deformation
        inputs: img ~ H*W*C; disp ~ H*W*2; borderMode ~ 'empty', 'reflective'
        'reflective' means ...abcdcbabcdcbabcd...
        '''

        image = torch.tensor(image,dtype=torch.float32)
        H, W, C = image.shape
        img_vec = image.view([-1, C]) # N*3
        img_vec_aug = img_vec     # N*3

        ElasticDeformFlow = torch.tensor(ElasticDeformFlow,dtype=torch.float32)
        ElasticDeformFlow = torch.t(ElasticDeformFlow.view([2,-1]))

        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.arange(0,H,dtype=torch.float32)
        y = torch.arange(0,W,dtype=torch.float32)
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat,[-1]),torch.reshape(coord_y_hat,[-1])],axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow   # before deformation coordinates
        h_gap = torch.stack([torch.ceil(p[:,0])-p[:,0], p[:,0]-(torch.ceil(p[:,0])-1)],axis=-1)    # HW*2
        w_gap = torch.stack([torch.ceil(p[:,1])-p[:,1], p[:,1]-(torch.ceil(p[:,1])-1)],axis=-1)    # HW*2

        I_nn = torch.zeros([H*W,2,2,C],dtype=torch.float32) # neighbors for interpolation   HW*2*2*3

        h_fl = torch.floor(p[:,0])
        h_cl = torch.floor(p[:,0])
        w_fl = torch.floor(p[:,1])
        w_cl = torch.floor(p[:,1])

        def ReflectiveIndex(idx, N):
            '''
            idx ~ M, N ~ int
            max(idx)<N
            '''
            idx = np.abs(idx)

            idx_reflective = (idx // N % 2) * (N - idx % N) + (1 - idx // N % 2) * (idx % N)

            return idx_reflective


        # find reflective idx
        h_cl = ReflectiveIndex(h_cl,H)
        w_cl = ReflectiveIndex(w_cl,W)
        h_fl = ReflectiveIndex(h_fl,H)
        w_fl = ReflectiveIndex(w_fl,W)

        h_cl = torch.clamp(h_cl, 0, H - 1)
        h_fl = torch.clamp(h_fl, 0, H - 1)
        w_cl = torch.clamp(w_cl, 0, W - 1)
        w_fl = torch.clamp(w_fl, 0, W - 1)

        idx_lu = (h_fl * W + w_fl)
        idx_lb = (h_fl * W + w_cl)
        idx_ru = (h_cl * W + w_fl)
        idx_rb = (h_cl * W + w_cl)

        I_nn_lu = img_vec_aug[idx_lu.to(torch.long)]
        I_nn_lb = img_vec_aug[idx_lb.to(torch.long)]
        I_nn_ru = img_vec_aug[idx_ru.to(torch.long)]
        I_nn_rb = img_vec_aug[idx_rb.to(torch.long)]

        I_nn[:,0,0,:] = I_nn_lu
        I_nn[:,0,1,:] = I_nn_ru
        I_nn[:,1,0,:] = I_nn_lb
        I_nn[:,1,1,:] = I_nn_rb

        # interpolate deformed image
        img_hat = torch.einsum('ij,ijkl->ikl',h_gap,I_nn)   # HW*2*3
        img_hat = torch.einsum('ijk,ij->ik',img_hat,w_gap)

        img_hat = img_hat.numpy()
        img_hat = np.reshape(img_hat,[H,W,C])

        return img_hat

    def elastic_deform(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a numpy array image
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        img_hat= self.elastic_deform_v5(image, ElasticDeformFlow)

        return img_hat

    def elastic_deform_tensor(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a tensor array image
        Inputs: images ~ H*W*C, ElasticDeformFlow ~ H*W*2
        '''

        # image = torch.tensor(image)
        device = image.device
        ElasticDeformFlow = torch.tensor(ElasticDeformFlow, device=device)
        ElasticDeformFlow = torch.t(ElasticDeformFlow.view([2,-1]))

        B, C, H, W = image.shape
        # img_vec = img.view([-1, 3])
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H), device=device)
        y = torch.tensor(np.arange(0, W), device=device)
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        # import pdb; pdb.set_trace()
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - (torch.ceil(p[:, 0])-1.)], axis=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - (torch.ceil(p[:, 1])-1.)], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = torch.LongTensor

        x_f = torch.floor(p[:, 0]).type(dtype_long)
        x_c = x_f + 1

        y_f = torch.floor(p[:, 1]).type(dtype_long)
        y_c = y_f + 1

        x_f = torch.clamp(x_f, 0, H - 1)
        x_c = torch.clamp(x_c, 0, H - 1)

        y_f = torch.clamp(y_f, 0, W - 1)
        y_c = torch.clamp(y_c, 0, W - 1)

        Ia = image[:,:,x_f, y_f]
        Ib = image[:,:,x_f, y_c]
        Ic = image[:,:,x_c, y_f]
        Id = image[:,:,x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_new = Ia * wa +  Ib * wb + Ic * wc + Id * wd

        img_hat = img_new.view([B,C,H,W])

        # img_hat = np.reshape(img_new.numpy(), [H, W, C])

        return img_hat

    def elastic_deform_tensor_v1(self, image, ElasticDeformFlow):
        '''
        Elastic Deform a tensor
        Inputs: images ~ B*C*H*W, ElasticDeformFlow ~ B*H*W*2
        '''

        device = image.device
        ElasticDeformFlow = torch.einsum('ij->ji', torch.tensor(ElasticDeformFlow,device=device).view([2, -1]))
        _, C, H, W = image.shape
        img_vec = image.view([-1,C])  # 3*N
        img_vec_aug = torch.cat([torch.zeros([1,C],device=device), img_vec], dim=0)  # N+1*C

        # img_vec_aug = torch.cat([torch.tensor([[0, 0, 0]]), img_vec], dim=0)  # N+1*3
        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H),device=device)
        y = torch.tensor(np.arange(0, W),device=device)
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], dim=-1)

        # bilinear interpolation
        p = p_hat - ElasticDeformFlow  # before deformation coordinates
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - torch.floor(p[:, 0])], dim=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - torch.floor(p[:, 1])], dim=-1)  # HW*2

        I_nn = torch.zeros([H * W, 2, 2, C], dtype=torch.double, device=device)  # neighbors for interpolation   HW*2*2*3

        h_fl = torch.floor(p[:, 0])
        h_cl = torch.floor(p[:, 0])
        w_fl = torch.floor(p[:, 1])
        w_cl = torch.floor(p[:, 1])

        h_fl_validx = (h_fl >= 0) * (h_fl < H)
        h_cl_validx = (h_cl >= 0) * (h_cl < H)
        w_fl_validx = (w_fl >= 0) * (w_fl < W)
        w_cl_validx = (w_cl >= 0) * (w_cl < W)

        idx_lu = (h_fl * H + w_fl + 1) * h_fl_validx * w_fl_validx
        idx_lb = (h_fl * H + w_cl + 1) * h_fl_validx * w_cl_validx
        idx_ru = (h_cl * H + w_fl + 1) * h_cl_validx * w_fl_validx
        idx_rb = (h_cl * H + w_cl + 1) * h_cl_validx * w_cl_validx

        I_nn_lu = img_vec_aug[idx_lu.to(torch.long)]
        I_nn_lb = img_vec_aug[idx_lb.to(torch.long)]
        I_nn_ru = img_vec_aug[idx_ru.to(torch.long)]
        I_nn_rb = img_vec_aug[idx_rb.to(torch.long)]

        I_nn[:, 0, 0, :] = I_nn_lu
        I_nn[:, 0, 1, :] = I_nn_ru
        I_nn[:, 1, 0, :] = I_nn_lb
        I_nn[:, 1, 1, :] = I_nn_rb

        # interpolate deformed image
        img_hat = torch.einsum('ij,ijkl->ikl', h_gap, I_nn)  # HW*2*3
        img_hat = torch.einsum('ijk,ij->ik', img_hat, w_gap)
        img_hat = img_hat.view([1,C,H,W])
        # img_hat = img_hat.numpy()
        # img_hat = np.reshape(img_hat, [H, W, 3]).astype(int)

        return img_hat

    def affine_tensor_yj(self, image, T):
        ''' Apply Affine transformation to tensor image
        '''
        device = image.device
        B, C, H, W = image.shape

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H), device=device)
        y = torch.tensor(np.arange(0, W), device=device)
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # find coordinates before transformation
        p_hat_hm = torch.cat([p_hat,torch.ones([p_hat.shape[0],1],device=p_hat.device)],axis=1)
        p_hm = torch.einsum('ij,kj->ik',T,p_hat_hm)
        p = p_hm[0:2].permute(1,0)   # before deformation coordinates

        # bilinear interpolation
        # import pdb; pdb.set_trace()
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - torch.floor(p[:, 0])], axis=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - torch.floor(p[:, 1])], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = torch.LongTensor

        x_f = torch.floor(p[:, 0]).type(dtype_long)
        x_c = x_f + 1

        y_f = torch.floor(p[:, 1]).type(dtype_long)
        y_c = y_f + 1

        x_f = torch.clamp(x_f, 0, H - 1)
        x_c = torch.clamp(x_c, 0, H - 1)

        y_f = torch.clamp(y_f, 0, W - 1)
        y_c = torch.clamp(y_c, 0, W - 1)

        Ia = image[:, :, x_f, y_f]
        Ib = image[:, :, x_f, y_c]
        Ic = image[:, :, x_c, y_f]
        Id = image[:, :, x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_new = Ia * wa + Ib * wb + Ic * wc + Id * wd

        img_hat = img_new.view([B, C, H, W])

        return img_hat

    def affine_tensor_yj_v1(self, image, T):
        '''
        Apply Affine transformation to tensor image
        periodic extrapolation  ABCABCABC
        '''
        device = image.device
        B, C, H, W = image.shape

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H), device=device)
        y = torch.tensor(np.arange(0, W), device=device)
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # find coordinates before transformation
        p_hat_hm = torch.cat([p_hat,torch.ones([p_hat.shape[0],1],device=p_hat.device)],axis=1)
        p_hm = torch.einsum('ij,kj->ik',T,p_hat_hm)
        p = p_hm[0:2].permute(1,0)   # before deformation coordinates

        # bilinear interpolation
        # import pdb; pdb.set_trace()
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - torch.floor(p[:, 0])], axis=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - torch.floor(p[:, 1])], axis=-1)  # HW*2

        # I_nn = torch.zeros([H*W,2,2,3],dtype=torch.double) # neighbors for interpolation   HW*2*2*3

        dtype_long = torch.LongTensor

        x_f = torch.floor(p[:, 0]).type(dtype_long)
        x_c = x_f + 1

        y_f = torch.floor(p[:, 1]).type(dtype_long)
        y_c = y_f + 1

        x_f = torch.remainder(x_f, H)
        x_c = torch.remainder(x_c, H)

        y_f = torch.remainder(y_f, W)
        y_c = torch.remainder(y_c, W)

        Ia = image[:, :, x_f, y_f]
        Ib = image[:, :, x_f, y_c]
        Ic = image[:, :, x_c, y_f]
        Id = image[:, :, x_c, y_c]

        # import pdb; pdb.set_trace()

        # torch.cat((Ia, Ib, Ic, Id), 0)
        wa = h_gap[:, 0] * w_gap[:, 0]
        wb = h_gap[:, 0] * w_gap[:, 1]
        wc = h_gap[:, 1] * w_gap[:, 0]
        wd = h_gap[:, 1] * w_gap[:, 1]

        img_new = Ia * wa + Ib * wb + Ic * wc + Id * wd

        img_hat = img_new.view([B, C, H, W])

        return img_hat

    def affine_tensor_xx(self, image, T):
        ''' Apply Affine transformation to tensor image
        '''
        device = image.device
        B, C, H, W = image.shape
        img_vec = image.view([C,-1]).permute(1,0) # B*N*C
        img_vec_aug = torch.cat([torch.zeros([1, C], device=device), img_vec], dim=0)  # N+1*C

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H), device=device)
        y = torch.tensor(np.arange(0, W), device=device)
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # find coordinates before transformation
        p_hat_hm = torch.cat([p_hat,torch.ones([p_hat.shape[0],1],device=p_hat.device)],axis=1)
        p_hm = torch.einsum('ij,kj->ik',T,p_hat_hm)
        p = p_hm[0:2].permute(1,0)   # before deformation coordinates

        # bilinear interpolation
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - torch.floor(p[:, 0])], dim=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - torch.floor(p[:, 1])], dim=-1)  # HW*2

        I_nn = torch.zeros([H * W, 2, 2, C], dtype=torch.float32,
                           device=device)  # neighbors for interpolation   HW*2*2*3

        h_fl = torch.floor(p[:, 0])
        h_cl = torch.floor(p[:, 0])
        w_fl = torch.floor(p[:, 1])
        w_cl = torch.floor(p[:, 1])

        h_fl_validx = (h_fl >= 0) * (h_fl < H)
        h_cl_validx = (h_cl >= 0) * (h_cl < H)
        w_fl_validx = (w_fl >= 0) * (w_fl < W)
        w_cl_validx = (w_cl >= 0) * (w_cl < W)

        idx_lu = (h_fl * H + w_fl + 1) * h_fl_validx * w_fl_validx
        idx_lb = (h_fl * H + w_cl + 1) * h_fl_validx * w_cl_validx
        idx_ru = (h_cl * H + w_fl + 1) * h_cl_validx * w_fl_validx
        idx_rb = (h_cl * H + w_cl + 1) * h_cl_validx * w_cl_validx

        I_nn_lu = img_vec_aug[idx_lu.to(torch.long)]
        I_nn_lb = img_vec_aug[idx_lb.to(torch.long)]
        I_nn_ru = img_vec_aug[idx_ru.to(torch.long)]
        I_nn_rb = img_vec_aug[idx_rb.to(torch.long)]

        I_nn[:, 0, 0, :] = I_nn_lu
        I_nn[:, 0, 1, :] = I_nn_ru
        I_nn[:, 1, 0, :] = I_nn_lb
        I_nn[:, 1, 1, :] = I_nn_rb

        # interpolate deformed image
        img_hat = torch.einsum('ij,ijkl->ikl', h_gap, I_nn)  # HW*2*3
        img_hat = torch.einsum('ijk,ij->ik', img_hat, w_gap)
        img_hat = img_hat.view([1, C, H, W])
        # img_hat = img_hat.numpy()
        # img_hat = np.reshape(img_hat, [H, W, 3]).astype(int)

        return img_hat

    def affine_tensor_reflective_bk(self, image, T):
        '''
        Apply Affine transformation to tensor image. Reflective extrapolation.
        '''
        device = image.device
        B, C, H, W = image.shape
        img_vec = image.view([C,-1]).permute(1,0) # B*N*C
        img_vec_aug = torch.cat([torch.zeros([1, C], device=device), img_vec], dim=0)  # N+1*C

        # coordinates after deformation
        x = torch.tensor(np.arange(0, H), device=device)
        y = torch.tensor(np.arange(0, W), device=device)
        # import pdb; pdb.set_trace()
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat, [-1]), torch.reshape(coord_y_hat, [-1])], axis=-1)

        # find coordinates before transformation
        p_hat_hm = torch.cat([p_hat,torch.ones([p_hat.shape[0],1],device=p_hat.device)],axis=1)
        p_hm = torch.einsum('ij,kj->ik',T,p_hat_hm)
        p = p_hm[0:2].permute(1,0)   # before deformation coordinates

        # bilinear interpolation
        h_gap = torch.stack([torch.ceil(p[:, 0]) - p[:, 0], p[:, 0] - torch.floor(p[:, 0])], dim=-1)  # HW*2
        w_gap = torch.stack([torch.ceil(p[:, 1]) - p[:, 1], p[:, 1] - torch.floor(p[:, 1])], dim=-1)  # HW*2

        I_nn = torch.zeros([H * W, 2, 2, C], dtype=torch.float32,
                           device=device)  # neighbors for interpolation   HW*2*2*3

        h_fl = torch.floor(p[:, 0])
        h_cl = torch.floor(p[:, 0])
        w_fl = torch.floor(p[:, 1])
        w_cl = torch.floor(p[:, 1])

        h_fl_validx = (h_fl >= 0) * (h_fl < H)
        h_cl_validx = (h_cl >= 0) * (h_cl < H)
        w_fl_validx = (w_fl >= 0) * (w_fl < W)
        w_cl_validx = (w_cl >= 0) * (w_cl < W)

        idx_lu = (h_fl * H + w_fl + 1) * h_fl_validx * w_fl_validx
        idx_lb = (h_fl * H + w_cl + 1) * h_fl_validx * w_cl_validx
        idx_ru = (h_cl * H + w_fl + 1) * h_cl_validx * w_fl_validx
        idx_rb = (h_cl * H + w_cl + 1) * h_cl_validx * w_cl_validx

        I_nn_lu = img_vec_aug[idx_lu.to(torch.long)]
        I_nn_lb = img_vec_aug[idx_lb.to(torch.long)]
        I_nn_ru = img_vec_aug[idx_ru.to(torch.long)]
        I_nn_rb = img_vec_aug[idx_rb.to(torch.long)]

        I_nn[:, 0, 0, :] = I_nn_lu
        I_nn[:, 0, 1, :] = I_nn_ru
        I_nn[:, 1, 0, :] = I_nn_lb
        I_nn[:, 1, 1, :] = I_nn_rb

        # interpolate deformed image
        img_hat = torch.einsum('ij,ijkl->ikl', h_gap, I_nn)  # HW*2*3
        img_hat = torch.einsum('ijk,ij->ik', img_hat, w_gap)
        img_hat = img_hat.view([1, C, H, W])
        # img_hat = img_hat.numpy()
        # img_hat = np.reshape(img_hat, [H, W, 3]).astype(int)

        return img_hat

    def affine_tensor_reflective(self, image, T):
        '''
        Affine Transformation
        inputs: img ~ H*W*C; disp ~ H*W*2; borderMode ~ 'empty', 'reflective'
        'reflective' means ...abcdcbabcdcbabcd...
        'reflective_v1' means ...abcddcbaabcddcba...
        '''

        # H, W, C = image.shape
        B, C, H, W = image.shape
        img_vec = image.view([B, -1, C]) # B*HW*C
        T = torch.tensor(T, dtype=torch.float32)

        img_vec_aug = img_vec     # N*3

        # img_hat = torch.zeros_like(img)

        # coordinates after deformation
        x = torch.tensor(np.arange(0,H))
        y = torch.tensor(np.arange(0,W))
        coord_x_hat, coord_y_hat = torch.meshgrid(x, y)
        p_hat = torch.stack([torch.reshape(coord_x_hat,[-1]),torch.reshape(coord_y_hat,[-1])],axis=-1)
        p_hat = torch.cat([p_hat,torch.ones(p_hat.shape[0],1)],axis=-1)

        # bilinear interpolation
        p = torch.einsum('ij,kj->ik', torch.inverse(T), p_hat)[0:2,:]   # before deformation coordinates
        p = torch.einsum('ij->ji',p)
        h_gap = torch.stack([torch.ceil(p[:,0])-p[:,0], p[:,0]-torch.floor(p[:,0])],axis=-1)    # HW*2
        w_gap = torch.stack([torch.ceil(p[:,1])-p[:,1], p[:,1]-torch.floor(p[:,1])],axis=-1)    # HW*2

        I_nn = torch.zeros([H*W,2,2,3],dtype=torch.float32) # neighbors for interpolation   HW*2*2*3

        h_fl = torch.floor(p[:,0])
        h_cl = torch.floor(p[:,0])
        w_fl = torch.floor(p[:,1])
        w_cl = torch.floor(p[:,1])

        h_fl_validx = (h_fl>=0)*(h_fl<H)
        h_cl_validx = (h_cl>=0)*(h_cl<H)
        w_fl_validx = (w_fl>=0)*(w_fl<W)
        w_cl_validx = (w_cl>=0)*(w_cl<W)

        def ReflectiveIndex(idx, N):
            '''
            idx ~ M, N ~ int
            max(idx)<N
            '''
            idx = np.abs(idx)

            idx_reflective = (idx // N % 2) * (N - idx % N) + (1 - idx // N % 2) * (idx % N)

            return idx_reflective

        # find reflective idx
        h_cl = ReflectiveIndex(h_cl,H)
        w_cl = ReflectiveIndex(w_cl,W)
        h_fl = ReflectiveIndex(h_fl,H)
        w_fl = ReflectiveIndex(w_fl,W)


        h_cl = torch.clamp(h_cl,0,H-1)
        h_fl = torch.clamp(h_fl,0,H-1)
        w_cl = torch.clamp(w_cl,0,W-1)
        w_fl = torch.clamp(w_fl,0,W-1)

        idx_lu = (h_fl * H + w_fl)
        idx_lb = (h_fl * H + w_cl)
        idx_ru = (h_cl * H + w_fl)
        idx_rb = (h_cl * H + w_cl)

        I_nn_lu = img_vec_aug[idx_lu.to(torch.long)]
        I_nn_lb = img_vec_aug[idx_lb.to(torch.long)]
        I_nn_ru = img_vec_aug[idx_ru.to(torch.long)]
        I_nn_rb = img_vec_aug[idx_rb.to(torch.long)]

        I_nn[:,0,0,:] = I_nn_lu
        I_nn[:,0,1,:] = I_nn_ru
        I_nn[:,1,0,:] = I_nn_lb
        I_nn[:,1,1,:] = I_nn_rb

        # interpolate deformed image
        img_hat = torch.einsum('ij,ijkl->ikl',h_gap,I_nn)   # HW*2*3
        img_hat = torch.einsum('ijk,ij->ik',img_hat,w_gap)

        img_hat = img_hat.numpy()
        img_hat = np.reshape(img_hat,[H,W,C])

        return img_hat

    def affine_tensor(self, image, T):
        ''' Apply Affine transformation to tensor image
        '''

        img_hat = self.affine_tensor_yj_v1(image,T)
        # img_hat = self.affine_tensor_reflective(image,T)

        return img_hat


    def __get_transformation_matrix(self, imgsize, inverse=False):
        '''
        get full transformation matrix
        :param imgsize:
        :param inverse:
        :return:
        '''
        assert(self.imgcount != 0, 'call construct_random_transform on every batch before calling transform operations')
        T = np.zeros((self.imgcount, 3, 3))
        T[:] = self.identity[None,...]
        it = self.random_values if (inverse == False) else reversed(self.random_values)
        for op, random_ in it:
            if op == Op.FLIP_LR: #inverse of flip-lr is itself
                op_T = np.array([self.fliplr_matrix if r==True else self.identity for r in random_])
            elif op == Op.FLIP_UD:
                op_T = np.array([self.flipud_matrix if r==True else self.identity for r in random_])
            elif op == Op.ROTATE:
                theta = np.radians(random_)
                cos, sin = np.cos(theta), np.sin(theta)
                if inverse == True:
                    sin = -sin
                #this is counter-clockwise matrix, but due to y-inverse, it will do clockwise instead
                op_T = np.array([[[c,-s, 0], [s, c, 0], [0, 0, 1]] for c, s in zip(cos, sin)])
            elif op == Op.TRANSLATE_X:
                op_T = np.repeat(self.identity[None, ...], self.imgcount, axis=0) #first assign each identity matrix for all images
                if inverse == True:
                    random_ = -random_
                op_T[:, 0, 2] = random_ * imgsize[1] #replace x-translate value with generated random_ x imgsize
            elif op == Op.TRANSLATE_Y:
                op_T = np.repeat(self.identity[None, ...], self.imgcount, axis=0) #first assign each identity matrix for all images
                if inverse == True:
                    random_ = -random_
                op_T[:, 1, 2] = random_ * imgsize[0] #replace y-translate value with generated random_ x imgsize
            T = op_T @ T
        return T

    def __transform_images_with_inversemap(self, images, inv_map, mode, interpolation):
        self.repositionimage_center_to_origin(inv_map, images.shape[1:3])
        dsize = (
            int(np.round(images.shape[2])),
            int(np.round(images.shape[1]))
        )
        img_transformed = np.zeros_like(images)
        for img_idx, image in enumerate(images):
            #img_transformed[img_idx, :] = warp(image, inv_map[img_idx], mode=mode, order=SK_INTERPOLATION[interpolation], preserve_range=True)
            image_warped = cv2.warpAffine(image, inv_map[img_idx][:2],
                dsize=dsize, flags=CV2_INTERPOLATION[interpolation],
                borderMode=CV2_EXTRAPOLATION[mode], borderValue=0)
            image_warped = np.atleast_3d(image_warped)
            img_transformed[img_idx, :] = image_warped
        '''
        img_transformed = np.zeros_like(images)
        manual_start = timer()
        for img_idx, image in enumerate(images):
            for j, col in enumerate(image):
                for i, _row in enumerate(col):
                    img_transformed[img_idx, i, j, :] = self.nearest_neighbors(j, i, image, inv_map[img_idx])
        manual_time = "{0:.2f}".format(timer() - manual_start)
        print(manual_time)
        '''

        return img_transformed

    def inv_transform_images(self, images, mode='constant', interpolation='linear'):
        T = self.__get_transformation_matrix(images.shape[1:3], inverse=False)
        return self.__transform_images_with_inversemap(images, T, mode=mode, interpolation=interpolation)

    def __get_grid(self, width, height, homogenous=False):
        coords = np.indices((width, height)).reshape(2, -1)
        return np.vstack((coords, np.ones(coords.shape[1]))) if homogenous else coords


    def inv_transform_tensors(self, t_images):
        T = self.__get_transformation_matrix(t_images.shape[1:3], inverse=False) #inverse of inverse
        self.repositionimage_center_to_origin(T, t_images.shape[1:3])
        height, width = t_images.shape[1:3]
        coords = self.__get_grid(width, height, True)
        x_ori, y_ori = coords[0], coords[1]

        original_image_tensor = torch.zeros_like(t_images)
        for img_idx, image in enumerate(t_images):
            warp_coords = np.round(T[img_idx]@coords).astype(np.int) #transformed coordinates
            xcoord, ycoord = warp_coords[0, :], warp_coords[1, :] #transformed coordinates
            indices = np.where((xcoord >= 0) & (xcoord < width) & #filtered indices, coordinates
                   (ycoord >= 0) & (ycoord < height))
            vx_transformed, vy_transformed = xcoord[indices].astype(np.int), ycoord[indices].astype(np.int)
            vx_original, vy_original = x_ori[indices].astype(np.int), y_ori[indices].astype(np.int)
            for i in range(t_images.shape[3]):
                original_image_tensor[(img_idx, vy_original, vx_original, i)] = image[(vy_transformed, vx_transformed, i)]#reverse transform
        return original_image_tensor