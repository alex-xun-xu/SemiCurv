#currently support affine only

from enum import Enum, auto
import numpy as np
from skimage.transform import warp
from timeit import default_timer as timer
import torch
import cv2
class Op(Enum):
    FLIP_LR = auto()
    FLIP_UD = auto()
    ROTATE = auto()
    TRANSLATE_X = auto()
    TRANSLATE_Y = auto()

CV2_MODE = {
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
        self.operations = []
        self.random_values = []
        self.matrix_templates = {}
        self.fliplr_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.flipud_matrix = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        self.identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
        self.imgcount = 0
    
    def add_fliplr(self, probability=0.5):
        self.operations.append((Op.FLIP_LR, probability))
    
    def add_flipud(self, probability=0.5):
        self.operations.append((Op.FLIP_UD, probability))

    def add_rotate(self, range):
        self.operations.append((Op.ROTATE, range))
    
    def add_translate_x(self, range):
        self.operations.append((Op.TRANSLATE_X, range))

    def add_translate_y(self, range):
        self.operations.append((Op.TRANSLATE_Y, range))

    #tranformation to apply from center of image as origin
    def repositionimage_center_to_origin(self, T, imgsize):
        #we don't apply inverse order for inverse_map, because we don't want inverse of reposition. we want the same reposition logic.
        T_movecenter = np.zeros_like(T)
        T_inv_move_center = np.zeros_like(T)
        T_movecenter[:] = T_inv_move_center[:] = self.identity[None,...]

        #negative translation of imagezie/2 will move image center to (0,0) 
        T_movecenter[:, 0, 2] = -(imgsize[1]-1)/2 
        T_movecenter[:, 1, 2] = -(imgsize[0]-1)/2

        T_inv_move_center[:, 0, 2] = (imgsize[1]-1)/2
        T_inv_move_center[:, 1, 2] = (imgsize[0]-1)/2

        T[:] = T_inv_move_center @ T @ T_movecenter #inplace

    def construct_random_transform(self, count, shuffle_order=True):
        op_indices = np.arange(len(self.operations))
        if shuffle_order == True:
            np.random.shuffle(op_indices)

        self.random_values = []
        self.imgcount = count
        for idx in op_indices:
            op, val = self.operations[idx]
            if op == Op.FLIP_LR:
                random_ = np.random.choice(a=[True, False], size=count, p=[val, 1-val])
            elif op == Op.FLIP_UD:
                random_ = np.random.choice(a=[True, False], size=count, p=[val, 1-val])
            elif op == Op.ROTATE:
                random_ = np.random.choice(a=val, size=count)
            elif op == Op.TRANSLATE_X:
                random_ = np.random.choice(a=val, size=count)
            elif op == Op.TRANSLATE_Y:
                random_ = np.random.choice(a=val, size=count)
            self.random_values.append((op, random_))

    def __get_transformation_matrix(self, imgsize, inverse=False):
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
                dsize=dsize, flags=CV2_INTERPOLATION[interpolation] | cv2.WARP_INVERSE_MAP,
                borderMode=CV2_MODE[mode], borderValue=0)
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

    def transform_images(self, images, mode='constant', interpolation='linear'):
        T = self.__get_transformation_matrix(images.shape[1:3], inverse=True)
        return self.__transform_images_with_inversemap(images, T, mode=mode, interpolation=interpolation)


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