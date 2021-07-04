import os
import numpy as np
import pandas as pd
import cv2
import shutil
import sys
from utils import calculate_angles

from landmarks_mapping_300w_wflw import list_wflw_to_300w

def rotate(angle, center, landmarks):
    rad = angle * np.pi / 180.0
    alpha = np.cos(rad)
    beta = np.sin(rad)
    M = np.zeros((2,3), dtype=np.float32)
    M[0, 0] = alpha
    M[0, 1] = beta
    M[0, 2] = (1-alpha)*center[0] - beta*center[1]
    M[1, 0] = -beta
    M[1, 1] = alpha
    M[1, 2] = beta*center[0] + (1-alpha)*center[1]

    rotated_landmarks = np.asarray([(M[0,0]*x+M[0,1]*y+M[0,2],
                                     M[1,0]*x+M[1,1]*y+M[1,2]) for (x,y) in landmarks])
    return M, rotated_landmarks

class DataPreprocessor():
    def __init__(self, out_dir, is_train, image_size=224):
        self.img_save_preffix = 0
        self.out_dir = out_dir
        self.is_train = is_train
        self.image_size = image_size
        self.annotation_paths = []
        self.annotation_landmarks = []
        self.annotation_angles = []
        landmarks_mirror_mapping = os.path.dirname(os.path.realpath(__file__)) + '/landmarks_mirrored_indices.txt'
        self.mirror_idx = []
        with open(landmarks_mirror_mapping, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            self.mirror_idx = lines[0].strip().split(',')
            self.mirror_idx = list(map(int, self.mirror_idx))

    def preprocess_image(self, path, landmarks, repeat):
        xy = np.min(landmarks, axis=0).astype(np.int32) 
        zz = np.max(landmarks, axis=0).astype(np.int32)
        wh = zz - xy + 1

        center = (xy + wh/2).astype(np.int32)
        img = cv2.imread(path)
        boxsize = int(np.max(wh) * 1.2)
        xy = center - boxsize // 2
        x1, y1 = xy
        x2, y2 = xy + boxsize
        height, width, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)

        imgT = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        imgT = cv2.resize(imgT, (self.image_size, self.image_size))
        landmarks_cropped = (landmarks - xy) / boxsize
        self.save_image(imgT, landmarks_cropped, path)

        if self.is_train:
            num_rotations = 0
            while num_rotations < repeat:
                angle = np.random.randint(-20, 20)
                cx, cy = center
                cx = cx + int(np.random.randint(-boxsize*0.1, boxsize*0.1))
                cy = cy + int(np.random.randint(-boxsize * 0.1, boxsize * 0.1))
                M, landmarks_rotated = rotate(angle, (cx,cy), landmarks)

                imgT = cv2.warpAffine(img, M, (int(img.shape[1] * 1.1), int(img.shape[0] * 1.1)))

                wh = np.ptp(landmarks_rotated, axis=0).astype(np.int32) + 1
                size = np.random.randint(int(np.min(wh)), np.ceil(np.max(wh) * 1.25))
                xy = np.asarray((cx - size // 2, cy - size//2), dtype=np.int32)
                landmarks_cropped = (landmarks_rotated - xy) / size
                if (landmarks_cropped < 0).any() or (landmarks_cropped > 1).any():
                    continue

                x1, y1 = xy
                x2, y2 = xy + size
                height, width, _ = imgT.shape
                dx = max(0, -x1)
                dy = max(0, -y1)
                x1 = max(0, x1)
                y1 = max(0, y1)

                edx = max(0, x2 - width)
                edy = max(0, y2 - height)
                x2 = min(width, x2)
                y2 = min(height, y2)

                imgT = imgT[y1:y2, x1:x2]
                if (dx > 0 or dy > 0 or edx >0 or edy > 0):
                    imgT = cv2.copyMakeBorder(imgT, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

                imgT = cv2.resize(imgT, (self.image_size, self.image_size))

                if np.random.choice((True, False)):
                    landmarks_cropped[:,0] = 1 - landmarks_cropped[:,0]
                    landmarks_cropped = landmarks_cropped[self.mirror_idx]
                    imgT = cv2.flip(imgT, 1)

                self.save_image(imgT, landmarks_cropped, path)
                num_rotations = num_rotations + 1

    def save_image(self, img, landmarks, path):
        _, filename = os.path.split(path)
        filename, _ = os.path.splitext(filename)
        filename = str(self.img_save_preffix) + '_' + filename + '.png'
        save_path = os.path.join(self.out_dir, filename)
        self.img_save_preffix= self.img_save_preffix + 1
        assert not os.path.exists(save_path), save_path
        cv2.imwrite(save_path, img)

        pitch, yaw, roll = calculate_angles(landmarks)
        euler_angles = np.asarray((pitch, yaw, roll), dtype=np.float32)

        self.annotation_paths.append(filename)
        self.annotation_landmarks.append(landmarks)
        self.annotation_angles.append(euler_angles)

    def save_annotations(self):
        df = pd.DataFrame({
            'path' : self.annotation_paths,
            'landmarks' : self.annotation_landmarks,
            'angles' : self.annotation_angles,
        })
        df['landmarks'] = df['landmarks'].apply(lambda x: x.flatten())
        save_path = os.path.join(self.out_dir, 'annotations.csv')
        df.to_csv(save_path)

    def prepare_data_wflw(self, images_dir, annotations_file):
        with open(annotations_file,'r') as f:
            lines = f.readlines()

            for i, line in enumerate(lines):
                line = line.strip().split()
                landmarks = np.reshape(list(map(float, line[:196])), (-1, 2))
                landmarks = list_wflw_to_300w(landmarks)
                landmarks = np.asarray(landmarks, dtype=np.float32)
                path = os.path.join(images_dir, line[206])
                self.preprocess_image(path, landmarks, 5)
    
    def prepare_data_300w(self, images_dir, use_all = True):
        for root, dirs, files in os.walk(images_dir, topdown=False):
            for file in files:
                if file.endswith('.png'):
                    filename, _ = os.path.splitext(file)
                    idx = int(filename[-1])
                    if (use_all or self.is_train == bool(idx % 2)):
                        annotation_file = filename + '.pts'
                        landmarks = []
                        with open(os.path.join(root, annotation_file),'r') as f:
                            lines = f.readlines()
                            for line in lines[3:71]:
                                line = line.strip().split()
                                landmarks.append(list(map(float, line)))
                        landmarks = np.asarray(landmarks, dtype=np.float32)
                        path = os.path.join(images_dir, root, file)
                        self.preprocess_image(path, landmarks, 20)

if __name__ == '__main__':
    root_dir = os.path.dirname(os.path.realpath(__file__))
    images_dir_wflw = root_dir + '/data/WFLW_images'
    images_dir_300w = root_dir + '/data/300W'
    test_annotation_files_wflw = root_dir + '/data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt'
    train_annotation_files_wflw = root_dir + '/data/WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'

    preprocessed_data_dir = root_dir + '/preprocessed_data'
    train_data_dir = preprocessed_data_dir + '/train'
    test_data_dir_300W = preprocessed_data_dir + '/test/300w'
    test_data_dir_300W_selected = preprocessed_data_dir + '/test/300w_selected'
    test_data_dir_wflw = preprocessed_data_dir + '/test/wflw'

    for dir in [train_data_dir, test_data_dir_300W, test_data_dir_300W_selected, test_data_dir_wflw]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
            os.makedirs(dir)

    train_preprocessor = DataPreprocessor(train_data_dir, is_train = True)
    train_preprocessor.prepare_data_wflw(images_dir_wflw, train_annotation_files_wflw)
    print('WFLW train set - Done')
    train_preprocessor.prepare_data_300w(images_dir_300w)
    print('300W train set - Done')
    train_preprocessor.save_annotations()
    print('train annotations - Done')

    test_preprocessor = DataPreprocessor(test_data_dir_wflw, is_train = False)
    test_preprocessor.prepare_data_wflw(images_dir_wflw, test_annotation_files_wflw)
    print('WFLW test set - Done')
    test_preprocessor.save_annotations()
    print('WFLW test annotations - Done')

    test_preprocessor = DataPreprocessor(test_data_dir_300W, is_train = False)
    test_preprocessor.prepare_data_300w(images_dir_300w, use_all = True)
    print('300w full test set - Done')
    test_preprocessor.save_annotations()
    print('300w full test annotations - Done')

    test_preprocessor = DataPreprocessor(test_data_dir_300W_selected, is_train = False)
    test_preprocessor.prepare_data_300w(images_dir_300w)
    print('300w selected test set - Done')
    test_preprocessor.save_annotations()
    print('300w selected annotations - Done')
    print('All done')