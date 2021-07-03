import os
import cv2
from ast import literal_eval
import shutil
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import onnx
import onnxruntime as onnxrt

from utils import custom_nme, calculate_and_plot_auc

def evaluate(test_data_dir, output_dir):
    annotations_df = pd.read_csv(test_data_dir + '/annotations.csv')
    annotations_df['path'] = annotations_df['path'].astype(str)
    annotations_df['landmarks'] = annotations_df['landmarks'].apply(lambda x: ','.join(x.split()))
    annotations_df['landmarks'] = annotations_df['landmarks'].apply(lambda x: x.replace('[,', '['))
    annotations_df['landmarks'] = annotations_df['landmarks'].apply(literal_eval)
    annotations_df['landmarks'] = annotations_df['landmarks'].apply(np.array)
    annotations_df['landmarks'] = annotations_df['landmarks'].apply(lambda x: x.flatten())
    annotations_df['angles'] = annotations_df['angles'].apply(lambda x: ','.join(x.split()))
    annotations_df['angles'] = annotations_df['angles'].apply(lambda x: x.replace('[,', '['))
    annotations_df['angles'] = annotations_df['angles'].apply(literal_eval)
    annotations_df['angles'] = annotations_df['angles'].apply(np.array)

    img_width, img_height = (224, 224)
    batch_size = 64

    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_dataframe(
        annotations_df,
        directory=test_data_dir,
        x_col = 'path', y_col = 'landmarks',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'raw')

    inference_times = []
    nme_inter_ocular = []
    nme_inter_pupil = []

    onnxSess = onnxrt.InferenceSession("trained_model.onnx")
    
    save_image_ixd = 0
    for i in range(len(test_generator)):
        image_batch, landmarks_batch = next(test_generator)
        for image, landmarks in zip(image_batch, landmarks_batch):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            local_start = time.perf_counter()
            predicted = onnxSess.run(None, {'input_1:0': image[np.newaxis]})[0]
            local_end = time.perf_counter()
            inference_times.append(local_end - local_start)

            landmarks = landmarks.reshape(-1, 2)
            predicted = predicted.reshape(-1, 2)

            inter_ocular, inter_pupil = custom_nme(landmarks, predicted)
            nme_inter_ocular.append(inter_ocular)
            nme_inter_pupil.append(inter_pupil)
            
            for x, y in landmarks:
                coords = (img_width * x + 0.5, img_height * y + 0.5)
                coords = [int(c) for c in coords]
                image = cv2.circle(image, coords, 2, (0, 255, 0))
            for x, y in predicted:
                coords = (img_width * x + 0.5, img_height * y + 0.5)
                coords = [int(c) for c in coords]
                image = cv2.circle(image, coords, 2, (0, 0, 255))
            save_path = os.path.join(output_dir, 'result ' + str(save_image_ixd) + '.png')
            cv2.imwrite(save_path, image)
            save_image_ixd = save_image_ixd + 1
    
    auc_inter_ocular = calculate_and_plot_auc(nme_inter_ocular, os.path.join(output_dir, 'auc_inter_ocular.png'))
    print("NME (inter-ocular) " + str(np.mean(nme_inter_ocular)))
    print("AUC (inter-ocular) " + str(auc_inter_ocular))

    auc_inter_pupil = calculate_and_plot_auc(nme_inter_pupil, os.path.join(output_dir, 'auc_inter_pupil.png'))
    print("NME (inter-pupil) " + str(np.mean(nme_inter_pupil)))
    print("AUC (inter-pupil) " + str(auc_inter_pupil))

    print("Mean inference time " + str(np.mean(inference_times)))

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_visible_devices(gpu, 'GPU')
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), 'Physical GPUs,', len(logical_gpus), 'Logical GPU')
        except RuntimeError as e:
            print(e)
    print(tf.config.list_physical_devices('GPU'))

    root_dir = os.path.dirname(os.path.realpath(__file__))
    test_data_dir = root_dir + '/preprocessed_data/test'
    test_data_dir_300w = test_data_dir + '/300w'
    test_data_dir_300w_selected = test_data_dir + '/300w_selected'
    test_data_dir_wflw = test_data_dir + '/wflw'

    output_dir = root_dir + '/output'
    output_dir_300w = output_dir + '/300w'
    output_dir_300w_selected = output_dir + '/300w_selected'
    output_dir_wflw = output_dir + '/wflw'

    for dir in [output_dir, output_dir_300w, output_dir_300w_selected, output_dir_wflw]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

    print('Evaluating on 300W:')
    evaluate(test_data_dir_300w, output_dir_300w)
    print('Evaluating on 300W (only images not used in training):')
    evaluate(test_data_dir_300w_selected, output_dir_300w_selected)
    print('Evaluating on WFLW:')
    evaluate(test_data_dir_wflw, output_dir_wflw)
        