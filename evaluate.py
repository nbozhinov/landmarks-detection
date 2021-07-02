import os
from ast import literal_eval
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import custom_nme
from model import CustomModel, CustomNME

def evaluate(test_data_dir, output_dir, model):
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

    test_datagen = ImageDataGenerator()
    test_generator = train_datagen.flow_from_dataframe(
        annotations_df,
        directory=test_data_dir,
        x_col = 'path', y_col = 'landmarks',
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'raw')

    inference_times = []
    inference_errors = []

    for i in range(len(test_generator)):
        image_batch, landmarks_batch = next(test_generator)
        pred = model.predict(image_batch)
        true = true.reshape(-1, 2)
        predicted = predicted.reshape(-1, 2)
        inference_errors.extend([
            custom_nme(true, predicted)
                for true, predicted in zip(landmarks_batch, pred)])
        for x, y in true:
            coords = (img_width, img_height) * (x, y) + (0.5, 0.5)
            coords = [int(c) for c in coords]
            image = cv2.circle(image, coords, 2, (0, 255, 0))
        for x, y in pred:
            coords = (img_width, img_height) * (x, y) + (0.5, 0.5)
            coords = [int(c) for c in coords]
            image = cv2.circle(image, coords, 2, (0, 0, 255))

        cv2.imwrite(save_path, img)

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

    img_width, img_height = (224, 224)
    batch_size = 64

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

    model = tf.keras.models.load_model(
        'trained_model', custom_objects={"CustomModel": CustomModel, "CustomNME": CustomNME})

    evaluate(test_data_dir_300w, output_dir_300w, model)
    evaluate(test_data_dir_300w_selected, output_dir_300w_selected, model)
    evaluate(test_data_dir_wflw, output_dir_wflw, model)
        