import os
from ast import literal_eval
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from model import create_landmarks_detector, CustomNME

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
    train_data_dir = root_dir + '/preprocessed_data/train'
    annotations_df = pd.read_csv(train_data_dir + '/annotations.csv')
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

    train_annotations = annotations_df.sample(frac = 0.75)
    validate_annotations = annotations_df.drop(train_annotations.index)
    num_validation_samples = len(validate_annotations.index)

    train_datagen = ImageDataGenerator(brightness_range=(-0.2,0.2))
    train_generator = train_datagen.flow_from_dataframe(
        train_annotations,
        directory=train_data_dir,
        x_col = 'path', y_col = ['landmarks','angles'],
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'multi_output')
    validation_generator = train_datagen.flow_from_dataframe(
        validate_annotations,
        directory=train_data_dir,
        x_col = 'path', y_col = ['landmarks','angles'],
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'multi_output')

    model = create_landmarks_detector(input_shape = (img_width, img_height, 3))
    adam_w = tfa.optimizers.AdamW(learning_rate=1e-4, weight_decay=1e-6)
    model.compile(run_eagerly=True, optimizer=optimizers.Adam(learning_rate=1e-4),
                  metrics=[CustomNME(name='NME')])
    print(model.summary())

    model_checkpoint = ModelCheckpoint(
        filepath=root_dir + '/checkpoints/ckpt-{epoch:03d}',
        save_weights_only=True,
        monitor='val_loss',
        mode='max')
    callbacks_list = [model_checkpoint]

    model_history = model.fit(
        train_generator,
        epochs=300,
        validation_data=validation_generator,
        validation_steps=num_validation_samples // batch_size,
        callbacks=callbacks_list)
    model.save('trained_model')
