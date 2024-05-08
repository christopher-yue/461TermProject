import keras
import numpy as np
import pandas as pd
import os
from glob import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, average_precision_score, accuracy_score, roc_auc_score
import tensorflow as tf
from sklearn.svm import SVC

# Image preprocessing


# Below is some helper code to read all of your full image filepaths into a dataframe for easier manipulation
# Load the NIH data to all_xray_df
all_xray_df = pd.read_csv('archive/Data_Entry_2017.csv')
all_image_paths = {os.path.basename(x): x for x in glob(os.path.join('archive', 'images*', '*', '*.png'))}
print('Scans found:', len(all_image_paths), ', Total Headers', all_xray_df.shape[0])
all_xray_df['path'] = all_xray_df['Image Index'].map(all_image_paths.get)
all_xray_df.sample(3)

# Using only PA images
pa_xray_df = all_xray_df.drop(all_xray_df.loc[all_xray_df['View Position'] == 'AP'].index)

# Splitting finding lables into individual rows
cleaned = pa_xray_df.rename(columns={'Finding Labels': 'labels'})
cleaned = cleaned.set_index('Image Index').labels.str.split('|', expand=True).stack().reset_index(level=1, drop=True).to_frame(
    'lables')
cleaned.head()

# getting dummy variables for the lables and grouping by the index.
cleaned = pd.get_dummies(cleaned, columns=['lables']).groupby(level=0).sum()

cleaned.head()

# ensuring both data frames use the same index
pa_xray_df.set_index('Image Index', inplace=True)

# merging dummy variable columns with the data frame containing the image paths.
prepared_df = pa_xray_df.merge(cleaned, left_index=True, right_index=True)

prepared_df.head()

# Renamiong dummy column to 'pneumonia_class' that will allow us to look at
# images with or without pneumonia for binary classification

prepared_df.rename(columns={'lables_Pneumonia': 'pneumonia_class'}, inplace=True)

# Checking that class is binary
prepared_df.pneumonia_class.unique()

prepared_df.to_csv("prepared_df.csv")

prepared_df = pd.read_csv("prepared_df.csv", index_col="Image Index")

# checking class imbalance
prepared_df['pneumonia_class'].value_counts()

train_data, val_data = train_test_split(prepared_df, test_size=0.2, stratify=prepared_df['pneumonia_class'], random_state=42)

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    featurewise_std_normalization=False,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255,
    featurewise_center=False,
    featurewise_std_normalization=False)

train_generator = train_datagen.flow_from_dataframe(
    train_data, directory=None, x_col='path', y_col='pneumonia_class', weight_col=None,
    target_size=(224, 224), color_mode='rgb', classes=None,
    class_mode='raw', batch_size=32, shuffle=True, seed=42,
    save_to_dir=None, save_prefix='', save_format='png', subset=None,
    interpolation='nearest', validate_filenames=True
)
validation_generator = val_datagen.flow_from_dataframe(
    val_data, directory=None, x_col='path', y_col='pneumonia_class', weight_col=None,
    target_size=(224, 224), color_mode='rgb', classes=None,
    class_mode='raw', batch_size=32, shuffle=True, seed=42,
    save_to_dir=None, save_prefix='', save_format='png', subset=None,
    interpolation='nearest', validate_filenames=True
)

t_x, t_y = next(train_generator)
fig, m_axs = plt.subplots(4, 4, figsize=(16, 16))
for (c_x, c_y, c_ax) in zip(t_x, t_y, m_axs.flatten()):
    c_ax.imshow(c_x[:, :, 0], cmap='bone')
    if c_y == 1:
        c_ax.set_title('Pneumonia')
    else:
        c_ax.set_title('No Pneumonia')
    c_ax.axis('off')

    METRICS = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
    ]


# Feature extraction


def get_model(metrics=None, output_bias=None):
    if metrics is None:
        metrics = METRICS
    if output_bias is not None:
        output_bias = keras.initializers.Constant(output_bias)

    densenet = keras.applications.DenseNet169(weights='imagenet', include_top=False, pooling='avg', input_shape=[224, 224, 3])
    densenet.trainable = True  # Using pretrained weights due to compute limitation on the worspace.
    model = keras.Sequential([
        densenet,
        keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=METRICS
    )
    return model


# defining learning rate sheduler (currently not used)
LR_START = 0.0001
LR_MAX = 0.0001
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 3
LR_SUSTAIN_EPOCHS = 0
LR_EXP_DECAY = .8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = np.random.random_sample() * LR_START  # Using random learning rate for initial epochs.
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
                    epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN  # Rapid decay of learning rate to improve convergence.
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

# defining early stopping
es_callback = tf.keras.callbacks.EarlyStopping(
    monitor='val_auc',
    verbose=1,
    patience=5,
    mode='max',
    restore_best_weights=True)

checkpoint_path = "training/cp.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)

# check gpu status
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# calculating class weights to adress class imbalance
positive_findings = 630
negative_findings = 66680
total = positive_findings+negative_findings

initial_bias = np.log([positive_findings/negative_findings])

weight_for_0 = (1 / negative_findings)*(total)/2.0
weight_for_1 = (1 / positive_findings)*(total)/2.0

class_weight = {0: weight_for_0, 1: weight_for_1}

print('Weight for class 0: {:.2f}'.format(weight_for_0))
print('Weight for class 1: {:.2f}'.format(weight_for_1))

model = get_model(output_bias = initial_bias)

model.summary()


# Classification with SVM


# Extract features from pretrained DenseNet
train_features = model.predict(train_generator)
val_features = model.predict(validation_generator)

# Extract labels
train_labels = train_generator.labels
val_labels = validation_generator.labels

# Define SVM model with RBF kernel
svm_model = SVC(kernel='rbf', gamma='scale', probability=True, random_state=42)

# Train SVM model
svm_model.fit(train_features, train_labels)

# Predict probabilities on validation data
val_probabilities = svm_model.predict_proba(val_features)[:, 1]

# Calculate AUC score
svm_auc = roc_auc_score(val_labels, val_probabilities)
print("SVM AUC:", svm_auc)
