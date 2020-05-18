# Efficient
!pip install -q efficientnet
import efficientnet.tfkeras as efn

# Learning Rate Schedule
LR_START = 0.0001
LR_MAX = 0.0004
LR_MIN = 0.000001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 6
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = np.random.random_sample() * LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
    lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN

    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

# EfficientNet07
def get_model():
    with strategy.scope():
        rnet = efn.EfficientNetB7(
            input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
            weights='noisy-student',
            include_top=False
        )
        # trainable rnet
        rnet.trainable = True
        model = tf.keras.Sequential([
            rnet,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(len(CLASSES), activation='softmax')
        ])
    model.compile(
        optimizer='adam',
        loss = 'sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy']
    )
    return model

# EfficientNet 교차검증
from sklearn.model_selection import KFold

def train_cross_validate(folds = 5):
    histories = []
    models = []
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', patience = 3)
    kfold = KFold(folds, shuffle = True, random_state = SEED)
    for f, (trn_ind, val_ind) in enumerate(kfold.split(TRAINING_FILENAMES)):
        print(); print('#'*25)
        print('### FOLD',f+1)
        print('#'*25)
        train_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[trn_ind]['TRAINING_FILENAMES']), labeled = True)
        val_dataset = load_dataset(list(pd.DataFrame({'TRAINING_FILENAMES': TRAINING_FILENAMES}).loc[val_ind]['TRAINING_FILENAMES']), labeled = True, ordered = True)
        model = get_model()
        history = model.fit(
            get_training_dataset(train_dataset), 
            steps_per_epoch = STEPS_PER_EPOCH,
            epochs = EPOCHS,
            callbacks = [lr_callback],
            validation_data = get_validation_dataset(val_dataset),
            verbose=1
        )
        models.append(model)
        histories.append(history)
    return histories, models

def train_and_predict(folds = 5):
    test_ds = get_test_dataset(ordered=True)
    test_images_ds = test_ds.map(lambda image, idnum: image)
    histories, models = train_cross_validate(folds = folds)
    probabilities = np.average([models[i].predict(test_images_ds) for i in range(folds)], axis = 0)
    
histories, models = train_and_predict(folds = FOLDS)
