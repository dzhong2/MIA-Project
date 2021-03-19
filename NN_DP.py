import tensorflow.compat.v2 as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
import keras


class target_model():
    def __init__(self,
                 num_epoch=60,
                 dp_flag=0,
                 l2_norm_clip=1.0,
                 noise_multiplier=1.3,
                 num_microbatches=25,
                 learning_rate=0.01,
                 data_size=10000,
                 verbos=1,
                 reduce=1):
        self.nm = noise_multiplier
        self.l2 = l2_norm_clip
        self.dp = dp_flag
        self.num_epoch = num_epoch
        self.batch_size = num_microbatches
        self.data_size = data_size
        self.verbos = verbos
        self.t_model = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2, activation='softmax')])
        if reduce:
            self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        else:
            self.lr_schedule = learning_rate
        if self.dp:
            self.opt = DPKerasSGDOptimizer(
                l2_norm_clip=self.l2,
                noise_multiplier=self.nm,
                num_microbatches=self.batch_size,
                learning_rate=self.lr_schedule)
        else:
            self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        self.t_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X_train, y_train, epoch=-1):
        if epoch == -1:
            epoch = self.num_epoch
        y_train = y_train.reshape(-1, 1)
        self.t_model.fit(X_train, y_train, epochs=epoch, verbose=self.verbos, batch_size=self.batch_size)
    def fit_ds(self, dataset, epoch=-1):
        if epoch == -1:
            epoch = self.num_epoch
        self.t_model.fit(dataset, epochs=epoch, verbose=self.verbos, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.t_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.t_model.predict(X)
        y_pred = probs.argmax(axis=1)
        return y_pred

    def clf_prob(self, X):
        plables = self.predict(X)
        prob = self.predict_proba(X)
        return prob, plables

class attack_model():
    def __init__(self,
                 num_epoch=60,
                 learning_rate=0.01,
                 batch_size=25,
                 verbose=1):
        self.a_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(128, activation=tf.keras.layers.LeakyReLU(alpha=0.01)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.num_epoch = num_epoch
        self.verbos = verbose
        '''self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)'''
        self.lr_schedule = learning_rate
        self.batch_size = batch_size
        self.opt = tf.keras.optimizers.SGD(learning_rate=self.lr_schedule)
        self.loss = keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.a_model.compile(optimizer=self.opt,
                             loss=self.loss,
                             metrics=['accuracy'])

    def fit(self, X_train, y_train):
        y_train = y_train.reshape(-1, 1)
        self.a_model.fit(X_train, y_train, epochs=self.num_epoch, verbose=self.verbos, batch_size=self.batch_size)

    def predict_proba(self, X):
        probs = self.a_model.predict(X)
        return probs

    def predict(self, X):
        probs = self.a_model.predict(X)
        #y_pred = probs.argmax(axis=1)
        y_pred = probs.round()
        return y_pred

