import os
os.environ["PYTHONHASHSEED"] = str(1)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Embedding, Flatten, concatenate, \
    Conv1D, Reshape, LayerNormalization, Dropout
from tensorflow.keras import regularizers, metrics
import tensorflow as tf
from tensorflow_addons.losses import metric_learning
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike
import random as python_random
python_random.seed(1)
tf.random.set_seed(1)
import numpy as np
np.random.seed(1)
initializer = tf.keras.initializers.HeNormal(seed=1)
bias = tf.keras.initializers.Constant(0.1)

def mish(x):
	return x * tf.nn.tanh(tf.nn.softplus(x))

class TripletsNetwork(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, shape, batch_size=32, unit=32, margin=0, cnn_kernel_size=1, cnn_filter=1):
        super(TripletsNetwork, self).__init__()
        self.margin = margin
        self.shape = shape
        self.unit = unit
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.batch_size = batch_size
        self.triplets_network = self._triplet_model()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, input):
        return self.triplets_network(input)

    def _triplet_model(self):
        anchor_input_cat = Input(name="anchor_cat", shape=(self.shape[0],))
        anchor_input_num = Input(name="anchor_num", shape=(self.shape[1],))
        positive_input_cat = Input(name="positive_cat", shape=(self.shape[0],))
        positive_input_num = Input(name="positive_num", shape=(self.shape[1],))
        negative_input_cat = Input(name="negative_cat", shape=(self.shape[0],))
        negative_input_num = Input(name="negative_num", shape=(self.shape[1],))

        cat_input = Input(name="cat", shape=(self.shape[0],))
        emb = Embedding(1024, self.unit, input_length=self.shape[0], mask_zero=True)(cat_input)
        emb = Flatten()(emb)
        cat_dense_1 = Dense(16, activation=mish, kernel_regularizer=regularizers.l1_l2(1e-6),
                            kernel_initializer=initializer)(
            emb)
        cat_dense_1 = BatchNormalization()(cat_dense_1)
        cat_dense_2 = Dense(16, activation=mish, kernel_regularizer=regularizers.l1_l2(1e-6),
                            kernel_initializer=initializer)(
            cat_dense_1)
        cat_dense_2 = BatchNormalization()(cat_dense_2)
        cat_dense_2 = Reshape([-1, 1])(cat_dense_2)
        cat_dense_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                             kernel_regularizer=regularizers.l1_l2(1e-6), kernel_initializer=initializer,
                             activation=mish)(
            cat_dense_2)
        cat_conv_1 = BatchNormalization()(cat_dense_2)
        cat_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l1_l2(1e-6), kernel_initializer=initializer,
                            activation=mish)(
            cat_conv_1)
        cat_conv_3 = Flatten()(cat_conv_2)

        num_input = Input(name="num", shape=(self.shape[1],))
        num_input_bn = BatchNormalization()(num_input)
        num_dense_1 = Dense(16, activation=mish, kernel_regularizer=regularizers.l1_l2(1e-6),
                            kernel_initializer=initializer)(
            num_input_bn)
        num_dense_1 = BatchNormalization()(num_dense_1)
        num_dense_2 = Dense(16, activation=mish, kernel_regularizer=regularizers.l1_l2(1e-6),
                            kernel_initializer=initializer)(num_dense_1)
        num_dense_2 = BatchNormalization()(num_dense_2)
        num_dense_2 = Reshape([-1, 1])(num_dense_2)
        num_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l1_l2(1e-6), kernel_initializer=initializer,
                            activation=mish)(
            num_dense_2)
        num_conv_1 = BatchNormalization()(num_conv_1)
        num_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l1_l2(1e-6), kernel_initializer=initializer,
                            activation=mish)(
            num_conv_1)
        num_conv_3 = Flatten()(num_conv_2)

        output = concatenate([cat_conv_3, num_conv_3])

        embedding = Model([cat_input, num_input], output, name="Embedding")

        distances = self._distance(
            embedding([anchor_input_cat, anchor_input_num]),
            embedding([positive_input_cat, positive_input_num]),
            embedding([negative_input_cat, negative_input_num]),
        )

        triplets_network = Model(
            inputs=[anchor_input_cat, anchor_input_num,
                    positive_input_cat, positive_input_num,
                    negative_input_cat, negative_input_num], outputs=distances
        )

        return triplets_network

    def _distance(self, anchor, positive, negative):
        # ap_distance = tf.math.reduce_euclidean_norm([anchor, positive])
        # an_distance = tf.math.reduce_euclidean_norm([anchor, negative])
        # ap_distance = tf.exp(tf.math.reduce_euclidean_norm([anchor, positive]))
        # an_distance = tf.exp(tf.math.reduce_euclidean_norm([anchor, negative]))
        # rp_softmax = ap_distance / (ap_distance + an_distance)
        # rn_softmax = an_distance / (ap_distance + an_distance)

        an_distance = tf.reduce_sum(tf.square(anchor - negative), 1)
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), 1)

        return an_distance, ap_distance

    def train_step(self, data):
        # GradientTape는 내부에서 수행하는 모든 작업을 기록하는 컨텍스트 관리자입니다.
        # 여기서 손실을 계산하는 데 사용하므로 그래디언트를 가져올 수 있고,
        # `compile()`을 통해 그래디언트를 적용할 수 있습니다.

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # 가중치에 대한 손실 함수의 그래디언트를 저장합니다.
        gradients = tape.gradient(loss, self.triplets_network.trainable_weights)

        # 모델에 지정된 옵티마이저를 통해 그래디언트를 적용합니다.
        self.optimizer.apply_gradients(
            zip(gradients, self.triplets_network.trainable_weights)
        )

        # trainig loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # loss 계산하기
        # 우선 계산된 anchor-positive 거리와 anchor-negative 거리를 가져옵니다.
        an_distance, ap_distance = self.triplets_network(data)

        # triplet loss 정의에 따라
        # 두 거리를 빼고, 음수가 나오지 않도록 max(loss+margin, 0)을 적용합니다.
        # loss = tf.reduce_mean(tf.pow(ap_distance - (an_distance - 1), 2))
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        loss = tf.reduce_mean(loss)

        return loss

    @property
    def metrics(self):
        # `reset_states()`가 자동으로 호출될 수 있도록 여기에 메트릭을 나열해야 합니다.
        return [self.loss_tracker]

class TupletLoss(Model):
    def __init__(self, shape, unit=32, cnn_kernel_size=1, cnn_filter=1, num_neg_sample=1, batch_size=1):
        super(TupletLoss, self).__init__()
        self.shape = shape
        self.unit = unit
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.num_neg_sample = num_neg_sample
        self.batch_size = batch_size
        self.tuplet_loss = self._tuplet_loss()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, input):
        return self.tuplet_loss(input)

    def _tuplet_loss(self):
        anchor_input = {"cat" : Input(shape=(self.shape[0],)),
                          "num" : Input(shape=(self.shape[1],))}
        positive_input = {"cat" : Input(shape=(self.shape[0],)),
                          "num" : Input(shape=(self.shape[1],))}
        negative_input = {"cat_0": Input(shape=(self.shape[0],)),
                          "num_0": Input(shape=(self.shape[1],))}

        for i in range(self.num_neg_sample - 1):
            negative_input["cat_" + str(i+1)] = Input(shape=(self.shape[0],))
            negative_input["num_" + str(i+1)] = Input(shape=(self.shape[1],))

        embedding_neg_list = []
        neg_list = []

        cat_input = Input(shape=(self.shape[0],))
        emb = Embedding(1024, self.unit, input_length=self.shape[0] + 1)(cat_input) # ibk : 1024, taiwan : 128
        cat_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(emb)
        cat_conv_1 = BatchNormalization()(cat_conv_1)
        cat_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(cat_conv_1)
        cat_conv_2 = BatchNormalization()(cat_conv_2)
        cat_conv_3 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(cat_conv_2)
        cat_conv_3 = BatchNormalization()(cat_conv_3)
        cat_conv_4 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(cat_conv_3)
        cat_conv_4 = BatchNormalization()(cat_conv_4)
        cat_conv_8 = Flatten()(cat_conv_4)

        num_input = Input(shape=(self.shape[1],))
        num_res_input = Reshape([-1, 1])(num_input)
        num_res_input = BatchNormalization()(num_res_input)
        num_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(num_res_input)
        num_conv_1 = BatchNormalization()(num_conv_1)
        num_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(num_conv_1)
        num_conv_2 = BatchNormalization()(num_conv_2)
        num_conv_3 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(num_conv_2)
        num_conv_3 = BatchNormalization()(num_conv_3)
        num_conv_4 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal',
                            activation=mish)(num_conv_3)
        num_conv_4 = BatchNormalization()(num_conv_4)
        num_conv_8 = Flatten()(num_conv_4)

        output = concatenate([cat_conv_8, num_conv_8])
        output = Dense(8, activation=mish, kernel_regularizer=regularizers.l2(5e-3), kernel_initializer='HeNormal')(
            output)
        output = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-3),
                       kernel_initializer='HeNormal')(output)

        embedding = Model([cat_input, num_input], output, name="Embedding")

        for i in range(self.num_neg_sample):
            neg_list.append(negative_input["cat_" + str(i)])

        for i in range(self.num_neg_sample):
            neg_list.append(negative_input["num_" + str(i)])
            embedding_neg_list.append(embedding([negative_input["cat_" + str(i)], negative_input["num_" + str(i)]]))

        distances = self._distance(
            embedding([anchor_input["cat"], anchor_input["num"]]),
            embedding([positive_input["cat"], positive_input["num"]]),
            embedding_neg_list
        )

        tuplet_network = Model(
            inputs=[anchor_input["cat"], anchor_input["num"],
                    positive_input["cat"], positive_input["num"],
                    neg_list], outputs=distances
        )

        return tuplet_network

    def _distance(self, *args):
        anchor, pos, neg = args[0], args[1], args[2]
        pos_dot_product = tf.matmul(pos, anchor, transpose_b=True)
        ap_cos = tf.losses.cosine_similarity(anchor, pos, axis = 1)
        an_cos = 0.0
        distance = 0

        for i in neg:
            neg_dot_product = tf.matmul(i, anchor, transpose_b=True)
            tmp_an_cos = tf.losses.cosine_similarity(anchor, i, axis = 1)
            distance += tf.reduce_sum(tf.exp(neg_dot_product - pos_dot_product), 1)
            an_cos += tmp_an_cos

        an_cos = an_cos / len(neg)

        return distance, ap_cos, an_cos

    def train_step(self, data):
        # GradientTape는 내부에서 수행하는 모든 작업을 기록하는 컨텍스트 관리자입니다.
        # 여기서 손실을 계산하는 데 사용하므로 그래디언트를 가져올 수 있고,
        # `compile()`을 통해 그래디언트를 적용할 수 있습니다.

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # 가중치에 대한 손실 함수의 그래디언트를 저장합니다.
        gradients = tape.gradient(loss, self.tuplet_loss.trainable_weights)

        # 모델에 지정된 옵티마이저를 통해 그래디언트를 적용합니다.
        self.optimizer.apply_gradients(
            zip(gradients, self.tuplet_loss.trainable_weights)
        )

        # trainig loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # loss 계산하기
        # 우선 계산된 anchor-positive 거리와 anchor-negative 거리를 가져옵니다.
        distance, _, _ = self.tuplet_loss(data)

        # tuplet loss
        loss = tf.math.log1p(distance)
        loss = tf.reduce_mean(loss)
        return loss

    @property
    def metrics(self):
        # `reset_states()`가 자동으로 호출될 수 있도록 여기에 메트릭을 나열해야 합니다.
        return [self.loss_tracker]

class LiftedStructLoss(Model):
    def __init__(self, shape, unit=32, margin=0, cnn_kernel_size=1, cnn_filter=1, batch_size = 32, dataset=None):
        super(LiftedStructLoss, self).__init__()
        self.margin = margin
        self.shape = shape
        self.unit = unit
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.batch_size = batch_size
        self.dataset = dataset
        os.environ['PYTHONHASHSEED'] = str(1)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(1)
        np.random.seed(1)
        python_random.seed(1)
        self.lfe_network = self._lifted_feature_embedding_model()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, input):
        return self.lfe_network(input)

    def _lifted_feature_embedding_model_before(self):
        os.environ['PYTHONHASHSEED'] = str(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        python_random.seed(1)

        cat_input = Input(name="cat", shape=(self.shape[0],))
        emb = Embedding(1024, self.unit, input_length=self.shape[0])(cat_input)
        cat_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            emb)
        cat_conv_1 = BatchNormalization()(cat_conv_1)
        cat_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            cat_conv_1)
        cat_conv_2 = BatchNormalization()(cat_conv_2)
        cat_conv_3 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            cat_conv_2)
        cat_conv_3 = BatchNormalization()(cat_conv_3)
        cat_conv_3 = Flatten()(cat_conv_3)

        # num_input = Input(name="num", shape=(self.shape[1],))
        # input = concatenate([num_input, cat_input])
        # input = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal')(
        #     input)
        # input = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal')(
        #     input)
        # input = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal')(
        #     input)
        # input = Reshape([-1, 1])(input)
        # cat_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
        #                     kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal', activation=mish)(
        #     input)
        # cat_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
        #                     kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal', activation=mish)(
        #     cat_conv_1)
        # cat_conv_3 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
        #                     kernel_regularizer=regularizers.l2(5e-4), kernel_initializer='HeNormal', activation=mish)(
        #     cat_conv_2)
        # output = Flatten()(cat_conv_3)

        num_input = Input(name="num", shape=(self.shape[1],))
        num_res_input = Reshape([-1, 1])(num_input)
        num_res_input = BatchNormalization()(num_res_input)
        num_conv_1 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            num_res_input)
        num_conv_1 = BatchNormalization()(num_conv_1)
        num_conv_2 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            num_conv_1)
        num_conv_2 = BatchNormalization()(num_conv_2)
        num_conv_3 = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer, activation=mish)(
            num_conv_2)
        num_conv_3 = BatchNormalization()(num_conv_3)
        num_conv_3 = Flatten()(num_conv_3)

        output = concatenate([cat_conv_3, num_conv_3])
        output = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer)(
            output)
        output = Dense(16, activation=mish, kernel_regularizer=regularizers.l2(5e-4), kernel_initializer=initializer)(
            output)

        lifted_feature_embedding = Model(
            inputs=[cat_input, num_input], outputs=output
        )

        return lifted_feature_embedding

    def _lifted_feature_embedding_model(self):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        python_random.seed(1)

        cat_input = Input(name="cat", shape=(self.shape[0],))
        emb = Embedding(1024, self.unit, input_length=self.shape[0], mask_zero=True)(cat_input)
        emb = BatchNormalization()(emb)
        emb = Flatten()(emb)
        cat_dense_1 = Dense(96, activation=mish, kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer)(emb) # taiwan : 32, lending : 64
        cat_dense_1 = BatchNormalization()(cat_dense_1)
        cat_dense_2 = Dense(32, activation=mish, kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer)(
            cat_dense_1)
        cat_dense_2 = BatchNormalization()(cat_dense_2)
        cat_dense_2 = Reshape([-1, 1])(cat_dense_2)

        num_input = Input(name="num", shape=(self.shape[1],))
        num_dense_1 = Dense(96, activation=mish, kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer)( # taiwan : 32, lending : 64
            num_input)
        num_dense_1 = BatchNormalization()(num_dense_1)
        num_dense_2 = Dense(32, activation=mish, kernel_regularizer=regularizers.l2(1e-6),
                          kernel_initializer=initializer)(num_dense_1)
        num_dense_2 = BatchNormalization()(num_dense_2)
        num_dense_2 = Reshape([-1, 1])(num_dense_2)

        output = concatenate([cat_dense_2, num_dense_2])
        output = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer, activation=mish)(
            output)
        output = BatchNormalization()(output)
        output = Conv1D(1, self.cnn_kernel_size, padding='valid',
                            kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer, activation=mish)(
            output)
        output = Flatten()(output)

        lifted_feature_embedding = Model(
            inputs=[cat_input, num_input], outputs=output
        )

        return lifted_feature_embedding

    def train_step(self, data):
        # GradientTape는 내부에서 수행하는 모든 작업을 기록하는 컨텍스트 관리자입니다.
        # 여기서 손실을 계산하는 데 사용하므로 그래디언트를 가져올 수 있고,
        # `compile()`을 통해 그래디언트를 적용할 수 있습니다.

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # 가중치에 대한 손실 함수의 그래디언트를 저장합니다.
        gradients = tape.gradient(loss, self.lfe_network.trainable_weights)

        # 모델에 지정된 옵티마이저를 통해 그래디언트를 적용합니다.
        self.optimizer.apply_gradients(
            zip(gradients, self.lfe_network.trainable_weights)
        )

        # trainig loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        embedding = self.lfe_network(data[0])
        loss = self.lifted_struct_loss(data[1], embedding)
        return loss

    # @tf.function
    def lifted_struct_loss(self,
            labels: TensorLike, embeddings: TensorLike, margin: FloatTensorLike = 1.0
    ) -> tf.Tensor:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['PYTHONHASHSEED'] = str(1)
        tf.random.set_seed(1)
        np.random.seed(1)
        python_random.seed(1)
        """Computes the lifted structured loss.
        Args:
          labels: 1-D tf.int32 `Tensor` with shape `[batch_size]` of
            multiclass integer labels.
          embeddings: 2-D float `Tensor` of embedding vectors. Embeddings should
            not be l2 normalized.
          margin: Float, margin term in the loss definition.
        Returns:
          lifted_loss: float scalar with dtype of embeddings.
        """
        convert_to_float32 = (
                embeddings.dtype == tf.dtypes.float16 or embeddings.dtype == tf.dtypes.bfloat16
        )
        precise_embeddings = (
            tf.cast(embeddings, tf.dtypes.float32) if convert_to_float32 else embeddings
        )

        # Reshape [batch_size] label tensor to a [batch_size, 1] label tensor.
        lshape = tf.shape(labels)
        labels = tf.reshape(labels, [lshape[0], 1])

        # Build pairwise squared distance matrix.
        pairwise_distances = metric_learning.pairwise_distance(precise_embeddings)

        # Build pairwise binary adjacency matrix.
        adjacency = tf.math.equal(labels, tf.transpose(labels))
        # Invert so we can select negatives only.
        adjacency_not = tf.math.logical_not(adjacency)

        batch_size = tf.size(labels)

        diff = self.margin - pairwise_distances
        mask = tf.cast(adjacency_not, dtype=tf.dtypes.float32)
        # Safe maximum: Temporarily shift negative distances
        #   above zero before taking max.
        #     this is to take the max only among negatives.
        row_minimums = tf.math.reduce_min(diff, 1, keepdims=True)
        row_negative_maximums = (
                tf.math.reduce_max(
                    tf.math.multiply(diff - row_minimums, mask), 1, keepdims=True
                )
                + row_minimums
        )

        # Compute the loss.
        # Keep track of matrix of maximums where M_ij = max(m_i, m_j)
        #   where m_i is the max of alpha - negative D_i's.
        # This matches the Caffe loss layer implementation at:
        #   https://github.com/rksltnl/Caffe-Deep-Metric-Learning-CVPR16/blob/0efd7544a9846f58df923c8b992198ba5c355454/src/caffe/layers/lifted_struct_similarity_softmax_layer.cpp

        max_elements = tf.math.maximum(
            row_negative_maximums, tf.transpose(row_negative_maximums)
        )
        diff_tiled = tf.tile(diff, [batch_size, 1])
        mask_tiled = tf.tile(mask, [batch_size, 1])
        max_elements_vect = tf.reshape(tf.transpose(max_elements), [-1, 1])

        loss_exp_left = tf.reshape(
            tf.math.reduce_sum(
                # tf.math.multiply(tf.math.exp(diff_tiled - max_elements_vect), mask_tiled),
                # tf.math.multiply(tf.math.softplus(diff_tiled - max_elements_vect), mask_tiled),
                tf.math.multiply(diff_tiled - max_elements_vect, mask_tiled),
                1,
                keepdims=True,
            ),
            [batch_size, batch_size],
        )

        # loss_mat = max_elements + tf.math.log(loss_exp_left + tf.transpose(loss_exp_left))
        loss_mat = max_elements + tf.math.softplus(loss_exp_left + tf.transpose(loss_exp_left))
        # Add the positive distance.
        loss_mat += pairwise_distances

        mask_positives = tf.cast(adjacency, dtype=tf.dtypes.float32) - tf.linalg.diag(
            tf.ones([batch_size])
        )

        # *0.5 for upper triangular, and another *0.5 for 1/2 factor for loss^2.
        num_positives = tf.math.reduce_sum(mask_positives) / 2.0

        lifted_loss = tf.math.truediv(
            0.25
            * tf.math.reduce_sum(
                tf.math.square(
                    tf.math.maximum(tf.math.multiply(loss_mat, mask_positives), 0.0)
                )
            ),
            num_positives,
        )

        if convert_to_float32:
            return tf.cast(lifted_loss, embeddings.dtype)
        else:
            return lifted_loss

class SiameseNetwork(Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, shape, unit=32, margin=0, cnn_kernel_size=1, cnn_filter=1, batch_size=32, sharpness_param=0):
        super(SiameseNetwork, self).__init__()
        self.margin = margin
        self.shape = shape
        self.unit = unit
        self.cnn_kernel_size = cnn_kernel_size
        self.cnn_filter = cnn_filter
        self.batch_size = batch_size
        self.sharpness_param = sharpness_param
        self._siamese_network = self._siamese_model()
        self.loss_tracker = metrics.Mean(name="loss")

    def call(self, input):
        return self._siamese_network(input)

    def _siamese_model(self):
        anchor_input_cat = Input(name="anchor_cat", shape=(self.shape[0],))
        anchor_input_num = Input(name="anchor_num", shape=(self.shape[1],))
        disc_input_cat = Input(name="positive_cat", shape=(self.shape[0],))
        disc_input_num = Input(name="positive_num", shape=(self.shape[1],))

        cat_input = Input(name="cat", shape=(self.shape[0],))
        emb = Embedding(1024, self.unit, input_length=self.shape[0], mask_zero=True)(cat_input)
        emb = BatchNormalization()(emb)
        emb = Flatten()(emb)
        cat_dense_1 = Dense(96, activation=mish, kernel_regularizer=regularizers.l2(1e-6),
                            kernel_initializer=initializer)(emb)  # taiwan : 32, lending : 64
        cat_dense_1 = BatchNormalization()(cat_dense_1)
        cat_dense_2 = Dense(32, activation=mish, kernel_regularizer=regularizers.l2(1e-6),
                            kernel_initializer=initializer)(
            cat_dense_1)
        cat_dense_2 = BatchNormalization()(cat_dense_2)
        cat_dense_2 = Reshape([-1, 1])(cat_dense_2)

        num_input = Input(name="num", shape=(self.shape[1],))
        num_dense_1 = Dense(96, activation=mish, kernel_regularizer=regularizers.l2(1e-6),
                            kernel_initializer=initializer)(  # taiwan : 32, lending : 64
            num_input)
        num_dense_1 = BatchNormalization()(num_dense_1)
        num_dense_2 = Dense(32, activation=mish, kernel_regularizer=regularizers.l2(1e-6),
                            kernel_initializer=initializer)(num_dense_1)
        num_dense_2 = BatchNormalization()(num_dense_2)
        num_dense_2 = Reshape([-1, 1])(num_dense_2)

        output = concatenate([cat_dense_2, num_dense_2])
        output = Conv1D(self.cnn_filter, self.cnn_kernel_size, padding='valid',
                        kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer, activation=mish)(
            output)
        output = BatchNormalization()(output)
        output = Conv1D(1, self.cnn_kernel_size, padding='valid',
                        kernel_regularizer=regularizers.l2(1e-6), kernel_initializer=initializer, activation=mish)(
            output)
        output = Flatten()(output)

        embedding = Model([cat_input, num_input], output, name="Embedding")

        distance = self._distance(
            embedding([anchor_input_cat, anchor_input_num]),
            embedding([disc_input_cat, disc_input_num]),
        )

        triplets_network = Model(
            inputs=[anchor_input_cat, anchor_input_num,
                    disc_input_cat, disc_input_num], outputs=distance
        )

        return triplets_network

    def _distance(self, anchor, disc):
        distance = tf.reduce_sum(tf.square(anchor - disc), 1)

        return distance

    def train_step(self, data):
        # GradientTape는 내부에서 수행하는 모든 작업을 기록하는 컨텍스트 관리자입니다.
        # 여기서 손실을 계산하는 데 사용하므로 그래디언트를 가져올 수 있고,
        # `compile()`을 통해 그래디언트를 적용할 수 있습니다.

        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # 가중치에 대한 손실 함수의 그래디언트를 저장합니다.
        gradients = tape.gradient(loss, self._siamese_network.trainable_weights)

        # 모델에 지정된 옵티마이저를 통해 그래디언트를 적용합니다.
        self.optimizer.apply_gradients(
            zip(gradients, self._siamese_network.trainable_weights)
        )

        # trainig loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # loss를 갱신해줍니다.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def _compute_loss(self, data):
        # loss 계산하기
        distance = self._siamese_network(data[0])
        sim_loss = tf.maximum(0.0, (tf.cast(data[1][0], tf.float32) * distance) - (self.margin - 0.7))
        dis_loss = tf.maximum(0.0, self.margin - (tf.cast(data[1][1], tf.float32) * distance))
        loss = tf.reduce_sum(sim_loss + dis_loss)
        # const = tf.cast(data[1], tf.float32) * (self.margin - distance)
        # logistic_input = self.sharpness_param * (1 - const)
        # generalized_logistic = (1 / self.sharpness_param) * tf.math.log1p(tf.exp(logistic_input))
        # loss = 1/2 * tf.reduce_mean(generalized_logistic)

        return loss

    @property
    def metrics(self):
        # `reset_states()`가 자동으로 호출될 수 있도록 여기에 메트릭을 나열해야 합니다.
        return [self.loss_tracker]



