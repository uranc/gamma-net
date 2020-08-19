import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import os
import pdb
from tensorflow.keras.applications.vgg16 import preprocess_input


class TensorBoardCustom(tf.keras.callbacks.TensorBoard):
    def __init__(self, **kwargs):
        super(TensorBoardCustom, self).__init__(**kwargs)
        self.mean = K.constant([103.939, 116.779, 123.68],
                               dtype=tf.float32,
                               shape=[1, 1, 1, 3],
                               name='img_mean')  # BGR
        self.var_y_true = tf.Variable(0., validate_shape=False)
        self.var_y_mask = tf.Variable(0., validate_shape=False)
        self.var_y_pred = tf.Variable(0., validate_shape=False)
        # self.var_vgg = tf.Variable(0., validate_shape=False)

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        if self.histogram_freq and self.merged is None:
            for layer in self.model.layers:
                for weight in layer.weights:
                    mapped_weight_name = weight.name.replace(':', '_')
                    tf.summary.histogram(mapped_weight_name, weight)
                    if self.write_grads and weight in layer.trainable_weights:
                        grads = model.optimizer.get_gradients(model.total_loss,
                                                              weight)

                        def is_indexed_slices(grad):
                            return type(grad).__name__ == 'IndexedSlices'
                        grads = [
                            grad.values if is_indexed_slices(grad) else grad
                            for grad in grads]
                        tf.summary.histogram('{}_grad'.format(mapped_weight_name),
                                             grads)
                    if self.write_images:
                        w_img = tf.squeeze(weight)
                        shape = K.int_shape(w_img)
                        if len(shape) == 2:  # dense layer kernel case
                            if shape[0] > shape[1]:
                                w_img = tf.transpose(w_img)
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       shape[1],
                                                       1])
                        elif len(shape) == 3:  # convnet case
                            if K.image_data_format() == 'channels_last':
                                # switch to channels_first to display
                                # every kernel as a separate image
                                w_img = tf.transpose(w_img, perm=[2, 0, 1])
                                shape = K.int_shape(w_img)
                            w_img = tf.reshape(w_img, [shape[0],
                                                       shape[1],
                                                       shape[2],
                                                       1])
                        elif len(shape) == 1:  # bias case
                            w_img = tf.reshape(w_img, [1,
                                                       shape[0],
                                                       1,
                                                       1])
# elif len(shape) == 4:
#     # w_img = tf.transpose(w_img, perm=[3, 0, 1, 2])
#     if not (shape[3] == 3):
#         w_img = tf.reshape(w_img, [1,
#                                    shape[0] *
#                                    int(shape[2]
#                                        * shape[3]/64),
#                                    shape[1]*64,
#                                    1])

                        else:
                            # not possible to handle 3D convnets etc.
                            continue
                        shape = K.int_shape(w_img)
                        assert len(shape) == 4 and shape[-1] in [1, 3, 4]
                        tf.summary.image(mapped_weight_name, w_img)

                if hasattr(layer, 'output'):
                    if isinstance(layer.output, list):
                        for i, output in enumerate(layer.output):
                            tf.summary.histogram('{}_out_{}'.format(layer.name, i),
                                                 output)
                    else:
                        tf.summary.histogram('{}_out'.format(layer.name),
                                             layer.output)
        self.merged = tf.summary.merge_all()

        if self.write_graph:
            self.writer = tf.summary.FileWriter(self.log_dir,
                                                self.sess.graph)
        else:
            self.writer = tf.summary.FileWriter(self.log_dir)

        if self.embeddings_freq and self.embeddings_data is not None:
            self.embeddings_data = standardize_input_data(self.embeddings_data,
                                                          model.input_names)

            embeddings_layer_names = self.embeddings_layer_names

            if not embeddings_layer_names:
                embeddings_layer_names = [layer.name for layer in self.model.layers
                                          if type(layer).__name__ == 'Embedding']
            self.assign_embeddings = []
            embeddings_vars = {}

            self.batch_id = batch_id = tf.placeholder(tf.int32)
            self.step = step = tf.placeholder(tf.int32)

            for layer in self.model.layers:
                if layer.name in embeddings_layer_names:
                    embedding_input = self.model.get_layer(layer.name).output
                    embedding_size = np.prod(embedding_input.shape[1:])
                    embedding_input = tf.reshape(embedding_input,
                                                 (step, int(embedding_size)))
                    shape = (self.embeddings_data[0].shape[0], int(
                        embedding_size))
                    embedding = tf.Variable(tf.zeros(shape),
                                            name=layer.name + '_embedding')
                    embeddings_vars[layer.name] = embedding
                    batch = tf.assign(embedding[batch_id:batch_id + step],
                                      embedding_input)
                    self.assign_embeddings.append(batch)

            self.saver = tf.train.Saver(list(embeddings_vars.values()))

            if not isinstance(self.embeddings_metadata, str):
                embeddings_metadata = self.embeddings_metadata
            else:
                embeddings_metadata = {layer_name: self.embeddings_metadata
                                       for layer_name in embeddings_vars.keys()}

            config = projector.ProjectorConfig()

            for layer_name, tensor in embeddings_vars.items():
                embedding = config.embeddings.add()
                embedding.tensor_name = tensor.name

                if layer_name in embeddings_metadata:
                    embedding.metadata_path = embeddings_metadata[layer_name]

            projector.visualize_embeddings(self.writer, config)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        if not self.validation_data and self.histogram_freq:
            raise ValueError("If printing histograms, validation_data must be "
                             "provided, and cannot be a generator.")
        if self.embeddings_data is None and self.embeddings_freq:
            raise ValueError("To visualize embeddings, embeddings_data must "
                             "be provided.")
        if self.validation_data and self.histogram_freq:
            if epoch % self.histogram_freq == 0:

                val_data = self.validation_data
                tensors = (self.model.inputs +
                           self.model.targets +
                           self.model.sample_weights)

                if self.model.uses_learning_phase:
                    tensors += [K.learning_phase()]

                assert len(val_data) == len(tensors)
                val_size = val_data[0].shape[0]
                i = 0
                while i < val_size:
                    print(i)
                    step = min(self.batch_size, val_size - i)
                    if self.model.uses_learning_phase:
                        # do not slice the learning phase
                        batch_val = [x[i:i + step] for x in val_data[:-1]]
                        batch_val.append(val_data[-1])
                    else:
                        batch_val = [x[i:i + step] for x in val_data]
                    assert len(batch_val) == len(tensors)
                    feed_dict = dict(zip(tensors, batch_val))
                    result = self.sess.run([self.merged], feed_dict=feed_dict)
                    summary_str = result[0]

                    self.writer.add_summary(summary_str, epoch)
                    i += self.batch_size

        if self.embeddings_freq and self.embeddings_data is not None:
            if epoch % self.embeddings_freq == 0:
                # We need a second forward-pass here because we're passing
                # the `embeddings_data` explicitly. This design allows to pass
                # arbitrary data as `embeddings_data` and results from the fact
                # that we need to know the size of the `tf.Variable`s which
                # hold the embeddings in `set_model`. At this point, however,
                # the `validation_data` is not yet set.

                # More details in this discussion:
                # https://github.com/keras-team/keras/pull/7766#issuecomment-329195622

                embeddings_data = self.embeddings_data
                n_samples = embeddings_data[0].shape[0]

                i = 0
                while i < n_samples:
                    step = min(self.batch_size, n_samples - i)
                    batch = slice(i, i + step)

                    if type(self.model.input) == list:
                        feed_dict = {_input: embeddings_data[idx][batch]
                                     for idx, _input in enumerate(self.model.input)}
                    else:
                        feed_dict = {
                            self.model.input: embeddings_data[0][batch]}

                    feed_dict.update({self.batch_id: i, self.step: step})

                    if self.model.uses_learning_phase:
                        feed_dict[K.learning_phase()] = False

                    self.sess.run(self.assign_embeddings, feed_dict=feed_dict)
                    self.saver.save(self.sess,
                                    os.path.join(self.log_dir,
                                                 'keras_embedding.ckpt'),
                                    epoch)

                    i += self.batch_size

        if self.update_freq == 'epoch':
            index = epoch
        else:
            index = self.samples_seen
        # if (index % 50) == 0:
        self._write_logs(logs, index)
        # print(self.var_vgg)
        # print(K.eval(self.var_vgg))
    def _write_logs(self, logs, index):
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            summary = tf.Summary()
            summary_value = summary.value.add()
            if isinstance(value, np.ndarray):
                summary_value.simple_value = value.item()
            else:
                summary_value.simple_value = value
            # summary_value.tag = self.model.mode + '_' + name
            summary_value.tag = name
            self.writer.add_summary(summary, index)

        def tf_summary_image(tensor, mask):
            import io
            from PIL import Image
            import numpy as np
            # pdb.set_trace()
            tensor += [103.939, 116.779, 123.68]
            tensor[tensor < 0] = 0
            tensor[tensor > 255] = 255
            tensor = tensor.astype(np.uint8)
            np.pad(mask, 1, mode='constant')
            mask = np.abs(np.diff(mask[:, :, 0])).astype(np.uint8)
            # mask = [mask np.zeros((mask[:,0,:].shape))]
            # mask = np.matlib.repmat(mask, 1, 1, 3)
            mask = np.stack([mask,mask,mask], axis=2)
            if np.mean(mask.flatten()) < 0.1:
                tensor = np.multiply(tensor[:,:-1,:], 1-mask)
            # ba, hi, wi, ch = tensor.shape
            hi, wi, ch = tensor.shape
            # image = Image.fromarray(tensor[img_id, :, :, ::-1])
            image = Image.fromarray(tensor[:, :, ::-1])
            # image = Image.fromarray(tensor)
            output = io.BytesIO()
            image.save(output, format='PNG')
            image_string = output.getvalue()
            output.close()
            return tf.Summary.Image(height=hi,
                                    width=wi,
                                    colorspace=ch,
                                    encoded_image_string=image_string)
        # cem hacks
        # im_out = tf.clip_by_value(self.model.output+self.mean, 0, 255)
        # im_out = tf.clip_by_value(self.model.output, 0, 255)
        # lab_out = self.model.targets  # + self.mean
        # pdb.set_trace()
        train_pred = K.eval(self.var_y_pred)
        mask = K.eval(self.var_y_mask)
        # train_lab = K.eval(self.var_y_true)
        # with K.get_session().as_default() as ses:
        # img_out = ses.run([im_out])
        # img_out, lab_out = ses.run([self.model.outputs, self.model.inputs])
        # lab_out = lab_out[:, :, :, ::-1]

        # val data
        # val_lab = K.eval(self.validation_data[2])
        # val_pred = self.model.predict(self.validation_data[0:2], steps=1)

        # plot filter responses
        # def feature_to_nets(self, train_pred):
        #     self.model
        #     train_pred

        im_summaries = []
        # for img_id in range(train_pred.shape[0]):
        for img_id in range(4):
            tr_img_sum = tf_summary_image(
                train_pred[img_id, :, :, :], mask[img_id, :, :, :])
            # tr_lab_sum = tf_summary_image(train_lab[img_id, :, :, :])
            # val_img_sum = tf_summary_image(val_pred[img_id, :, :, :])
            # val_lab_sum = tf_summary_image(val_lab[img_id, :, :, :])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(
                tag='train_images_'+str(img_id), image=tr_img_sum))
            # im_summaries.append(tf.Summary.Value(
            #     tag='train_labels_'+str(img_id), image=tr_lab_sum))
            # im_summaries.append(tf.Summary.Value(
            #     tag='val_images_'+str(img_id), image=val_img_sum))
            # im_summaries.append(tf.Summary.Value(
            #     tag='val_labels_'+str(img_id), image=val_lab_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, index)

        # hackend
        self.writer.flush()
