"""Tensorflow utility functions for training"""
import logging
import os
from tqdm import trange
import tensorflow as tf
from tensorflow.keras import callbacks as cb
from model.utils import save_dict_to_json
from tensorflow.keras import backend as K
from model.tensorboard_custom import TensorBoardCustom

def train_Upred(model,
               train_inputs,
               train_steps,
               valid_inputs,
               valid_steps,
               current_epoch,
               n_epoch,
               save_dir):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    # callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss',
    #                                       factor=0.5,
    #                                       patience=10,
    #                                       verbose=1,
    #                                       mode='auto',
    #                                       min_delta=0.0001,
    #                                       cooldown=5,
    #                                       min_lr=1e-6))
    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.h5',
                                        monitor='val_loss', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.train_last.h5',
                                        monitor='loss', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=200))    
    # Get relevant graph operations or nodes needed for training
    images, label, masks = train_inputs.get_next()
    val_im, val_lab, val_mask = valid_inputs.get_next()
    return model.fit([images]+[masks], label,
                     steps_per_epoch=train_steps,
                     initial_epoch=current_epoch,
                     epochs=current_epoch+n_epoch,
                     validation_data=(([val_im]+[val_mask], val_lab)),
                     validation_steps=valid_steps,
                     callbacks=callbacks,
                     verbose=1)    

def train_pred(model,
               train_inputs,
               train_steps,
               valid_inputs,
               valid_steps,
               current_epoch,
               n_epoch,
               save_dir):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    # callbacks.append(cb.ReduceLROnPlateau(monitor='val_loss',
    #                                       factor=0.5,
    #                                       patience=15,
    #                                       verbose=1,
    #                                       mode='auto',
    #                                       min_delta=0.0001,
    #                                       cooldown=5,
    #                                       min_lr=1e-6))
    callbacks.append(cb.TensorBoard(log_dir=save_dir,
                                      write_graph=True,
                                      write_images=True,
                                      update_freq='epoch'))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.h5',
                                        monitor='val_loss', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1))
    # Get relevant graph operations or nodes needed for training
    images, label, im_id = train_inputs.get_next()
    val_im, val_lab, val_id = valid_inputs.get_next()
    return model.fit([images], label,
                     steps_per_epoch=train_steps,
                     initial_epoch=current_epoch,
                     epochs=current_epoch+n_epoch,
                     validation_data=((val_im, val_lab)),
                     validation_steps=valid_steps,
                     callbacks=callbacks,
                     verbose=1)

def train_sess(model,
               train_inputs,
               train_steps,
               valid_inputs,
               valid_steps,
               current_epoch,
               n_epoch,
               save_dir):
    """Train the model on `num_steps` batches
    Args:
        model: (Keras Model) contains the graph
        num_steps: (int) train for this number of batches
        current_epoch: (Params) hyperparameters
    """
    callbacks = []
    # callbacks.append(cb.EarlyStopping(monitor="val_ssim",
    #                                   mode="min",
    #                                   patience=10))  # probably
    # callbacks.append(cb.ReduceLROnPlateau(monitor='val_ssim_loss',
    #                                       factor=0.5,
    #                                       patience=10,
    #                                       verbose=1,
    #                                       mode='auto',
    #                                       min_delta=0.0001,
    #                                       cooldown=5,
    #                                       min_lr=1e-6))
    ctb = TensorBoardCustom(log_dir=save_dir,
                            write_graph=True,
                            write_grads=False,
                            write_images=False,
                            update_freq='epoch')
    # callbacks.append(cb.TensorBoard(log_dir=save_dir,
    #                                 # histogram_freq=1,
    #                                 write_graph=True,
    #                                 write_images=False,
    #                                 update_freq='epoch'))
    callbacks.append(cb.ModelCheckpoint(save_dir + '/weights.last.h5',
                                        # monitor='val_loss', # val_ssim
                                        monitor='val_loss', # val_ssim
                                        verbose=1,
                                        save_best_only=True,
                                        save_weights_only=True,
                                        mode='auto',
                                        period=1))
    # Get relevant graph operations or nodes needed for training
    images, masks, mtype = train_inputs.get_next()
    val_im, val_mask, val_mtype = valid_inputs.get_next()
    tr_var = tf.trainable_variables()
    # print(tr_var)
    fetches = [tf.assign(ctb.var_y_true, model.inputs[0], validate_shape=False),
               tf.assign(ctb.var_y_mask, model.inputs[1], validate_shape=False),
               tf.assign(ctb.var_y_pred, model.outputs[0], validate_shape=False),
               # tf.assign(ctb.var_vgg, tr_var[0], validate_shape=False)
               ]
    model._function_kwargs = {'fetches': fetches}
    callbacks.append(ctb)
    return model.fit([images]+[masks], images,
                     steps_per_epoch=train_steps,
                     initial_epoch=current_epoch,
                     epochs=current_epoch+n_epoch,
                     validation_data=(([val_im]+[val_mask], val_im)),
                     validation_steps=valid_steps,
                     callbacks=callbacks,
                     verbose=1)    
    # return model.fit([images]+[masks], labels,
    #                  steps_per_epoch=train_steps,
    #                  initial_epoch=current_epoch,
    #                  epochs=current_epoch+n_epoch,
    #                  validation_data=(([val_im]+[val_mask], val_lab)),
    #                  validation_steps=valid_steps,
    #                  callbacks=callbacks,
    #                  verbose=1)
    # model.fit([images]+[masks], labels,
    #           steps_per_epoch=train_steps,
    #           initial_epoch=current_epoch,
    #           epochs=current_epoch+n_epoch,
    #           validation_split=0.1,
    #           callbacks=callbacks,
    #           verbose=1)
    # def make_iterator(iterator):
    #     # iterator = dataset.make_one_shot_iterator()
    #     next_val = iterator.get_next()
    #     with K.get_session().as_default() as sess:
    #         while True:
    #             inputs, labels, masks = sess.run(next_val)
    #             yield [inputs]+[masks], labels
    # itr_train = make_iterator(train_inputs) 
    # model.fit_generator(generator=itr_train,
    #                      steps_per_epoch=train_steps,
    #                      initial_epoch=current_epoch,
    #                      epochs=current_epoch+n_epoch,
    #                      callbacks=callbacks,
    #                      verbose=1)


# def train_and_evaluate(train_model_spec,
#                        eval_model_spec,
#                        model_dir,
#                        params,
#                        restore_from=None):
#     """Train the model and evaluate every epoch.
#     Args:
#         train_model_spec: (dict) contains the graph ops / nodes need to train
#         eval_model_spec: (dict) contains the graph ops / nodes to evaluate
#         model_dir: (string) directory containing config, weights and log
#         params: (Params) contains hyperparameters of the model.
#                 Must define: num_epochs, train_size, batch_size
#                 Must define: eval_size, save_summary_steps
#         restore_from: (string) directory / file of weights to restore the graph
#     """
#     # Initialize tf.Saver instances to save weights during training
#     last_saver = tf.train.Saver()  # will keep last 5 epochs
#     # only keep 1 best checkpoint (best on eval)
#     best_saver = tf.train.Saver(max_to_keep=3)
#     begin_at_epoch = 0

#     with tf.Session() as sess:
#         # Initialize model variables
#         sess.run(train_model_spec['variable_init_op'])
#         print('RESTORE OR NOT')
#         # Reload weights from directory if specified
#         if restore_from is not None:
#             print('RESTORE OR NOT')
#             logging.info("Restoring parameters from {}".format(restore_from))
#             if os.path.isdir(restore_from):
#                 restore_from = tf.train.latest_checkpoint(restore_from)
#                 begin_at_epoch = int(restore_from.split('-')[-1])
#             last_saver.restore(sess, restore_from)

#         # For tensorboard (takes care of writing summaries to files)
#         train_writer = tf.summary.FileWriter(
#             os.path.join(model_dir, 'train_summaries'), sess.graph)
#         eval_writer = tf.summary.FileWriter(
#             os.path.join(model_dir, 'eval_summaries'), sess.graph)

#         best_eval_acc = 0.0
#         for epoch in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
#             sess.run(train_model_spec['iterator_init_op'])
#             # Run one epoch
#             logging.info("Ep. {}/{}".format(epoch + 1,
#                                             begin_at_epoch + params.num_epochs)
#                          )
#             # Compute number of batches in one epoch (one pass over training)
#             num_steps = (params.train_size + params.batch_size -
#                          1) // params.batch_size
#             train_sess(sess, train_model_spec, num_steps, train_writer, params)
#             # Save weights
#             last_save_path = os.path.join(
#                 model_dir, 'last_weights', 'after-epoch')
#             if not os.path.exists(last_save_path):
#                 os.makedirs(last_save_path)
#             last_saver.save(sess, last_save_path, global_step=epoch + 1)

#             # Evaluate for one epoch on validation set
#             num_steps = (params.eval_size + params.batch_size -
#                          1) // params.batch_size
#             # num_steps = 10
#             metrics = evaluate_sess(
#                 sess, eval_model_spec, num_steps, eval_writer)

#             # If best_eval, best_save_path
#             eval_acc = metrics['loss']
#             if eval_acc < best_eval_acc:
#                 # Store new best accuracy
#                 best_eval_acc = eval_acc
#                 # Save weights
#                 best_pth = os.path.join(
#                     model_dir, 'best_weights', 'after-epoch')
#                 best_pth = best_saver.save(
#                     sess, best_pth, global_step=epoch + 1)
#                 logging.info(
#                     "- Found new best accuracy, saving in {}".format(best_pth))
#                 # Save best eval metrics in a json file in the model directory
#                 best_json_path = os.path.join(
#                     model_dir, "metrics_eval_best_weights.json")
#                 save_dict_to_json(metrics, best_json_path)

#             # Save latest eval metrics in a json file in the model directory
#             last_json_path = os.path.join(
#                 model_dir, "metrics_eval_last_weights.json")
#             save_dict_to_json(metrics, last_json_path)

#             # hyperopt
