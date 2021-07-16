##################################################
# Train a RAW-to-RGB model using training images #
##################################################
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import os
import torch
import imageio
import tensorflow as tf
import imageio
import numpy as np
import sys
from datetime import datetime
from load_dataset import LoadData
from model import PUNET
import utils
import vgg

# Processing command arguments
dataset_dir, model_dir, result_dir, vgg_dir, dslr_dir, phone_dir,\
    arch, LEVEL, inst_norm, num_maps_base, restore_iter, patch_w, patch_h,\
        batch_size, train_size, learning_rate, eval_step, num_train_iters, save_mid_imgs = \
            utils.process_command_args(sys.argv)

# Defining the size of the input and target image patches
PATCH_WIDTH, PATCH_HEIGHT = patch_w//2, patch_h//2

DSLR_SCALE = float(1) / (2 ** (max(LEVEL,0) - 1))
TARGET_WIDTH = int(PATCH_WIDTH * DSLR_SCALE)
TARGET_HEIGHT = int(PATCH_HEIGHT * DSLR_SCALE)
TARGET_DEPTH = 3
TARGET_SIZE = TARGET_WIDTH * TARGET_HEIGHT * TARGET_DEPTH

NUM_EPOCHS = 10

np.random.seed(0)
train_size = len(os.listdir(os.path.join(dataset_dir, 'train', 'raw')))
# Defining the model architecture
with tf.Graph().as_default(), tf.compat.v1.Session() as sess:
    time_start = datetime.now()

    # determine model name
    if arch == "punet":
        name_model = "punet"
    
    # Placeholders for training data
    phone_ = tf.compat.v1.placeholder(tf.float32, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3])
    dslr_ = tf.compat.v1.placeholder(tf.float32, [batch_size, TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])

    # Get the processed enhanced image
    if arch == "punet":
        enhanced = PUNET(phone_, instance_norm=inst_norm, instance_norm_level_1=False, num_maps_base=num_maps_base)

    # Losses
    enhanced_flat = tf.reshape(enhanced, [-1, TARGET_SIZE])
    dslr_flat = tf.reshape(dslr_, [-1, TARGET_SIZE])

    # MSE loss
    loss_mse = tf.reduce_sum(tf.pow(dslr_flat - enhanced_flat, 2))/(TARGET_SIZE * batch_size)

    # PSNR loss
    loss_psnr = 20 * utils.log10(1.0 / tf.sqrt(loss_mse))

    # SSIM loss
    loss_ssim = tf.reduce_mean(tf.image.ssim(enhanced, dslr_, 1.0))

    # MS-SSIM loss
    loss_ms_ssim = tf.reduce_mean(tf.image.ssim_multiscale(enhanced, dslr_, 1.0))

    # Content loss
    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_ * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size

    # Final loss function
    loss_generator = loss_mse * 20 + loss_content + (1 - loss_ssim) * 20

    # Optimize network parameters
    generator_vars = [v for v in tf.compat.v1.global_variables() if v.name.startswith("generator")]
    train_step_gen = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)

    # Initialize and restore the variables
    print("Initializing variables...")
    sess.run(tf.compat.v1.global_variables_initializer())

    saver = tf.compat.v1.train.Saver(var_list=generator_vars, max_to_keep=100)


    # Loading training and validation data
    print("Loading validation data...")
    test_dataset = LoadData(dataset_dir, 50, DSLR_SCALE, test=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1,
                             pin_memory=True, drop_last=False)

    print("Validation data was loaded\n")

    print("Loading training data...")
    train_dataset = LoadData(dataset_dir, train_size, DSLR_SCALE, test=False)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                              pin_memory=True, drop_last=True)
    print("Training data was loaded\n")



    print("Training network...")


    training_loss = 0.0

    name_model_save = name_model

    for epoch in range(NUM_EPOCHS):
      name_model_save_full = name_model_save + "_epoch_" + str(epoch)
      torch.cuda.empty_cache()
      train_iter = iter(train_loader)

      for i in range(len(train_loader)):

        phone_images, dslr_images = next(train_iter)

        [loss_temp, temp] = sess.run([loss_generator, train_step_gen], feed_dict={phone_: phone_images, dslr_: dslr_images})
        training_loss += loss_temp / eval_step

        if i == 0:

              # Evaluate model
              val_losses = np.zeros((1, 5))

              test_iter = iter(test_loader)
              for j in range(len(test_loader)):

                  torch.cuda.empty_cache()

                  phone_images, dslr_images = next(test_iter)

                  losses = sess.run([loss_generator, loss_content, loss_mse, loss_psnr, loss_ms_ssim], \
                                      feed_dict={phone_: phone_images, dslr_: dslr_images})

                  val_losses += np.asarray(losses) / len(test_loader)

              logs_gen = "Epoch %d | training: %.4g, validation: %.4g | content: %.4g, mse: %.4g, psnr: %.4g, " \
                            "ms-ssim: %.4g\n" % (epoch, training_loss, val_losses[0][0], val_losses[0][1],
                                                  val_losses[0][2], val_losses[0][3], val_losses[0][4])
              print(logs_gen)

              # Save the results to log file
              logs = open(model_dir + "logs_" + str(epoch) + "-" + str(i) + ".txt", "a")
              logs.write(logs_gen)
              logs.write('\n')
              logs.close()

              # Optional: save visual results for several validation image crops
              if False:
                  enhanced_crops = sess.run(enhanced, feed_dict={phone_: phone_images, dslr_: dslr_images})

                  idx = 0
                  for crop in enhanced_crops:
                      if idx < 4:
                          before_after = np.hstack((crop,
                                          np.reshape(visual_target_crops[idx], [TARGET_HEIGHT, TARGET_WIDTH, TARGET_DEPTH])))
                          imageio.imwrite(result_dir + name_model_save_full + "_img_" + str(idx) + ".jpg",
                                          before_after)
                      idx += 1

              # Saving the model that corresponds to the current iteration
              saver.save(sess, model_dir + name_model_save_full + ".ckpt", write_meta_graph=False)

              training_loss = 0.0

          # Loading new training data
        if False:

            del train_data
            del train_answ
            train_data, train_answ = load_train_patch(dataset_dir, dslr_dir, phone_dir, train_size, PATCH_WIDTH, PATCH_HEIGHT, DSLR_SCALE)

    print('total train/eval time:', datetime.now() - time_start)

