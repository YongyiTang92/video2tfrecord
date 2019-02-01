from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import tensorflow as tf
import image_utils
from tensorflow.python.platform import gfile


class DataReader(object):
    def __init__(self, file_pattern, batch_size, num_classes, num_epoch=None, fps=24,
                 data_type='rgb', max_frames=250, num_seg=3, frame_per_seg=1, sampling_sequence=True,
                 scale_jittering=True, scale_min=256, scale_max=480, crop_size=224, is_training=False,
                 num_parallel_reads=16, shuffle_buffer_size=500, prefetch=1):
        """Define the dataset to extract feature (and flow images) from tfrecords

        Args:
          file_pattern: file pattern of tfrecords
          batch_size: batch size
          num_classes: number of classes of this dataset. ActivityNet:200, Kinetics:400, Kinetics-600: 600
          num_epoch: number of epoch
          data_type: if 'rgb': output [frames, height, width, 3]; elif 'flow' output [frames, height, width, 2]; 
                    elif 'all' output [frames, height, width, 5] where the first 3 dims are rgb and the 4-5 are flow-x and flow-y
          max_frames: maximum number of frames per video
          num_seg: the video first segment into num_seg cilp, then sample frame_per_seg frames either independent sampling or sampling a sub-sequence
          frame_per_seg: number of frame sameples per seg
          sampling_sequence: the output shape is [num_seg*frame_per_seg, height, width, channels], 
                             if true the frame_per_seg should be consucutive, other wise it is independent samples
          scale_jittering: if True: use scale jittering from [256,480] for data argumentation, else: resize video with short size equal to 256
          crop_size: size for cropping
          is_training: training flag. If True: data argumentation will be apply, otherwise, only resize to 256 and center cropping with 224
          num_parallel_reads, shuffle_buffer_size, num_parallel_calls: parameters for tf.dataset.TFRecordDataset

          For example: if num_seg=3, frame_per_seg=10, sampling_sequence=True, it will return [3*10, h, w, c] images, 
                       where it is 10 consecutive frames per segment and there are 3 segment in total.
                       if num_seg=3, frame_per_seg=10, sampling_sequence=False, it will return [3*10, h, w, c] images, 
                       where it is 10 independent frames per segment from the same segment and there are 3 segment in total.

          Currently, the output data are range from [0-255].
        """
        self.file_pattern = file_pattern
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_seg = num_seg
        self.max_frames = num_seg * (max_frames // num_seg)
        self.frame_per_seg = frame_per_seg
        self.sampling_sequence = sampling_sequence
        self.scale_jittering = scale_jittering
        self.crop_size = crop_size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.is_training = is_training
        self.fps = fps

        self.num_epoch = num_epoch
        self.shuffle_buffer_size = shuffle_buffer_size
        self.num_parallel_reads = num_parallel_reads
        self.prefetch = prefetch
        self.data_type = data_type
        assert (self.max_frames is not None)

    def get_dataset(self):
        files = gfile.Glob(self.file_pattern)
        number_of_files = len(files)
        files = tf.data.Dataset.from_tensor_slices(files)
        files = files.shuffle(number_of_files)  # shuffle across tfrecords --> global shuffle
        dataset = files.apply(tf.data.experimental.parallel_interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=5, block_length=100))
        # You can use this line instead if you use TF<1.12.0
        # dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x), cycle_length=5, block_length=100)

        if self.is_training is True:
            dataset = dataset.repeat(self.num_epoch)
            dataset = dataset.shuffle(self.shuffle_buffer_size)
        dataset = dataset.map(self.parse_example_proto, num_parallel_calls=self.num_parallel_reads)
        dataset = dataset.filter(filter_func)  # skip the empty data
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(self.prefetch)
        return dataset

    def parse_example_proto(self, example_serialized):
            # TBD
            # 1. define self.frame_per_clip
            # 2. well define feature's key words
        """Parses an Example proto containing a training example of an image.
        Args:
            example_serialized: scalar Tensor tf.string containing a serialized
                Example protocol buffer.

        Returns:
            image_buffer: Tensor tf.string containing the contents of a JPEG file.
            label: Tensor tf.int32 containing the label.
            bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
                where each coordinate is [0, 1) and the coordinates are arranged as
                [ymin, xmin, ymax, xmax].
            text: Tensor tf.string containing the human-readable label.
        """
        # Dense features in Example proto.
        contexts, features = tf.parse_single_sequence_example(
            example_serialized,
            context_features={"number_of_frames": tf.FixedLenFeature([], tf.int64)},
            sequence_features={
                'images': tf.FixedLenSequenceFeature([], dtype=tf.string),
            })

        # Truncate or pad images
        images = tf.cond(tf.shape(features['images'])[0] > self.num_seg * self.frame_per_seg,
                         lambda: self.truncate(features['images']),
                         lambda: self.padding(features['images']))

        # sampling frames with specific fps
        if 24.0 % self.fps != 0:
            print("Warning, the fps is {} ".format(self.fps))
        interval = int(24.0 / self.fps)
        images, flow_x, flow_y = images[::interval], flow_x[::interval], flow_y[::interval]

        # sampling from segments
        images, flow_x, flow_y = tf.cond(tf.shape(features['images'])[0] > 0,
                                         lambda: self.sampling(images, flow_x, flow_y),  # if is not empty, do the sampling
                                         lambda: self.identity(images, flow_x, flow_y))  # else leave it

        # Decode jpg-string to TF-tensor
        output = tf.map_fn(image_utils.img_decode, images, dtype=tf.float32)

        # output shoud have shorter side as 256 or jittered scale, pixel values are range from [0, 1.0]

        if self.is_training is True:
            # scale jittering: random resize a short side range in [256, 480]
            if self.scale_jittering:
                output = image_utils.image_scale_jittering(output, self.scale_min, self.scale_max)  # it works for multiple-channels
            else:
                output = image_utils.image_resize(output, self.scale_min)
            # horizontal filping
            random_float = tf.random_uniform([1], minval=0, maxval=1, dtype=tf.float32)[0]
            output = tf.cond(random_float > 0.5, lambda: tf.image.flip_left_right(output), lambda: output)
            # corner or center cropping
            random_int = tf.random_uniform([1], minval=0, maxval=6, dtype=tf.int32)[0]
            output = tf.case({tf.equal(random_int, tf.constant(0, dtype=tf.int32)): lambda: self._crop_center(output),
                              tf.equal(random_int, tf.constant(1, dtype=tf.int32)): lambda: self._crop_upper_left(output),
                              tf.equal(random_int, tf.constant(2, dtype=tf.int32)): lambda: self._crop_upper_right(output),
                              tf.equal(random_int, tf.constant(3, dtype=tf.int32)): lambda: self._crop_bottom_left(output),
                              tf.equal(random_int, tf.constant(4, dtype=tf.int32)): lambda: self._crop_bottom_right(output),
                              }, default=lambda: self._crop_center(output), exclusive=True)
        else:
            output = image_utils.image_resize(output, self.scale_min)  # it works for multiple-channels
            output = self._crop_center(output)  # center crop only

        output = (output * 2) - 1.0  # range from [-1.0, 1.0] with shape (frames_per_clip, height, width, channels)
        # output = tf.transpose(output, [0, 3, 1, 2])  # from (frames_per_clip, height, width, channels) -> (frames_per_clip, channels, height, width)

        return output, label  # , contexts, features

    def padding(self, serialized_string):
        # padding serialized_string to self.max_frames
        shape = tf.shape(serialized_string)
        pad = tf.cond(shape[0] > 0,
                      lambda: tf.maximum(0, (self.num_seg * self.frame_per_seg) - shape[0]),
                      lambda: tf.maximum(0, self.max_frames - shape[0]))
        # pad = tf.maximum(0, (self.num_seg * self.frame_per_seg) - shape[0])
        repeat_last = tf.tile(serialized_string[-1:], [pad])
        x_images = tf.concat([serialized_string, repeat_last], 0)
        return x_images

    def truncate(self, serialized_string):
        # truncating serialized_string to self.max_frames
        # x_images = serialized_string[:self.max_frames]
        end_index = tf.cast(tf.shape(serialized_string)[0] / (self.num_seg * self.frame_per_seg), dtype=tf.int32) * (self.num_seg * self.frame_per_seg)
        x_images = tf.slice(serialized_string, [0], [end_index])
        return x_images

    def sampling(self, images_in, flow_x_in, flow_y_in):
        # Random sample a frame from a segment
        # segment a sequence and sample a frame from it
        if self.num_seg is not None:
            images = tf.reshape(images_in, [self.num_seg, tf.shape(images_in)[0] // self.num_seg])
            flow_x = tf.reshape(flow_x_in, [self.num_seg, tf.shape(flow_x_in)[0] // self.num_seg])
            flow_y = tf.reshape(flow_y_in, [self.num_seg, tf.shape(flow_y_in)[0] // self.num_seg])

            seg_input = tf.stack([images, flow_x, flow_y], axis=2)
            if self.sampling_sequence:
                seg_output = image_utils.SampleRandomSequence(seg_input, tf.shape(images_in)[0] // self.num_seg, self.frame_per_seg)
            else:  # sampling independent frames
                seg_output = image_utils.SampleRandomFrames(seg_input, tf.shape(images_in)[0] // self.num_seg, self.frame_per_seg)
            images, flow_x, flow_y = tf.split(seg_output, 3, axis=2)
            images = tf.reshape(images, [self.num_seg * self.frame_per_seg])
            flow_x = tf.reshape(flow_x, [self.num_seg * self.frame_per_seg])
            flow_y = tf.reshape(flow_y, [self.num_seg * self.frame_per_seg])
        return images, flow_x, flow_y

    def identity(self, images, flow_x, flow_y):
        return images, flow_x, flow_y

    def _crop_center(self, input_images):
        return image_utils.crop_center(input_images, number_of_images=self.num_seg * self.frame_per_seg, resize_size=[self.crop_size, self.crop_size])

    def _crop_upper_left(self, input_images):
        return image_utils.crop_upper_left(input_images, number_of_images=self.num_seg * self.frame_per_seg, resize_size=[self.crop_size, self.crop_size])

    def _crop_upper_right(self, input_images):
        return image_utils.crop_upper_right(input_images, number_of_images=self.num_seg * self.frame_per_seg, resize_size=[self.crop_size, self.crop_size])

    def _crop_bottom_left(self, input_images):
        return image_utils.crop_bottom_left(input_images, number_of_images=self.num_seg * self.frame_per_seg, resize_size=[self.crop_size, self.crop_size])

    def _crop_bottom_right(self, input_images):
        return image_utils.crop_bottom_right(input_images, number_of_images=self.num_seg * self.frame_per_seg, resize_size=[self.crop_size, self.crop_size])
