from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import os
import tensorflow as tf
import time
import shutil
import subprocess

FLAGS = flags.FLAGS
flags.DEFINE_string('video_source', './samples', 'Directory with video files')
flags.DEFINE_string('output_dir', './output_tmp',
                    'Directory for storing tf records')
flags.DEFINE_string('jpg_path', './images_tmp', 'Directory with video files')

flags.DEFINE_integer('FPS', 25,
                     'specifies the FPS to be taken from each video')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def mp4_to_jpgs(image_path, mp4_filenames, fps):
    # convert original video to mp4_filenames;
    # and extract jpgs and optical flows
    # image_path = os.path.join(jpg_path, video_name)
    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # For instance you want to convert to a video with fps first
    # video_command = ['ffmpeg',
    #                  '-i', '"%s"' % mp4_filenames,
    #                  '-c:v', 'libx264', '-acodec', 'aac',
    #                  '-strict', '-2',
    #                  '-max_muxing_queue_size', '1024',
    #                  '-threads', '1',
    #                  '-loglevel', 'panic',
    #                  '-r', '"%s"' % fps,
    #                  '"%s"' % mp4_filenames]

    # Decode video to images
    video_command = ['ffmpeg',
                     '-i', '"%s"' % mp4_filenames,
                     '-c:v', 'libx264',
                     '-strict', '-2',
                     '-max_muxing_queue_size', '1024',
                     '-threads', '1',
                     '-loglevel', 'panic',
                     '-r', '"%s"' % fps,
                     '"%s"' % (mp4_filenames + '/frame_%06d.jpg')]
    video_command = ' '.join(video_command)
    try:
        output = subprocess.check_output(video_command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, 'decode mp4 error'

    return True, 'Decode mp4 success.'


def clear_path(image_path):
    if os.path.exists(image_path):
        shutil.rmtree(image_path, ignore_errors=False, onerror=None)


def save_jpgs_to_tfrecords(image_patterns):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    Args:
        image_patterns: image file pattern, e.g. '/data/video/xxx/*.jpg'
    Return:
        example: tf.train.SequenceExample instance that stores data.
    """
    images_name = gfile.Glob(image_patterns)  # Read all jpg-name
    images_name.sort()  # sort images by names
    image_list = image_to_bytelist(images_name)

    number_of_frames = len(image_list)

    feature_list = {}
    feature_list['images'] = tf.train.FeatureList(feature=image_list)
    context_features = {}
    context_features['number_of_frames'] = _int64_feature(number_of_frames)

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    return example


def image_to_bytelist(images_name):
    # Read jpg files to byte list
    num_frames = len(images_name)
    resize_image_list = []
    for i_frames in range(num_frames):
        frame = cv2.imread(images_name[i_frames])
        # You may need to resize video here
        # Then encode it back to .jpg file
        image_encode = cv2.imencode('.jpg', frame)[1]
        image_encode = image_encode.tostring()
        resize_image_list.append(_bytes_feature(image_encode))
    return resize_image_list


def main():
    video_source = FLAGS.video_source
    jpg_path = FLAGS.jpg_path

    video_list = gfile.Glob(video_source + '/*.mp4')

    if len(video_list) > 0:
        tfrecord_name = os.path.join(FLAGS.output_dir, 'test.tfrecords')
        writer = tf.python_io.TFRecordWriter(tfrecord_name)
        for mp4_filenames in video_list:
            fps = FLAGS.FPS
            video_name = mp4_filenames.split('/')[-1].split('.')[0]
            image_path = os.path.join(jpg_path, video_name)

            status, message = mp4_to_jpgs(image_path, mp4_filenames, fps)
            if status:
                example = save_jpgs_to_tfrecords(image_path + '/*.jpg')
                clear_path(image_path)

        writer.close()
    else:
        print('Not found any video exist in %s ' % video_source)


if __name__ == '__main__':
    main()
