from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import math
import os
import tensorflow as tf
import time
import json
import shutil
import librosa
import csv
import multiprocessing
import subprocess
import pdb
import pandas as pd
import pickle
from functools import partial
from joblib import Parallel
from joblib import delayed

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_videos_in_record', 100,
                     'Number of videos stored in one single tfrecord file')
flags.DEFINE_string('video_source', './samples', 'Directory with video files')
flags.DEFINE_string('video_source_400', './samples', 'Directory with video files')
flags.DEFINE_string('destination', './output_tmp/videos',
                    'Directory for storing tf records')
flags.DEFINE_string('jpg_path', './images_tmp', 'Directory with video files')

flags.DEFINE_string('csv_path', '/dockerdata/trimmed_kinetics_600', 'Directory with csv label files')
flags.DEFINE_integer('FPS', 25,
                     'specifies the FPS to be taken from each video')
flags.DEFINE_string('data_split', 'train', 'data split')

flags.DEFINE_integer('workers', 16,
                     'Number of workers for multiprocessing')
flags.DEFINE_integer('batch_start', 0,
                     'batch_start')
flags.DEFINE_integer('batch_end', 1,
                     'batch_end')

flags.DEFINE_string('denseflow_path', '/dockerdata/tools/dense_flow_opencv3/build/',
                    'path for accessing denseflow tools, version of OpenCV should be 3.3.0 or 4.0.0')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def mp4_to_jpgs(video_name, jpg_path, mp4_filenames):
    '''
    Args: video_name: 
    '''

    # decode mp4 to jpg and optical flow
    # Assume the video has been process with 25 fps, although some video may not meet the requirements
    denseflow_path = FLAGS.denseflow_path

    image_path = os.path.join(jpg_path, video_name)
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    if not os.path.exists(mp4_filenames):
        return False, 'Mp4 not exists.'
    command = ['ffmpeg',
               '-vidFile=%s' % mp4_filenames,
               '-xFlowFile=%s' % (image_path + '/flow_x'),
               '-yFlowFile=%s' % (image_path + '/flow_y'),
               '-imgFile=%s' % (image_path + '/frame'),
               '-bound=20',
               '-type=1',
               '-device_id=0',
               '-step=1']
    command = ' '.join(command)
    # pdb.set_trace()

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, err.output

    return True, 'Decode mp4 success.'


def mp4_to_jpgs_with_flow(video_name, jpg_path, mp4_filenames):
    # decode mp4 to jpg and optical flow
    # Assume the video has been process with 25 fps, although some video may not meet the requirements
    denseflow_path = FLAGS.denseflow_path

    image_path = os.path.join(jpg_path, video_name)
    if not os.path.isdir(image_path):
        os.makedirs(image_path)

    if not os.path.exists(mp4_filenames):
        return False, 'Mp4 not exists.'
    command = [os.path.join(denseflow_path, 'denseFlow_gpu'),
               '-vidFile=%s' % mp4_filenames,
               '-xFlowFile=%s' % (image_path + '/flow_x'),
               '-yFlowFile=%s' % (image_path + '/flow_y'),
               '-imgFile=%s' % (image_path + '/frame'),
               '-bound=20',
               '-type=1',
               '-device_id=0',
               '-step=1']
    command = ' '.join(command)
    # pdb.set_trace()

    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return False, err.output

    return True, 'Decode mp4 success.'


def clear_path(jpg_path, video_name):
    image_path = os.path.join(jpg_path, video_name)
    if os.path.exists(image_path):
        shutil.rmtree(image_path, ignore_errors=False, onerror=None)


def decode_audio(file_path, sample_per_sec=22050):
    audio, _ = librosa.load(file_path, sr=sample_per_sec, mono=False)  # Extract audio
    pro_audio, MFCC_feature = preprocess_audio(audio, sample_per_sec)
    return pro_audio, MFCC_feature


def preprocess_audio(raw_audio, sample_per_sec, minimum_seconds=10):
    # Re-scale audio from [-1.0, 1.0] to [-256.0, 256.0]
    # Return audio with size max(sample_per_sec*minimum_seconds, sample_per_sec*ground_truth_second)
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = librosa.core.to_mono(raw_audio)

    # Make minimum length available
    min_length = sample_per_sec * minimum_seconds
    if min_length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(min_length / raw_audio.shape[0]) + 1)

    MFCC_feature = librosa.feature.mfcc(librosa.core.to_mono(raw_audio), sample_per_sec)  # (20, time*sr/512) -> for 10s video -> ~(20, 432)

    raw_audio[raw_audio < -1.0] = -1.0
    raw_audio[raw_audio > 1.0] = 1.0
    # Make range [-256, 256]
    raw_audio *= 256.0

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    # raw_audio = np.reshape(raw_audio, [1, -1, 1, 1])

    return raw_audio.copy(), MFCC_feature


def processing_tfrecord_upload(_video_name_list, index, destination_path, database, data_split, jpg_path, fps, activity_index, video_path, video_path_400, tfrecord_content_400, tfrecord_content_600):
    print('current index is %d' % index)

    if not os.path.isdir(os.path.join(destination_path, data_split)):
        os.makedirs(os.path.join(destination_path, data_split))
    tfrecord_base = os.path.join(data_split, (data_split + '{:04d}.tfrecord'.format(index)))
    tfrecord_dir = os.path.join(destination_path, tfrecord_base)
    writer = tf.python_io.TFRecordWriter(tfrecord_dir)

    for _video_name in _video_name_list:
        frame_data_tmp = database[database['youtube_id'] == _video_name]  # CSV data for the current video

        for index in range(len(frame_data_tmp)):
            frame_data = frame_data_tmp.iloc[index]
            # decode mp42jpgs with ffmpeg
            video_name = _video_name + '_{:06d}_{:06d}'.format(int(frame_data['time_start'].item()),
                                                               int(frame_data['time_end'].item()))
            mp4_filenames = os.path.join(video_path, video_name + '.mp4')
            mp4_filenames_400 = os.path.join(video_path_400, video_name + '.mp4')
            try:
                # pdb.set_trace()
                assert (os.path.isfile(mp4_filenames) or os.path.isfile(mp4_filenames_400))
                # Decode video to mp4
                # pdb.set_trace()

                if len(frame_data_tmp) == 1 and video_name in tfrecord_content_400.keys():  # if the video contain only one annotation and if the video in tfrecord_content_400
                    # get the images
                    this_tfrecord_content = get_example_from_tfrecord(tfrecord_content_400[video_name], video_name)
                    if this_tfrecord_content is None:
                        status, output_message = mp4_to_jpgs(video_name, jpg_path, mp4_filenames)
                        assert (status is True)
                    elif len(this_tfrecord_content['images']) < 248:
                        status, output_message = mp4_to_jpgs(video_name, jpg_path, mp4_filenames)
                        assert (status is True)
                        this_tfrecord_content = None

                    mp4_filenames_400 = os.path.join(video_path_400, video_name + '.mp4')
                    if this_tfrecord_content is not None and os.path.isfile(mp4_filenames_400):  # if the kinetics-400 data is available
                        # extract audio from the 400 files
                        example = save_jpgs_to_tfrecords(video_name, frame_data, jpg_path,
                                                         fps, activity_index, mp4_filenames_400, tfrecord_content=this_tfrecord_content)
                    else:
                        example = save_jpgs_to_tfrecords(video_name, frame_data, jpg_path,
                                                         fps, activity_index, mp4_filenames, tfrecord_content=None)

                else:  # if the video contain more than one annotation or not in kinetics-400
                    if video_name in tfrecord_content_600.keys():  # if in kinetics-600 records
                        # get the images
                        this_tfrecord_content = get_example_from_tfrecord(tfrecord_content_600[video_name], video_name)
                        if this_tfrecord_content is None:
                            status, output_message = mp4_to_jpgs(video_name, jpg_path, mp4_filenames)
                            assert (status is True)
                        elif len(this_tfrecord_content['images']) < 248:
                            status, output_message = mp4_to_jpgs(video_name, jpg_path, mp4_filenames)
                            assert (status is True)
                            this_tfrecord_content = None
                            # extract audio from the 600 files
                        example = save_jpgs_to_tfrecords(video_name, frame_data, jpg_path,
                                                         fps, activity_index, mp4_filenames, tfrecord_content=this_tfrecord_content)
                    else:
                        status, output_message = mp4_to_jpgs(video_name, jpg_path, mp4_filenames)
                        example = save_jpgs_to_tfrecords(video_name, frame_data, jpg_path,
                                                         fps, activity_index, mp4_filenames, tfrecord_content=None)

                # delete jpgs file
                clear_path(jpg_path, video_name)
                # upload tfrecord according to different subset
                writer.write(example.SerializeToString())
                print('Video %s succeed' % _video_name)

            except:
                print('Video %s failed' % _video_name)
                with open('failed_videos_' + data_split + '.csv', 'a') as myfile:
                    csv_writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                    csv_writer.writerow([video_name])

    writer.close()
    # save tf record and upload
    upload_status, message = upload_tfrecords(tfrecord_dir, tfrecord_base)
    if upload_status:
        os.remove(tfrecord_dir)


def save_jpgs_to_tfrecords(video_name, frame_data, jpg_path, fps, activity_index, mp4_filenames, audio_fps=22050, tfrecord_content=None):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    Args:
        data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos,
        i=number of images, c=number of image channels, h=image height, w=image
        width
        name: filename; data samples type (train|valid|test)
        total_batch_number: indicates the total number of batches
    """
    # if not os.path.isdir(os.path.join(destination_path, video_name)):
    #     os.makedirs(os.path.join(destination_path, video_name))

    if tfrecord_content is not None:
        image_list = tfrecord_content['images']
        flow_x_list = tfrecord_content['flow_x']
        flow_y_list = tfrecord_content['flow_y']
    else:
        images_name = gfile.Glob(os.path.join(jpg_path, video_name, 'frame*.jpg'))  # Read all jpg-name
        images_name.sort()
        image_list = image_to_bytelist(images_name)
        flow_x_name = gfile.Glob(os.path.join(jpg_path, video_name, 'flow_x*.jpg'))  # Read all jpg-name
        flow_x_name.sort()
        flow_y_name = gfile.Glob(os.path.join(jpg_path, video_name, 'flow_y*.jpg'))  # Read all jpg-name
        flow_y_name.sort()
        flow_x_list = flow_to_bytelist(flow_x_name)
        flow_y_list = flow_to_bytelist(flow_y_name)

    # pdb.set_trace()
    audio_flag = 0
    try:
        audio, MFCC_feature = decode_audio(mp4_filenames, audio_fps)
        current_audio_raw = audio.tostring()
        current_mfcc = MFCC_feature.tostring()
        audio_flag = 1
    except:
        audio_len = (len(image_list) // fps) * audio_fps
        audio = np.asarray(np.random.randint(-256, 256, audio_len), np.float32)
        MFCC_feature = np.asarray(np.random.randint(-256, 256, [20, audio_len // 512]), np.float32)
        current_audio_raw = audio.tostring()
        current_mfcc = MFCC_feature.tostring()
        audio_flag = 0

    # pdb.set_trace()
    try:
        info_label = frame_data['label']
        # info_label = frame_data['label'].item()
    except:
        info_label = 'unknown'

    number_of_frames = len(image_list)
    print(len(image_list))
    print(len(flow_x_list))
    assert len(flow_x_list) == len(image_list), "The number of RGB not equal to the number of flow"
    feature_list = {}
    feature_list['images'] = tf.train.FeatureList(feature=image_list)
    feature_list['flow_x'] = tf.train.FeatureList(feature=flow_x_list)
    feature_list['flow_y'] = tf.train.FeatureList(feature=flow_y_list)
    context_features = {}
    context_features['audio'] = _bytes_feature(current_audio_raw)
    context_features['audio_MFCC'] = _bytes_feature(current_mfcc)
    context_features['has_audio'] = _int64_feature(audio_flag)
    context_features['number_of_frames'] = _int64_feature(number_of_frames)
    context_features['video'] = _bytes_feature(str.encode(video_name))
    context_features['label_index'] = _int64_feature(activity_index[info_label])
    context_features['label_name'] = _bytes_feature(str.encode(info_label))

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    # writer.write(example.SerializeToString())
    # writer.close()
    return example


def image_to_bytelist(images_name):
    num_frames = len(images_name)
    resize_image_list = []
    for i_frames in range(num_frames):
        frame = cv2.imread(images_name[i_frames])
        image_encode = cv2.imencode('.jpg', frame)[1]
        image_encode = image_encode.tostring()
        resize_image_list.append(_bytes_feature(image_encode))
    return resize_image_list


def flow_to_bytelist(images_name):
    num_frames = len(images_name)
    resize_image_list = []
    for i_frames in range(num_frames):
        frame = cv2.imread(images_name[i_frames], 0)
        image_encode = cv2.imencode('.jpg', frame)[1]
        image_encode = image_encode.tostring()
        resize_image_list.append(_bytes_feature(image_encode))
    return resize_image_list


def convert_videos_to_tfrecord(video_path, video_path_400, destination_path, jpg_path, csv_path, data_split, csv_file,
                               n_videos_in_record=100, fps=25):
    """calls sub-functions convert_video_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
    Args:
        video_path: directory where video videos are stored
        destination_path: directory where tfrecords should be stored
        n_videos_in_record: Number of videos stored in one single tfrecord file
        video_filenames: specify, if the the full paths to the videos can be
            directly be provided. In this case, the source will be ignored.
    """
    # activity_index, _ = _import_ground_truth('/dockerdata/trimmed_kinetics_600/kinetics-600_train.csv')
    activity_index, _ = _import_ground_truth(os.path.join(csv_path, 'kinetics-600_train.csv'))
    csv_data = pd.read_csv(csv_file)
    if not activity_index:
        raise RuntimeError('No activity_index files found.')
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)

    database = csv_data.sort_values('youtube_id').reset_index(drop=True)  # mess up the video order such that the labels are not the same in a tfrecord
    video_name_list = database['youtube_id']
    # batch_size = FLAGS.workers * 5
    batch_size = n_videos_in_record
    total_video = len(video_name_list)
    # Get the video from existing kinetics-400 tfrecord for saving time
    pkl_name = 'kinetics-400-mapping-%s.pkl' % data_split
    if not os.path.exists(pkl_name):
        tfrecord_content_400 = get_tfrecord_videos(data_split, 400)
        with open(pkl_name, 'wb') as pkl_file:
            pickle.dump(tfrecord_content_400, pkl_file)
    else:
        with open(pkl_name, 'rb') as pkl_file:
            tfrecord_content_400 = pickle.load(pkl_file)

    # Get the video from existing kinetics-400 tfrecord for saving time
    pkl_name = 'kinetics-600-mapping-%s.pkl' % data_split
    if not os.path.exists(pkl_name):
        tfrecord_content_600 = get_tfrecord_videos(data_split, 600)
        with open(pkl_name, 'wb') as pkl_file:
            pickle.dump(tfrecord_content_600, pkl_file)
    else:
        with open(pkl_name, 'rb') as pkl_file:
            tfrecord_content_600 = pickle.load(pkl_file)

    print('kinetics-400-split', data_split, ' contain: ', len(tfrecord_content_400), 'video')
    print('kinetics-600-split', data_split, ' contain: ', len(tfrecord_content_600), 'video')

    st = time.time()
    for i in range(FLAGS.batch_start, min(total_video // batch_size + 1, FLAGS.batch_end)):
        # for i in range(FLAGS.batch_start, FLAGS.batch_end):
        print('Processing {:04d} of {:04d} batches, time/batch: {:.4f}s'.format(i + 1, min(total_video // batch_size + 1,
                                                                                           FLAGS.batch_end), time.time() - st))
        st = time.time()
        processing_tfrecord_upload(video_name_list[i * batch_size: min(len(database), (i + 1) * batch_size)],
                                   index=i,
                                   destination_path=destination_path, database=database, data_split=data_split,
                                   jpg_path=jpg_path, fps=fps, activity_index=activity_index, video_path=video_path, video_path_400=video_path_400, tfrecord_content_400=tfrecord_content_400, tfrecord_content_600=tfrecord_content_600)


if __name__ == '__main__':
    # pdb.set_trace()
    if FLAGS.data_split == 'train':
        csv_file = os.path.join(FLAGS.csv_path, 'kinetics-600_train.csv')
    elif FLAGS.data_split == 'val':
        csv_file = os.path.join(FLAGS.csv_path, 'kinetics-600_val.csv')
    elif FLAGS.data_split == 'test':
        csv_file = os.path.join(FLAGS.csv_path, 'kinetics-600_test.csv')
    else:
        raise('wrong data_split', FLAGS.data_split)
        # return 0
    video_path = os.path.join(FLAGS.video_source, FLAGS.data_split)
    video_path_400 = os.path.join(FLAGS.video_source_400, FLAGS.data_split)
    convert_videos_to_tfrecord(video_path, video_path_400, destination_path=FLAGS.destination, jpg_path=FLAGS.jpg_path, csv_path=FLAGS.csv_path, data_split=FLAGS.data_split, csv_file=csv_file,
                               n_videos_in_record=FLAGS.n_videos_in_record, fps=FLAGS.FPS)

    # Check mp4_to_jpgs
    # status, message = mp4_to_jpgs('aRWe4Qpte0s_000035_000045', './tmp', '/dockerdata/trimmed_kinetics_600/val/aRWe4Qpte0s_000035_000045.mp4')
    # print(message)

    # Check decode_audio
    # pro_audio, MFCC_feature = decode_audio('/dockerdata/trimmed_kinetics_600/val/aRWe4Qpte0s_000035_000045.mp4')
    # print(pro_audio.shape)
    # print(MFCC_feature.shape)

    # Check _import_ground_truth
    # pdb.set_trace()
    # activity_index, _ = _import_ground_truth('/dockerdata/trimmed_kinetics_600/kinetics-600_train.csv')

    # database = pd.read_csv('/dockerdata/trimmed_kinetics_600/kinetics-600_val.csv')
    # _video_name = 'aRWe4Qpte0s'
    # frame_data = database[database['youtube_id'] == _video_name]  # CSV data for the current video
    # # decode mp42jpgs with ffmpeg
    # video_name = _video_name + '_{:06d}_{:06d}'.format(int(frame_data['time_start'].item()),
    #                                                    int(frame_data['time_end'].item()))
    # mp4_filenames = '/dockerdata/trimmed_kinetics_600/val/aRWe4Qpte0s_000035_000045.mp4'
    # status, message = mp4_to_jpgs(video_name, './tmp', mp4_filenames)
    # example = save_jpgs_to_tfrecords(video_name, frame_data, './tmp',
    #                                  25, activity_index, mp4_filenames)
    # pdb.set_trace()
