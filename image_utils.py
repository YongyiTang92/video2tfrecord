import tensorflow as tf
# Functions for decoding jpgs to tensor


def img_decode_resize_and_crop(jpg_string):
    depth = 3
    img = tf.image.decode_jpeg(jpg_string, channels=depth)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = image_resize(img, 256)
    img = crop_center(tf.expand_dims(img, 0))  # Center cropping
    img = tf.squeeze(img, 0)
    return img


def flow_decode_resize_and_crop(jpg_string):
    depth = 1
    img = tf.image.decode_jpeg(jpg_string, channels=depth)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = image_resize(img, 256)
    img = crop_center(tf.expand_dims(img, 0))  # Center cropping
    img = tf.squeeze(img, 0)
    return img


def img_decode(jpg_string):
    depth = 3
    img = tf.image.decode_jpeg(jpg_string, channels=depth)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img


def flow_decode(jpg_string):
    depth = 1
    img = tf.image.decode_jpeg(jpg_string, channels=depth)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    return img
##

# Functions for image processing or data argumentation


def image_processing(uint8_img_tensor):
    '''
        # Convert image from uint8(0-255) -> float32(-1.0-1.0)
        # Input: uint8_img_tensor: [frames, height, width, depth]
        # Output: float32_img_tensor: [1, frames, height, width, depth]
    '''
    img_data_jpg = tf.image.convert_image_dtype(uint8_img_tensor, dtype=tf.float32)  # [0,255]->[0,1.0]
    img_data_jpg = img_data_jpg * 2 - 1  # [-1.0, 1.0]
    float32_img_tensor = tf.expand_dims(img_data_jpg, 0)  # shape=[1, frames, 224, 224, depth]
    return float32_img_tensor


def image_resize(image_clip, short_size=256):
    '''
        # Resize image/video with shorter size to short_size
        # Input: image_clip: Image with [image_height, image_width, depth] or video with [num_frames, image_height, image_width, depth]
        # Output: resized image_clip with [image_height/ratio, image_height/ratio, depth] or [num_frames, image_height/ratio, image_height/ratio, depth]
    '''
    initial_height = tf.shape(image_clip)[-3]
    initial_width = tf.shape(image_clip)[-2]
    min_ = tf.minimum(initial_width, initial_height)
    ratio = tf.to_float(min_) / tf.constant(short_size, dtype=tf.float32)
    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    resized_image = tf.image.resize_images(image_clip, [new_height, new_width])
    return resized_image


def image_scale_jittering(image_clip, minval=256, maxval=480):
    '''
        # scale_jittering image/video with shorter size randomly from 256 to 480
        # Input: image_clip: Image with [image_height, image_width, depth] or video with [num_frames, image_height, image_width, depth]
        # Output: resized image_clip with [image_height/ratio, image_height/ratio, depth] or [num_frames, image_height/ratio, image_height/ratio, depth]
    '''
    initial_height = tf.shape(image_clip)[-3]
    initial_width = tf.shape(image_clip)[-2]
    min_ = tf.minimum(initial_width, initial_height)
    short_size = tf.random_uniform([1], minval=minval, maxval=maxval, dtype=tf.int32)[0]
    ratio = tf.to_float(min_) / tf.cast(short_size, dtype=tf.float32)
    new_width = tf.to_int32(tf.to_float(initial_width) / ratio)
    new_height = tf.to_int32(tf.to_float(initial_height) / ratio)
    resized_image = tf.image.resize_images(image_clip, [new_height, new_width])
    return resized_image


def crop_center(image_clip, number_of_images, resize_size=[224, 224]):
    initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
    initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
    resize_width = tf.cast(resize_size[0], dtype=tf.float32)
    resize_height = tf.cast(resize_size[1], dtype=tf.float32)

    x1 = 0.5 * (initial_width - resize_width) / initial_width
    y1 = 0.5 * (initial_height - resize_height) / initial_height
    x2 = resize_width / initial_width + x1
    y2 = resize_height / initial_height + y1
    cropped_img = tf.image.crop_and_resize(
        image_clip,
        [[x1, y1, x2, y2] for i in range(number_of_images)],
        [i for i in range(number_of_images)],  # batch index
        resize_size
    )
    return cropped_img


def crop_upper_left(image_clip, number_of_images, resize_size=[224, 224]):
    initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
    initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
    resize_width = tf.cast(resize_size[0], dtype=tf.float32)
    resize_height = tf.cast(resize_size[1], dtype=tf.float32)
    x1 = 0.0
    y1 = 0.0
    x2 = resize_width / initial_width
    y2 = resize_height / initial_height
    cropped_img = tf.image.crop_and_resize(
        image_clip,
        [[x1, y1, x2, y2] for i in range(number_of_images)],
        [i for i in range(number_of_images)],  # batch index
        resize_size
    )
    return cropped_img


def crop_upper_right(image_clip, number_of_images, resize_size=[224, 224]):
    initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
    initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
    resize_width = tf.cast(resize_size[0], dtype=tf.float32)
    resize_height = tf.cast(resize_size[1], dtype=tf.float32)
    x1 = 0.0
    y1 = (initial_height - resize_height) / initial_height
    x2 = resize_width / initial_width
    y2 = 1.0
    cropped_img = tf.image.crop_and_resize(
        image_clip,
        [[x1, y1, x2, y2] for i in range(number_of_images)],
        [i for i in range(number_of_images)],  # batch index
        resize_size
    )
    return cropped_img


def crop_bottom_left(image_clip, number_of_images, resize_size=[224, 224]):
    initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
    initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
    resize_width = tf.cast(resize_size[0], dtype=tf.float32)
    resize_height = tf.cast(resize_size[1], dtype=tf.float32)
    x1 = (initial_width - resize_width) / initial_width
    y1 = 0.0
    x2 = 1.0
    y2 = resize_height / initial_height
    cropped_img = tf.image.crop_and_resize(
        image_clip,
        [[x1, y1, x2, y2] for i in range(number_of_images)],
        [i for i in range(number_of_images)],  # batch index
        resize_size
    )
    return cropped_img


def crop_bottom_right(image_clip, number_of_images, resize_size=[224, 224]):
    initial_width = tf.cast(tf.shape(image_clip)[1], dtype=tf.float32)
    initial_height = tf.cast(tf.shape(image_clip)[2], dtype=tf.float32)
    resize_width = tf.cast(resize_size[0], dtype=tf.float32)
    resize_height = tf.cast(resize_size[1], dtype=tf.float32)
    x1 = (initial_width - resize_width) / initial_width
    y1 = (initial_height - resize_height) / initial_height
    x2 = 1.0
    y2 = 1.0
    cropped_img = tf.image.crop_and_resize(
        image_clip,
        [[x1, y1, x2, y2] for i in range(number_of_images)],
        [i for i in range(number_of_images)],  # batch index
        resize_size
    )
    return cropped_img
##


# Functions for sampling frames
def SampleRandomSequence(model_input, num_frames, num_samples):
    """Samples a random sequence of frames of size num_samples.

    Args:
      model_input: A tensor of size num_seg x frames_in_seg x 3(rgb,flow_x,flow_y)
      num_frames: A tensor of size  1
      num_samples: A scalar

    Returns:
      `model_input`: A tensor of size num_seg x num_samples x 3(rgb,flow_x,flow_y)
    """

    num_seg = tf.shape(model_input)[0]
    num_frames = tf.tile(tf.reshape(tf.cast(num_frames, tf.float32), [1, 1]), [num_seg, 1])
    frame_index_offset = tf.tile(
        tf.expand_dims(tf.range(num_samples), 0), [num_seg, 1])
    max_start_frame_index = tf.maximum(num_frames - num_samples, 0)
    start_frame_index = tf.cast(
        tf.multiply(
            tf.random_uniform([num_seg, 1]),
            tf.cast(max_start_frame_index + 1, tf.float32)), tf.int32)
    frame_index = tf.minimum(start_frame_index + frame_index_offset,
                             tf.cast(num_frames - 1, tf.int32))
    seg_index = tf.tile(
        tf.expand_dims(tf.range(num_seg), 1), [1, num_samples])
    index = tf.stack([seg_index, frame_index], 2)
    return tf.gather_nd(model_input, index)


def SampleRandomFrames(model_input, num_frames, num_samples):
    """Samples a random set of frames of size num_samples.

    Args:
      model_input: A tensor of size num_seg x frames_in_seg x 3(rgb,flow_x,flow_y)
      num_frames: A tensor of size 1, number of frames per clips
      num_samples: A scalar, expect number of frames per flips

    Returns:
      `model_input`: A tensor of size num_seg x num_samples x 3(rgb,flow_x,flow_y)
    """
    num_seg = tf.shape(model_input)[0]
    frame_index = tf.cast(
        tf.multiply(
            tf.random_uniform([num_seg, num_samples]),
            tf.tile(tf.reshape(tf.cast(num_frames, tf.float32), [1, 1]), [num_seg, num_samples])), tf.int32)
    seg_index = tf.tile(
        tf.expand_dims(tf.range(num_seg), 1), [1, num_samples])
    index = tf.stack([seg_index, frame_index], 2)
    return tf.gather_nd(model_input, index)
