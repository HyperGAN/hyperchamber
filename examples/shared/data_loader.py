import glob
import tensorflow as tf
import shared.resize_image_patch

def build_labels(dirs, batch_size):
  next_id=0
  labels = {}
  for dir in dirs:
    labels[dir.split('/')[-1]]=next_id
    next_id+=1
  return labels,next_id
def labelled_image_tensors_from_directory(directory, batch_size):
  filenames = glob.glob(directory+"/**/*.jpg")
  labels,total_labels = build_labels(glob.glob(directory+"/*"), batch_size)
  num_examples_per_epoch = len(filenames)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)
  classes = [labels[f.split('/')[-2]] for f in filenames]
  class_queue = tf.train.input_producer(classes)

  # Read examples from files in the filename queue.
  reader = tf.WholeFileReader()
  key, value = reader.read(filename_queue)
  img = tf.image.decode_jpeg(value, channels=3)
  #img = tf.zeros([64,64,3])
  reshaped_image = tf.cast(img, tf.float32)
  tf.Tensor.set_shape(img, [None, None, None])
  print('img', img)
  label = class_queue.dequeue()

  IMAGE_SIZE=64
  height = IMAGE_SIZE
  width = IMAGE_SIZE

  # Image processing for evaluation.
  # Crop the central [height, width] of the image.
  resized_image = shared.resize_image_patch.resize_image_with_crop_or_pad(reshaped_image,
                                                         width, height, dynamic_shape=True)

  #resized_image = reshaped_image
  tf.Tensor.set_shape(resized_image, [64,64,3])
  print(resized_image)
  #resized_image = tf.image.convert_image_dtype(resized_image, tf.float32)
  # Subtract off the mean and divide by the variance of the pixels.
  #float_image = tf.image.per_image_whitening(resized_image)
  float_image = resized_image / 127.5 - 1.

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  x,y= _get_data(float_image, label, min_queue_examples, batch_size)



  return x, y,total_labels

def _get_data(image, label, min_queue_examples, batch_size):
  num_preprocess_threads = 3
  print(image, label)
  images, label_batch = tf.train.shuffle_batch(
      [image, label],
      batch_size=batch_size,
      num_threads=num_preprocess_threads,
      capacity= 10000,
      min_after_dequeue=100)
  return images, tf.reshape(label_batch, [batch_size])

