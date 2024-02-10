import tensorflow as tf

print(tf.__version__)

import tensorflow as tf

# Check if GPU is available
if tf.test.is_gpu_available():
    print("GPU is available and TensorFlow is using GPU.")
    # Additional information about GPU devices
    gpu_devices = tf.config.list_physical_devices('GPU')
    print("Available GPU devices:")
    for device in gpu_devices:
        print(device)
else:
    print("GPU is not available. TensorFlow is using CPU.")
