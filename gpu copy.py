import tensorflow as tf

# 檢查是否使用了GPU CUDA
if tf.test.is_built_with_cuda():
    print("TensorFlow已經使用了GPU CUDA")
else:
    print("TensorFlow沒有使用GPU CUDA")

# 檢查顯卡型號
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    for device in gpu_devices:
        print("顯卡型號:", tf.config.experimental.get_device_details(
            device)["name"])
else:
    print("沒有找到可用的顯卡")
