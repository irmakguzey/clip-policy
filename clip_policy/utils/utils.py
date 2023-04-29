import liblzfse
import numpy as np

from PIL import Image

def load_image(filepath):
    return np.asarray(Image.open(filepath))


def load_depth(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_conf(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    return depth_img