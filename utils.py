from source_kitti import KittiSource
from fcnvgg import FCNVGG


def load_data_source():
    return KittiSource()


def load_fcnvgg(session, num_classes):
    return FCNVGG(session, num_classes)
