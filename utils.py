from source_kitti import KittiSource
from source_voc import VOCSource
from fcnvgg import FCNVGG


def load_kitti_source():
    return KittiSource()


def load_voc_source():
    return VOCSource()


def load_fcnvgg(session, num_classes):
    return FCNVGG(session, num_classes)
