from source_kitti import KittiSource
from fcnvgg import FCNVGG


def init_data_source():
    return KittiSource()


def init_fcnvgg(session):
    return FCNVGG(session)
