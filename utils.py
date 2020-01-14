from source_kitti import KittiSource
from fcnvgg import FCNVGG


def load_data_source():
    return KittiSource()


def load_fcnvgg(session):
    return FCNVGG(session)
