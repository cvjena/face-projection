from pathlib import Path

import h5py
import numpy as np


class FaceModel:
    """Face model class.

    The underlying UV model is based on the canonical face model by google:
        Link: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
        Image: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    """

    def __init__(self) -> None:
        data_file = h5py.File(Path(__file__).parent / "face_model.h5", "r")

        self.points = np.array(data_file["points"])
        self.triangles = np.array(data_file["triangles"])
        self.facets = np.array(data_file["facets"])
        self.masking = np.array(data_file["masking_canonical"])
        data_file.close()
