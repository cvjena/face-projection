import cv2
import numpy as np
from meshpy import triangle

from . import consts


class FaceModel:
    """Face model class.

    The underlying UV model is based on the canonical face model by google:
        Link: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj
        Image: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

    """

    __HEIGHT: int = 4096
    __WIDTH: int = 4096

    def __init__(self) -> None:
        self.height = self.__HEIGHT
        self.width = self.__WIDTH

        self.scale: float = 1.0

        self.points = consts.FACE_COORDS

        assert len(self.points) == 468, "The number of points must be 468"

        self.hull_idx = cv2.convexHull(self.points, clockwise=False, returnPoints=False)
        self.hull = np.array(
            [self.points[self.hull_idx[i][0]] for i in range(0, len(self.hull_idx))]
        )

        # compute default triangulation
        outer_hull = self.__connect_hull()

        mesh_info = triangle.MeshInfo()
        # set the points from the annotated file
        mesh_info.set_points(self.points.tolist())

        # set the bounding values! ensure that each is a circle like structure
        mesh_info.set_facets(
            outer_hull + consts.EYE_HULL_L_O + consts.EYE_HULL_R_O + consts.LIPS_HULL_O
        )

        # inform the algorithm where some of the whole are!
        mesh_info.set_holes([[1500, 1500], [2500, 1500], [2000, 2800]])

        self.mesh = triangle.build(mesh_info=mesh_info, quality_meshing=False)
        self.points = np.array(self.mesh.points, dtype=np.int32)
        self.triangles = np.array(self.mesh.elements)
        self.facets = np.array(self.mesh.facets)

        self.masking = np.ones(len(consts.FACE_COORDS), dtype=np.bool_)
        self.masking[consts.EYE_HULL_L_IDX] = 0
        self.masking[consts.EYE_HULL_R_IDX] = 0
        self.masking[consts.LIPS_HULL_I] = 0

    def __connect_hull(self) -> list[tuple[int, int]]:
        results = []
        for i in range(len(self.hull_idx) - 1):
            pt1 = self.hull_idx[i]
            pt2 = self.hull_idx[i + 1]
            results.append((pt1, pt2))
        results.append((self.hull_idx[-1], self.hull_idx[0]))
        return np.array(results).squeeze().tolist()

    def set_scale(self, scale: float) -> None:
        """Set the scale of the face model.

        This will change the height and width of the model based on the original
        scale of the model (4096, 4096).

        Args:
            scale (float): The scale of the model.
        """

        if not isinstance(scale, float):
            raise TypeError("scale must be a float")

        if scale <= 0:
            raise ValueError("scale must be greater than 0")

        self.scale = scale

        self.height = int(self.__HEIGHT * scale)
        self.width = int(self.__WIDTH * scale)

        self.points = (np.array(self.mesh.points) * scale).astype(np.int32)

    def create_canvas(self) -> np.ndarray:
        """Create a canvas for the face model.

        Returns:
            np.ndarray: A canvas with the same shape as the face model.
        """
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def check_valid(self, image: np.ndarray) -> bool:
        """Check if the image is valid for the face model.

        Args:
            image (np.ndarray): The image to check.

        Returns:
            bool: True if the image is valid, False otherwise.
        """
        return image.shape[:2] == (self.height, self.width)
