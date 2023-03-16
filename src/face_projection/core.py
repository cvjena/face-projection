__all__ = ["Warper"]

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .face_model import FaceModel


class Warper:
    """Warper class.

    This class is used to warp the face image to the face model.
    It uses the face model to create a canvas and then uses the landmarks of the face
    if not provided they will be calculated using media pipe.

    This classes should be only instantiated once and then reused as it holds the
    internal buffers for faster processing.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.face_model = FaceModel()
        self.face_model.set_scale(scale)

        self.__landmarks = np.zeros((468, 3), dtype=np.int32)
        self.__face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.len_triangles = self.face_model.triangles.shape[0]
        self.rect_src_buffer = np.empty((self.len_triangles, 4), dtype=np.int32)
        self.rect_dst_buffer = np.empty((self.len_triangles, 4), dtype=np.int32)

        self.tri_src_crop_buffer = np.empty((self.len_triangles, 3, 2), dtype=np.float32)
        self.tri_dst_crop_buffer = np.empty((self.len_triangles, 3, 2), dtype=np.float32)

        self.buffer_3_2 = np.empty((3, 2), dtype=np.float32)
        self.depth_buffer = np.empty(self.len_triangles)

    def width(self) -> int:
        """Get the width of the face model."""
        return self.face_model.width

    def height(self) -> int:
        """Get the height of the face model."""
        return self.face_model.height

    def set_scale(self, scale: float) -> None:
        """Set the scale of the face model.

        Args:
            scale (float): The scale of the face model.
        """
        self.face_model.set_scale(scale)

    def create_canvas(self) -> np.ndarray[np.int8]:
        """Create a canvas for the face model.

        This will create a canvas for the face model based on the current scale of the
        face model.

        Returns:
            np.ndarray[np.int8]: A canvas for the face model.

        """
        return self.face_model.create_canvas()

    def get_landmarks(self, face_img: np.ndarray):
        """Get the landmarks of the face image.

        Writes the landmarks of the face image into the internal landmarks buffer.

        TODO: This should be moved somewhere else.

        Args:
            face_img (np.ndarray): The face image

        Returns:
            none

        """
        # this should be put somewhere else
        # locate the face with media pipe
        h, w = face_img.shape[:2]
        results = self.__face_mesh.process(face_img)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            for i in range(468):
                # scale z by w to net be removed by conversion to int
                self.__landmarks[i, :] = int(lms[i].x * w), int(lms[i].y * h), int(lms[i].z * w)

    def apply(
        self,
        img_face: np.ndarray[np.uint8],
        img_data: np.ndarray[np.uint8],
        lms_face: Optional[np.ndarray[np.int32]] = None,
        beta: float = 0.2,
    ) -> np.ndarray[np.uint8]:
        """Warps the face data onto the face image.

        This function warps the face data onto the face image. The face data is warped
        according to the landmarks of the face image. The landmarks of the face image
        are computed with the internal face model.

        Args:
            img_face (np.ndarray[np.uint8]): The face image
            img_data (np.ndarray[np.uint8]): The data which is warped onto the face image
            lms_face (Optional[np.ndarray[np.int32]], optional): The landmarks of the face image. Defaults to None.
            beta (float, optional): Blending parameter. Defaults to 0.2.

        Raises:
            TypeError: If the img_face is not a numpy array
            TypeError: If the img_data is not a numpy array
            TypeError: If the beta is not a float between 0 and 1
            ValueError: If the img_data is not valid for the face model
            TypeError: If the landmarks are not a numpy array

        Returns:
            np.ndarray[np.uint8]: The warped face image
        """
        if not isinstance(img_face, np.ndarray):
            raise TypeError("img_face must be a numpy array")

        if not isinstance(img_data, np.ndarray):
            raise TypeError("img_dat must be a numpy array")

        if not isinstance(beta, float) or (beta < 0 or beta > 1):
            raise TypeError("beta must be a float between 0 and 1")

        if not self.face_model.check_valid(img_data):
            raise ValueError(f"img_dat is not valid for the face model, expected shape: [{self.face_model.height, self.face_model.width, 3}]")

        if lms_face is None:
            self.get_landmarks(img_face)
        else:
            if not isinstance(lms_face, np.ndarray):
                raise TypeError("landmarks must be a numpy array")
            if lms_face.shape[0] != 468:
                raise ValueError("landmarks must have 468 landmarks")
            self.__landmarks = lms_face

        # # check if gray image, if so convert to rgb
        # if np.ndim(img_face) == 2:
        #     img_face = cv2.cvtColor(img_face, cv2.COLOR_GRAY2RGB)

        return self.__warp(
            cooridnates_dst=self.__landmarks[self.face_model.masking],
            image_src=img_data,
            image_dst=img_face,
            beta=beta,
        )

    def __warp(
        self,
        cooridnates_dst: np.ndarray[np.int32],
        image_src: np.ndarray[np.uint8],
        image_dst: np.ndarray[np.uint8],
        beta: float = 0.3,
    ) -> np.ndarray[np.int8]:
        """Warps triangulated area from one image to another image

        Uses the internal face model for triangulation and landmarks.
        The interval buffers are allocated in the constructor once and reused for performance reasons.

        Args:
            cooridnates_dst (np.ndarray[np.float32]): Landmarks of the destination image
            image_src (np.ndarray[np.int8]): Image which is warped onto the destination image
            image_dst (np.ndarray[np.int8]): Destination image where the source image is warped onto
            beta (float, optional): Blending parameter. Defaults to 0.3.

        Returns:
            np.ndarray[np.int8]:  The warped image of the destination image
        """
        image_out = image_dst.copy()
        # Compute affine transform between src and dst triangles
        for idx_tri in range(self.len_triangles):
            tri_src = self.face_model.points[self.face_model.triangles[idx_tri]]
            tri_dst = cooridnates_dst[self.face_model.triangles[idx_tri]]

            self.depth_buffer[idx_tri] = np.min(tri_dst, axis=1)[-1]
            tri_dst = np.delete(tri_dst, 2, 1)

            rect_src = cv2.boundingRect(tri_src)
            rect_dst = cv2.boundingRect(tri_dst)

            self.rect_src_buffer[idx_tri] = rect_src
            self.rect_dst_buffer[idx_tri] = rect_dst

            # Offset points by left top corner of the respective rectangles
            self.buffer_3_2[:, 0] = tri_src[:, 0] - rect_src[0]
            self.buffer_3_2[:, 1] = tri_src[:, 1] - rect_src[1]
            self.tri_src_crop_buffer[idx_tri] = self.buffer_3_2

            self.buffer_3_2[:, 0] = tri_dst[:, 0] - rect_dst[0]
            self.buffer_3_2[:, 1] = tri_dst[:, 1] - rect_dst[1]
            self.tri_dst_crop_buffer[idx_tri] = self.buffer_3_2

        # Sort triangles by depth (furthest to nearest)
        self.depth_buffer = np.argsort(self.depth_buffer)[::-1]

        # Warp triangles from src image to dst image
        for idx in range(self.len_triangles):
            i = self.depth_buffer[idx]
            # Crop input image
            image_src_crop = image_src[
                self.rect_src_buffer[i][1] : self.rect_src_buffer[i][1] + self.rect_src_buffer[i][3],
                self.rect_src_buffer[i][0] : self.rect_src_buffer[i][0] + self.rect_src_buffer[i][2],
            ]
            warping_matrix = cv2.getAffineTransform(self.tri_src_crop_buffer[i], self.tri_dst_crop_buffer[i])
            image_layer_t = cv2.warpAffine(
                image_src_crop,
                warping_matrix,
                (self.rect_dst_buffer[i][2], self.rect_dst_buffer[i][3]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Get mask by filling triangle
            mask_crop = np.zeros((self.rect_dst_buffer[i][3], self.rect_dst_buffer[i][2], 3), dtype=np.uint8)
            mask_crop = cv2.fillConvexPoly(mask_crop, np.int32(self.tri_dst_crop_buffer[i]), (1, 1, 1), cv2.LINE_AA, 0)

            slice_y = slice(self.rect_dst_buffer[i][1], self.rect_dst_buffer[i][1] + self.rect_dst_buffer[i][3])
            slice_x = slice(self.rect_dst_buffer[i][0], self.rect_dst_buffer[i][0] + self.rect_dst_buffer[i][2])

            image_layer_t[mask_crop == 0] = 0
            image_out[slice_y, slice_x] = image_out[slice_y, slice_x] * (1 - mask_crop) + image_layer_t

        return cv2.addWeighted(image_dst, 1 - beta, image_out, beta, 0.0, dtype=cv2.CV_8U)
