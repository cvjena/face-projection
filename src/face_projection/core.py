__all__ = ["Warper"]

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .face_model import FaceModel


class Warper:
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
        return self.face_model.width

    def height(self) -> int:
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
        """
        return self.face_model.create_canvas()

    def get_landmarks(self, face_img: np.ndarray) -> np.ndarray:
        # this should be put somewhere else
        # locate the face with media pipe
        h, w = face_img.shape[:2]
        results = self.__face_mesh.process(face_img)
        if results.multi_face_landmarks:
            lms = results.multi_face_landmarks[0].landmark
            for i in range(468):
                self.__landmarks[i, :] = int(lms[i].x * w), int(lms[i].y * h), lms[i].z

    def apply(
        self,
        face_img: np.ndarray[np.int8],
        face_data: np.ndarray[np.int8],
        landmarks: Optional[np.ndarray[np.int32]] = None,
        beta: float = 0.2,
    ) -> np.ndarray[np.int8]:
        if not isinstance(face_img, np.ndarray):
            raise TypeError("face_img must be a numpy array")

        if not isinstance(face_data, np.ndarray):
            raise TypeError("face_data must be a numpy array")

        if not isinstance(beta, float) or (beta < 0 or beta > 1):
            raise TypeError("beta must be a float between 0 and 1")

        if not self.face_model.check_valid(face_data):
            raise ValueError(f"face_data is not valid for the face model, expected shape: [{self.face_model.height, self.face_model.width, 3}]")

        if landmarks is None:
            self.get_landmarks(face_img)
        else:
            if not isinstance(landmarks, np.ndarray):
                raise TypeError("landmarks must be a numpy array")
            if landmarks.shape[0] != 468:
                raise ValueError("landmarks must have 468 landmarks")
            self.__landmarks = landmarks

        return self.__warp(
            cooridnates_dst=self.__landmarks[self.face_model.masking],
            image_src=face_data,
            image_dst=face_img,
            beta=beta,
        )

    def __warp(
        self,
        cooridnates_dst: np.ndarray[np.int32],
        image_src: np.ndarray[np.int8],
        image_dst: np.ndarray[np.int8],
        beta: float = 0.2,
    ) -> np.ndarray[np.int8]:
        """Warps triangulated area from one image to another image

        Args:
            coordiantes_src (np.ndarray[np.float32]): Triangle coordinates source image
            cooridnates_dst (np.ndarray[np.float32]): Triangle coordiantes destination image
            triangles (np.ndarray[np.int32]): Trigulation (array with corner indices)
            image_src (np.ndarray[np.int8]): Source image to take from
            image_dst (np.ndarray[np.int8]): Destination iamge to copy to (is deep copied)
            beta (float, optional): Blending parameter. Defaults to 0.2.

        Returns:
            np.ndarray[np.int8]: _description_
        """
        image_out = image_dst.copy()
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

        self.depth_buffer = np.argsort(self.depth_buffer)[::-1]

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
