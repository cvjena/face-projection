__all__ = ["Warper"]

from typing import Optional

import cv2
import mediapipe as mp
import numpy as np

from .face_model import FaceModel


class Warper:
    def __init__(self) -> None:
        self.face_model = FaceModel()

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

    def __get_landmarks(self, face_img: np.ndarray) -> np.ndarray:
        # this should be put somewhere else
        # locate the face with media pipe
        h, w = face_img.shape[:2]
        with mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(face_img)
            if results.multi_face_landmarks:
                lms = np.array(
                    [
                        [
                            results.multi_face_landmarks[0].landmark[i].x,
                            results.multi_face_landmarks[0].landmark[i].y,
                            results.multi_face_landmarks[0].landmark[i].z,
                        ]
                        for i in range(468)
                    ]
                )
                lms[:, 0] *= w
                lms[:, 1] *= h
                lms = lms.astype(np.int32)
        return lms

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
            raise ValueError(
                f"face_data is not valid for the face model, expected shape: [{self.face_model.height, self.face_model.width, 3}]"
            )

        if landmarks is None:
            landmarks = self.__get_landmarks(face_img)
        else:
            if not isinstance(landmarks, np.ndarray):
                raise TypeError("landmarks must be a numpy array")
            if landmarks.shape[0] != 468:
                raise ValueError("landmarks must have 468 landmarks")

        return self.__warp(
            cooridnates_dst=landmarks[self.face_model.masking],
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
        image_out = np.zeros_like(image_dst)

        rect_src_, rect_dst_, tri_src_crop_, tri_dst_crop_ = [], [], [], []
        depth = np.zeros(len(self.face_model.triangles))

        for idx_tri in range(len(self.face_model.triangles)):
            tri_src = self.face_model.points[self.face_model.triangles[idx_tri]]
            tri_dst = cooridnates_dst[self.face_model.triangles[idx_tri]]
            depth[idx_tri] = np.min(tri_dst, axis=1)[-1]
            tri_dst = np.delete(tri_dst, 2, 1)

            rect_src = cv2.boundingRect(tri_src)
            rect_dst = cv2.boundingRect(tri_dst)

            rect_src_.append(rect_src)
            rect_dst_.append(rect_dst)

            # Offset points by left top corner of the respective rectangles
            tri_src_crop_.append(
                np.array(
                    [
                        ((tri_src[i][0] - rect_src[0]), (tri_src[i][1] - rect_src[1]))
                        for i in range(3)
                    ],
                    dtype=np.float32,
                )
            )
            tri_dst_crop_.append(
                np.array(
                    [
                        ((tri_dst[i][0] - rect_dst[0]), (tri_dst[i][1] - rect_dst[1]))
                        for i in range(3)
                    ],
                    dtype=np.float32,
                )
            )

        # sort by detph
        rect_src_ = [
            x
            for _, x in sorted(
                zip(depth, rect_src_), key=lambda pair: pair[0], reverse=True
            )
        ]
        rect_dst_ = [
            x
            for _, x in sorted(
                zip(depth, rect_dst_), key=lambda pair: pair[0], reverse=True
            )
        ]
        tri_src_crop_ = [
            x
            for _, x in sorted(
                zip(depth, tri_src_crop_), key=lambda pair: pair[0], reverse=True
            )
        ]
        tri_dst_crop_ = [
            x
            for _, x in sorted(
                zip(depth, tri_dst_crop_), key=lambda pair: pair[0], reverse=True
            )
        ]

        for rect_src, rect_dst, tri_src_crop, tri_dst_crop in zip(
            rect_src_, rect_dst_, tri_src_crop_, tri_dst_crop_
        ):
            # Crop input image
            image_src_crop = image_src[
                rect_src[1] : rect_src[1] + rect_src[3],
                rect_src[0] : rect_src[0] + rect_src[2],
            ]
            warping_matrix = cv2.getAffineTransform(tri_src_crop, tri_dst_crop)
            image_layer_t = cv2.warpAffine(
                image_src_crop,
                warping_matrix,
                (rect_dst[2], rect_dst[3]),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )

            # Get mask by filling triangle
            mask_crop = np.zeros((rect_dst[3], rect_dst[2], 3), dtype=np.uint8)
            mask_crop = cv2.fillConvexPoly(
                mask_crop, np.int32(tri_dst_crop), (1, 1, 1), cv2.LINE_AA, 0
            )

            slice_y = slice(rect_dst[1], rect_dst[1] + rect_dst[3])
            slice_x = slice(rect_dst[0], rect_dst[0] + rect_dst[2])

            image_layer_t[mask_crop == 0] = 0
            image_out[slice_y, slice_x] = (
                image_out[slice_y, slice_x] * (1 - mask_crop) + image_layer_t
            )

        mask = np.isclose(image_out, [0, 0, 0])
        out = np.zeros_like(image_out)
        out[mask] = image_dst[mask]
        out[~mask] = image_dst[~mask] * (1 - beta) + image_out[~mask] * beta
        return out
