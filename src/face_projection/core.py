__all__ = ["Warper"]

from pathlib import Path
from typing import Optional

import cv2
import h5py
import mediapipe as mp
import numpy as np
from meshpy import triangle


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


class Warper:
    """Warper class.

    This class is used to warp the face image to the face model.
    It uses the face model to create a canvas and then uses the landmarks of the face
    if not provided they will be calculated using media pipe.

    This classes should be only instantiated once and then reused as it holds the
    internal buffers for faster processing.
    """

    def __init__(self) -> None:
        self.face_model = FaceModel()

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

        data_emotion = h5py.File(Path(__file__).parent / "emotion_landmarks.h5", "r")
        self.emotion_landmarks = {}
        for emotion in data_emotion.keys():
            self.emotion_landmarks[emotion] = np.array(data_emotion[emotion])
        data_emotion.close()

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
        return self.__landmarks

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

        if img_data.shape[0] != img_data.shape[1]:
            raise ValueError("img_data must be a square image")

        if not isinstance(beta, float) or (beta < 0 or beta > 1):
            raise TypeError("beta must be a float between 0 and 1")

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

    def emotion(
        self,
        emotion: str,
        img_data: np.ndarray[np.uint8],
        beta: float = 0.2,
    ) -> np.ndarray[np.uint8]:
        if not isinstance(img_data, np.ndarray):
            raise TypeError("img_dat must be a numpy array")

        if img_data.shape[0] != img_data.shape[1]:
            raise ValueError("img_data must be a square image")

        if not isinstance(beta, float) or (beta < 0 or beta > 1):
            raise TypeError("beta must be a float between 0 and 1")

        if emotion not in self.emotion_landmarks.keys():
            raise ValueError(f"emotion must be one of the following: {format(self.emotion_landmarks.keys())}")

        landmarks = self.emotion_landmarks[emotion][self.face_model.masking]
        middle_x = int(landmarks[:, 0].mean())
        middle_y = int(landmarks[:, 1].mean())

        padding = 0.3
        x_p = int(middle_x * (1 + padding))
        y_p = int(middle_y * (1 + padding))

        x_diff = x_p - middle_x
        y_diff = y_p - middle_y

        img_face = np.full((y_p * 2, x_p * 2, 3), fill_value=(255, 255, 255), dtype=np.uint8)
        landmarks[:, 0] += x_diff
        landmarks[:, 1] += y_diff

        warp = self.__warp(
            cooridnates_dst=landmarks,
            image_src=img_data,
            image_dst=img_face,
            beta=beta,
        )
        # cut the image to the size of the landmarks
        min_x, min_y = landmarks.min(axis=0)[:2]
        max_x, max_y = landmarks.max(axis=0)[:2]

        warp = warp[min_y:max_y, min_x:max_x]
        return warp

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
        points = np.copy(self.face_model.points)  # are normalized between 0 and 1
        points[:, 0] *= image_src.shape[1]
        points[:, 1] *= image_src.shape[0]
        points = points.astype(np.int32)

        image_out = image_dst.copy()
        # Compute affine transform between src and dst triangles
        for idx_tri in range(self.len_triangles):
            tri_src = points[self.face_model.triangles[idx_tri]]
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

    def unwrap_face(
        self,
        image_src: np.ndarray[np.uint8],
    ) -> np.ndarray[np.uint8]:
        """Warps triangulated area from one image to another image

        TODO THIS can be simplified!

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
        target_size = image_src.shape[0]
        coordinates_src = self.get_landmarks(image_src)
        coordinates_dst = self.face_model.points * (target_size / 4096)
        coordinates_dst = np.concatenate([coordinates_dst, np.ones((coordinates_dst.shape[0], 1))], axis=1, dtype=np.float32)

        points = self.face_model.points
        hull_idx = cv2.convexHull(points, clockwise=False, returnPoints=False)
        # hull = np.array([coordiantes_src[hull_idx[i][0]] for i in range(0, len(hull_idx))])

        # compute default triangulation
        outer_hull = self.face_model.connect_hull(hull_idx)

        mesh_info = triangle.MeshInfo()
        # set the points from the annotated file
        mesh_info.set_points(points.tolist())

        # set the bounding values! ensure that each is a circle like structure
        mesh_info.set_facets(outer_hull)

        # inform the algorithm where some of the whole are!
        # mesh_info.set_holes([[1500, 1500], [2500, 1500], [2000, 2800]])

        mesh = triangle.build(mesh_info=mesh_info, quality_meshing=False, verbose=False)
        points = np.array(mesh.points, dtype=np.int32)
        triangles = np.array(mesh.elements, dtype=np.int32)
        len_triangles = len(triangles)

        rect_src_buffer = np.empty((len_triangles, 4), dtype=np.int32)
        rect_dst_buffer = np.empty((len_triangles, 4), dtype=np.int32)

        tri_src_crop_buffer = np.empty((len_triangles, 3, 2), dtype=np.float32)
        tri_dst_crop_buffer = np.empty((len_triangles, 3, 2), dtype=np.float32)

        buffer_3_2 = np.empty((3, 2), dtype=np.float32)
        depth_buffer = np.empty(len_triangles)

        coordinates_src = np.array(coordinates_src, dtype=int)
        coordinates_dst = np.array(coordinates_dst, dtype=int)
        image_out = np.zeros_like(image_src, dtype=np.uint8)
        image_dst = np.zeros_like(image_src, dtype=np.uint8)
        beta = 1.0

        # Compute affine transform between src and dst triangles
        for idx_tri in range(len_triangles):
            tri_src = coordinates_src[triangles[idx_tri]]
            tri_dst = coordinates_dst[triangles[idx_tri]]

            depth_buffer[idx_tri] = np.min(tri_dst, axis=1)[-1]
            tri_src = np.delete(tri_src, 2, 1)
            tri_dst = np.delete(tri_dst, 2, 1)

            rect_src = cv2.boundingRect(tri_src)
            rect_dst = cv2.boundingRect(tri_dst)

            rect_src_buffer[idx_tri] = rect_src
            rect_dst_buffer[idx_tri] = rect_dst

            # Offset points by left top corner of the respective rectangles
            buffer_3_2[:, 0] = tri_src[:, 0] - rect_src[0]
            buffer_3_2[:, 1] = tri_src[:, 1] - rect_src[1]
            tri_src_crop_buffer[idx_tri] = buffer_3_2

            buffer_3_2[:, 0] = tri_dst[:, 0] - rect_dst[0]
            buffer_3_2[:, 1] = tri_dst[:, 1] - rect_dst[1]
            tri_dst_crop_buffer[idx_tri] = buffer_3_2

        # Sort triangles by depth (furthest to nearest)
        depth_buffer = np.argsort(depth_buffer)[::-1]

        # Warp triangles from src image to dst image
        for idx in range(len_triangles):
            i = depth_buffer[idx]
            # Crop input image
            image_src_crop = image_src[
                rect_src_buffer[i][1] : rect_src_buffer[i][1] + rect_src_buffer[i][3],
                rect_src_buffer[i][0] : rect_src_buffer[i][0] + rect_src_buffer[i][2],
            ]
            warping_matrix = cv2.getAffineTransform(tri_src_crop_buffer[i], tri_dst_crop_buffer[i])
            image_layer_t = cv2.warpAffine(
                image_src_crop,
                warping_matrix,
                (rect_dst_buffer[i][2], rect_dst_buffer[i][3]),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE,
            )

            # Get mask by filling triangle
            mask_crop = np.zeros((rect_dst_buffer[i][3], rect_dst_buffer[i][2], 3), dtype=np.uint8)
            mask_crop = cv2.fillConvexPoly(mask_crop, np.int32(tri_dst_crop_buffer[i]), (1, 1, 1), cv2.LINE_AA, 0)

            slice_y = slice(rect_dst_buffer[i][1], rect_dst_buffer[i][1] + rect_dst_buffer[i][3])
            slice_x = slice(rect_dst_buffer[i][0], rect_dst_buffer[i][0] + rect_dst_buffer[i][2])

            image_layer_t[mask_crop == 0] = 0
            image_out[slice_y, slice_x] = image_out[slice_y, slice_x] * (1 - mask_crop) + image_layer_t

        return cv2.addWeighted(image_dst, 1 - beta, image_out, beta, 0.0, dtype=cv2.CV_8U)
