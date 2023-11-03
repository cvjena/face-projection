# from meshpy import triangle
# from . import consts
# import cv2


# self.height = self.__HEIGHT
# self.width = self.__WIDTH

# self.scale: float = 1.0

# self.points = consts.FACE_COORDS

# assert len(self.points) == 468, "The number of points must be 468"

# self.hull_idx = cv2.convexHull(self.points, clockwise=False, returnPoints=False)
# self.hull = np.array([self.points[self.hull_idx[i][0]] for i in range(0, len(self.hull_idx))])

# # compute default triangulation
# outer_hull = self.connect_hull(self.hull_idx)

# mesh_info = triangle.MeshInfo()
# # set the points from the annotated file
# mesh_info.set_points(self.points.tolist())

# # set the bounding values! ensure that each is a circle like structure
# mesh_info.set_facets(outer_hull + consts.EYE_HULL_L_O + consts.EYE_HULL_R_O + consts.LIPS_HULL_O)

# # inform the algorithm where some of the whole are!
# mesh_info.set_holes([[1500, 1500], [2500, 1500], [2000, 2800]])

# self.mesh = triangle.build(mesh_info=mesh_info, quality_meshing=False)
# self.points = np.array(self.mesh.points, dtype=np.int32)
# self.triangles = np.array(self.mesh.elements)
# self.facets = np.array(self.mesh.facets)

# self.masking = np.ones(len(consts.FACE_COORDS), dtype=np.bool_)
# self.masking[consts.EYE_HULL_L_IDX] = 0
# self.masking[consts.EYE_HULL_R_IDX] = 0
# self.masking[consts.LIPS_HULL_I] = 0

# def connect_hull(self, hull_idx) -> list[tuple[int, int]]:
#     results = []
#     for i in range(len(hull_idx) - 1):
#         pt1 = hull_idx[i]
#         pt2 = hull_idx[i + 1]
#         results.append((pt1, pt2))
#     results.append((hull_idx[-1], hull_idx[0]))
#     return np.array(results).squeeze().tolist()
