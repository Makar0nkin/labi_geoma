from sidrpy.classes import Vector, Point
from sidrpy.polygon_utils import point_location_from_simple_polygon_octane, point_location_from_convex_polygon
from sidrpy.utils import point_location_from_vec


def calc_frames_number(x_outer, y_outer, x_inner, y_inner, points_and_vectors) -> int:
    points_outer = [Point(*coord) for coord in zip(x_outer, y_outer)]
    # print(f"{points_outer=}")
    lines_outer = [[points_outer[-1], points_outer[0]],
                   *[[points_outer[i], points_outer[i + 1]] for i in range(len(points_outer) - 1)]]
    global_orientation = point_location_from_vec(points_and_vectors[0]["point"], *lines_outer[0])
    frames_counter = 0
    while False in [el["vector"] == Vector() for el in points_and_vectors]:
        for point_vec in points_and_vectors:
            point_vec["point"].x_arr += point_vec["vector"].x_arr
            point_vec["point"].y += point_vec["vector"].y
            if point_location_from_simple_polygon_octane(x_inner, y_inner, point_vec["point"]) == "inside" and\
                    not point_vec["vector"] == Vector():
                point_vec["vector"] = Vector()
            if point_location_from_convex_polygon(x_outer, y_outer, point_vec["point"]) in ["outside", "on polygon"]:
                for line in lines_outer:
                    if point_location_from_vec(point_vec["point"], *line) != global_orientation:
                        v_i = point_vec["vector"]
                        q_j = Vector().from_points(*line)
                        point_vec["vector"] = 2 * ((v_i * q_j) / (q_j * q_j)) * q_j - v_i
                        break
        frames_counter += 1
    return frames_counter