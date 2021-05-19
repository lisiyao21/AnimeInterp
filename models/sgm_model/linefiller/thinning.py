import numpy as np
import cv2


def thinning(fillmap, max_iter=100):
    """Fill area of line with surrounding fill color.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.

    # Returns
        an image.
    """
    line_id = 0
    h, w = fillmap.shape[:2]
    result = fillmap.copy()

    for iterNum in range(max_iter):
        # Get points of line. if there is not point, stop.
        line_points = np.where(result == line_id)
        if not len(line_points[0]) > 0:
            break

        # Get points between lines and fills.
        line_mask = np.full((h, w), 255, np.uint8)
        line_mask[line_points] = 0
        line_border_mask = cv2.morphologyEx(line_mask, cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), anchor=(-1, -1),
                                            iterations=1) - line_mask
        line_border_points = np.where(line_border_mask == 255)

        result_tmp = result.copy()
        # Iterate over points, fill each point with nearest fill's id.
        for i, _ in enumerate(line_border_points[0]):
            x, y = line_border_points[1][i], line_border_points[0][i]

            if x - 1 > 0 and result[y][x - 1] != line_id:
                result_tmp[y][x] = result[y][x - 1]
                continue

            if x - 1 > 0 and y - 1 > 0 and result[y - 1][x - 1] != line_id:
                result_tmp[y][x] = result[y - 1][x - 1]
                continue

            if y - 1 > 0 and result[y - 1][x] != line_id:
                result_tmp[y][x] = result[y - 1][x]
                continue

            if y - 1 > 0 and x + 1 < w and result[y - 1][x + 1] != line_id:
                result_tmp[y][x] = result[y - 1][x + 1]
                continue

            if x + 1 < w and result[y][x + 1] != line_id:
                result_tmp[y][x] = result[y][x + 1]
                continue

            if x + 1 < w and y + 1 < h and result[y + 1][x + 1] != line_id:
                result_tmp[y][x] = result[y + 1][x + 1]
                continue

            if y + 1 < h and result[y + 1][x] != line_id:
                result_tmp[y][x] = result[y + 1][x]
                continue

            if y + 1 < h and x - 1 > 0 and result[y + 1][x - 1] != line_id:
                result_tmp[y][x] = result[y + 1][x - 1]
                continue

        result = result_tmp.copy()

    return result
