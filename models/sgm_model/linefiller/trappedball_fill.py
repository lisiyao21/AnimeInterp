import cv2
import numpy as np

def calculate_color_hist(mask, img):
    """
        Calculate colour histogram for the region.
        The output will be an array with n_BINS * n_color_channels.
        The number of channel is varied because of different
        colour spaces.
    """

    BINS = 25
    if len(img.shape) == 2:
        img = img.reshape(img.shape[0], img.shape[1], 1)

    channel_nums = img.shape[2]
    hist = np.array([])

    for channel in range(channel_nums):
        layer = img[:, :, channel][mask]
        hist = np.concatenate([hist] + [np.histogram(layer, BINS)[0]])

    # L1 normalize
    hist = hist / np.sum(hist)

    return hist

def get_ball_structuring_element(radius):
    """Get a ball shape structuring element with specific radius for morphology operation.
    The radius of ball usually equals to (leaking_gap_size / 2).
    
    # Arguments
        radius: radius of ball shape.
             
    # Returns
        an array of ball structuring element.
    """
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(2 * radius + 1), int(2 * radius + 1)))


def get_unfilled_point(image):
    """Get points belong to unfilled(value==255) area.

    # Arguments
        image: an image.

    # Returns
        an array of points.
    """
    y, x = np.where(image == 255)

    return np.stack((x.astype(int), y.astype(int)), axis=-1)


def exclude_area(image, radius):
    """Perform erosion on image to exclude points near the boundary.
    We want to pick part using floodfill from the seed point after dilation. 
    When the seed point is near boundary, it might not stay in the fill, and would
    not be a valid point for next floodfill operation. So we ignore these points with erosion.

    # Arguments
        image: an image.
        radius: radius of ball shape.

    # Returns
        an image after dilation.
    """
    return cv2.morphologyEx(image, cv2.MORPH_ERODE, get_ball_structuring_element(radius), anchor=(-1, -1), iterations=1)


def trapped_ball_fill_single(image, seed_point, radius):
    """Perform a single trapped ball fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
        radius: radius of ball shape.
    # Returns
        an image after filling.
    """
    ball = get_ball_structuring_element(radius)
    # make a ball

    pass1 = np.full(image.shape, 255, np.uint8)

    # cv2.imwrite('pass1.png', pass1)
    pass2 = np.full(image.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(image)
    # black --> white, white --> black
    # cv2.imwrite('im_inv.png', im_inv)

    # Floodfill the image
    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    # add borders
    # # cv2.imwrite('mask1.png', mask1)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)


    # cv2.imwrite('pass1_after_fill.png', pass1)

    # Perform dilation on image. The fill areas between gaps became disconnected.
    pass1 = cv2.morphologyEx(pass1, cv2.MORPH_DILATE, ball, anchor=(-1, -1), iterations=1)

    # cv2.imwrite('pass1_after_dilate.png', pass1)


    mask2 = cv2.copyMakeBorder(pass1, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)

    # Floodfill with seed point again to select one fill area.
    _, pass2, _, rect = cv2.floodFill(pass2, mask2, seed_point, 0, 0, 0, 4)
    # cv2.imwrite('pass2.png', pass2)
    # Perform erosion on the fill result leaking-proof fill.
    pass2 = cv2.morphologyEx(pass2, cv2.MORPH_ERODE, ball, anchor=(-1, -1), iterations=1)
    # cv2.imwrite('pass2_after_erode.png', pass2)

    return pass2


def trapped_ball_fill_multi(image, radius, method='mean', max_iter=1000):
    """Perform multi trapped ball fill operations until all valid areas are filled.

    # Arguments
        image: an image. The image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        radius: radius of ball shape.
        method: method for filtering the fills. 
               'max' is usually with large radius for select large area such as background.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    print('++ trapped-ball ' + str(radius))

    unfill_area = image
    filled_area, filled_area_size, result = [], [], []

    for _ in range(max_iter):
        points = get_unfilled_point(exclude_area(unfill_area, radius))

        # find positions of all white pixel

        if not len(points) > 0:
            break

        fill = trapped_ball_fill_single(unfill_area, (points[0][0], points[0][1]), radius)
        # start from the first point position in {points}

        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))
        filled_area_size.append(len(np.where(fill == 0)[0]))

    filled_area_size = np.asarray(filled_area_size)

    if method == 'max' or method == 'Q4':
        area_size_filter = np.max(filled_area_size)
    elif method == 'median' or method == 'Q2':
        area_size_filter = np.median(filled_area_size)
    elif method == 'Q1':
        sortedSizeArr = np.sort(filled_area_size)
        Q1_idx = int(len(sortedSizeArr)/4)
        area_size_filter = sortedSizeArr[Q1_idx]
    elif method == 'mean':
        area_size_filter = np.mean(filled_area_size)
    else:
        area_size_filter = 0

    result_idx = np.where(filled_area_size >= area_size_filter)[0]

    for i in result_idx:
        result.append(filled_area[i])

    return result


def flood_fill_single(im, seed_point):
    """Perform a single flood fill operation.

    # Arguments
        image: an image. the image should consist of white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        seed_point: seed point for trapped-ball fill, a tuple (integer, integer).
    # Returns
        an image after filling.
    """
    pass1 = np.full(im.shape, 255, np.uint8)

    im_inv = cv2.bitwise_not(im)

    mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
    _, pass1, _, _ = cv2.floodFill(pass1, mask1, seed_point, 0, 0, 0, 4)

    return pass1


def flood_fill_multi(image, max_iter=20000):
    """Perform multi flood fill operations until all valid areas are filled.
    This operation will fill all rest areas, which may result large amount of fills.

    # Arguments
        image: an image. the image should contain white background, black lines and black fills.
               the white area is unfilled area, and the black area is filled area.
        max_iter: max iteration number.
    # Returns
        an array of fills' points.
    """
    print('floodfill')

    unfill_area = image
    filled_area = []

    for _ in range(max_iter):
        points = get_unfilled_point(unfill_area)

        if not len(points) > 0:
            break

        fill = flood_fill_single(unfill_area, (points[0][0], points[0][1]))
        unfill_area = cv2.bitwise_and(unfill_area, fill)

        filled_area.append(np.where(fill == 0))

    return filled_area


def mark_fill(image, fills):
    """Mark filled areas with 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    result = image.copy()

    for fill in fills:
        result[fill] = 0

    return result


def build_fill_map(image, fills):
    """Make an image(array) with each pixel(element) marked with fills' id. id of line is 0.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an array.
    """
    result = np.zeros(image.shape[:2], np.int)

    for index, fill in enumerate(fills):
        result[fill] = index + 1

    return result


def show_fill_map(fillmap):
    """Mark filled areas with colors. It is useful for visualization.

    # Arguments
        image: an image.
        fills: an array of fills' points.
    # Returns
        an image.
    """
    # Generate color for each fill randomly.
    colors = np.random.randint(0, 255, (np.max(fillmap) + 1, 3))
    # Id of line is 0, and its color is black.
    colors[0] = [0, 0, 0]

    return colors[fillmap]


def get_bounding_rect(points):
    """Get a bounding rect of points.

    # Arguments
        points: array of points.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = np.min(points[1]), np.min(points[0]), np.max(points[1]), np.max(points[0])
    return x1, y1, x2, y2


def get_border_bounding_rect(h, w, p1, p2, r):
    """Get a valid bounding rect in the image with border of specific size.

    # Arguments
        h: image max height.
        w: image max width.
        p1: start point of rect.
        p2: end point of rect.
        r: border radius.
    # Returns
        rect coord
    """
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]

    x1 = x1 - r if 0 < x1 - r else 0
    y1 = y1 - r if 0 < y1 - r else 0
    x2 = x2 + r + 1 if x2 + r + 1 < w else w
    y2 = y2 + r + 1 if y2 + r + 1 < h else h

    return x1, y1, x2, y2


def get_border_point(points, rect, max_height, max_width):
    """Get border points of a fill area

    # Arguments
        points: points of fill .
        rect: bounding rect of fill.
        max_height: image max height.
        max_width: image max width.
    # Returns
        points , convex shape of points
    """
    # Get a local bounding rect.
    border_rect = get_border_bounding_rect(max_height, max_width, rect[:2], rect[2:], 2)

    # Get fill in rect.
    fill = np.zeros((border_rect[3] - border_rect[1], border_rect[2] - border_rect[0]), np.uint8)
    # Move points to the rect.
    fill[(points[0] - border_rect[1], points[1] - border_rect[0])] = 255

    # Get shape.
    _, contours, _ = cv2.findContours(fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approx_shape = cv2.approxPolyDP(contours[0], 0.02 * cv2.arcLength(contours[0], True), True)

    # Get border pixel.
    # Structuring element in cross shape is used instead of box to get 4-connected border.
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    border_pixel_mask = cv2.morphologyEx(fill, cv2.MORPH_DILATE, cross, anchor=(-1, -1), iterations=1) - fill
    border_pixel_points = np.where(border_pixel_mask == 255)

    # Transform points back to fillmap.
    border_pixel_points = (border_pixel_points[0] + border_rect[1], border_pixel_points[1] + border_rect[0])

    return border_pixel_points, approx_shape


def merge_fill(fillmap, max_iter=10):
    """Merge fill areas.

    # Arguments
        fillmap: an image.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for i in range(max_iter):
        print('merge ' + str(i + 1))

        result[np.where(fillmap == 0)] = 0

        fill_id = np.unique(result.flatten())
        fills = []

        for j in fill_id:
            point = np.where(result == j)

            fills.append({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point)
            })

        for j, f in enumerate(fills):
            # ignore lines
            if f['id'] == 0:
                continue

            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)

            ids = pixel_ids[np.nonzero(pixel_ids)]
            new_id = f['id']
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:
                    new_id = 0
            else:
                # region id may be set to region with largest contact
                new_id = ids[0]

            # a point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id

            #
            if len(approx_shape) in [2, 3, 4, 5] and f['area'] < 500:
                result[f['point']] = new_id

            if f['area'] < 250 and len(ids) == 1:
                result[f['point']] = new_id

            if f['area'] < 50:
                result[f['point']] = new_id

        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result


def my_merge_fill(in_image, fillmap, max_iter=10):
    """Merge fill areas.

    # Arguments
        in_image: color or gray image
        fillmap: label map.
        max_iter: max iteration number.
    # Returns
        an image.
    """
    max_height, max_width = fillmap.shape[:2]
    result = fillmap.copy()

    for i in range(max_iter):
        print('merge ' + str(i + 1))

        result[np.where(fillmap == 0)] = 0

        fill_id = np.unique(result.flatten())
        fills = dict()

        for j in fill_id:
            mask = (result == j)
            point = np.where(mask)
            
#             fills.append({
#                 'id': j,
#                 'point': point,
#                 'area': len(point[0]),
#                 'rect': get_bounding_rect(point),
#             })
            fills[j] = ({
                'id': j,
                'point': point,
                'area': len(point[0]),
                'rect': get_bounding_rect(point),
                'color_hist': calculate_color_hist(mask, in_image)
            })
            
        for j in fills.keys():
            f = fills[j]
#             print('-- key: ', j)
                
            # ignore lines
            if f['id'] == 0:
                continue

            if f['area'] >= 500:
                continue
                
            border_points, approx_shape = get_border_point(f['point'], f['rect'], max_height, max_width)
            border_pixels = result[border_points]
            pixel_ids, counts = np.unique(border_pixels, return_counts=True)
            
            ids = pixel_ids[np.nonzero(pixel_ids)]
            if pixel_ids[0] == 0:
                counts_no_border = counts[1:]
            else:
                counts_no_border = counts
            countSum = np.sum(counts_no_border)
            
            new_id = f['id']
            
            max_score = -1
            count_at_max = -1
            score = -1
            if len(ids) == 0:
                # points with lines around color change to line color
                # regions surrounded by line remain the same
                if f['area'] < 5:
                    new_id = 0
            else:
                # # region id may be set to region with largest contact
                new_id = ids[0]
                
#                 max_count = -1
#                 for order_id, nbor_id in enumerate(ids):
#                     if counts_no_border[order_id] > max_count:
#                         max_count = counts_no_border[order_id]
#                         new_id = nbor_id


                # similariest region is selected
                curHist = f['color_hist']
#                 print(curHist)

                for order_id, nbor_id in enumerate(ids):
                    nbor_hist = fills[nbor_id]['color_hist']
                    score = np.sum(np.min([curHist[np.newaxis,:], nbor_hist[np.newaxis,:]], axis=0))
#                     print('nbor_id: ', nbor_id)
#                     print(nbor_hist)
#                     print('score: ', score)
                    
                    if score > max_score:
                        max_score = score
                        count_at_max = counts_no_border[order_id]
                        new_id = nbor_id
                    
            # a point
            if len(approx_shape) == 1 or f['area'] == 1:
                result[f['point']] = new_id
                continue

            #
#             if len(approx_shape) <= 5 and f['area'] < 500:
#                 result[f['point']] = new_id
            
            # 300 is just fine
            if f['area'] < 300 and len(ids) == 1:
                result[f['point']] = new_id
                continue
                
            if f['area'] < 500 and len(ids) == 1 and count_at_max > 2:
                result[f['point']] = new_id
                continue
        
            if f['area'] < 50:
                result[f['point']] = new_id
                continue
    
            # not good at frame #1 dress
            if score > 0.2 and f['area'] < 150 and count_at_max > 1:
                result[f['point']] = new_id
            
            if score > 0.25 and f['area'] < 250 and count_at_max > 8:
                result[f['point']] = new_id

        if len(fill_id) == len(np.unique(result.flatten())):
            break

    return result