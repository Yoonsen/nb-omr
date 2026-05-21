import itertools

import numpy as np
from cv2 import dilate, erode
from numpy import floor, random
from PIL import Image, ImageOps
from skimage import transform as stf
from torchvision.transforms.functional import adjust_brightness, adjust_contrast


class BrighnessAjust:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return adjust_brightness(x, self.factor)


class ContrastAdjust:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        return adjust_contrast(x, self.factor)


class SignFlipping:
    def __call__(self, x):
        return ImageOps.invert(x)


class DPIAdjusting:
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, x):
        w, h = x.size
        return x.resize((int(np.ceil(w * self.factor)), int(np.ceil(h * self.factor))), Image.BILINEAR)


class Dilation:
    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(dilate(np.array(x), self.kernel, iterations=self.iterations))


class Erosion:
    def __init__(self, kernel, iterations):
        self.kernel = np.ones(kernel, np.uint8)
        self.iterations = iterations

    def __call__(self, x):
        return Image.fromarray(erode(np.array(x), self.kernel, iterations=self.iterations))


class ElasticDistortion:
    def __init__(self, grid, magnitude, min_sep):
        self.grid_width, self.grid_height = grid
        self.xmagnitude, self.ymagnitude = magnitude
        self.min_h_sep, self.min_v_sep = min_sep

    def __call__(self, x):
        w, h = x.size
        horizontal_tiles = self.grid_width
        vertical_tiles = self.grid_height

        width_of_square = int(floor(w / float(horizontal_tiles)))
        height_of_square = int(floor(h / float(vertical_tiles)))
        width_of_last_square = w - (width_of_square * (horizontal_tiles - 1))
        height_of_last_square = h - (height_of_square * (vertical_tiles - 1))

        dimensions = []
        shift = [[(0, 0) for _ in range(horizontal_tiles)] for _ in range(vertical_tiles)]

        for vertical_tile in range(vertical_tiles):
            for horizontal_tile in range(horizontal_tiles):
                if vertical_tile == vertical_tiles - 1 and horizontal_tile == horizontal_tiles - 1:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif vertical_tile == vertical_tiles - 1:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_last_square + (height_of_square * vertical_tile),
                        ]
                    )
                elif horizontal_tile == horizontal_tiles - 1:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_last_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )
                else:
                    dimensions.append(
                        [
                            horizontal_tile * width_of_square,
                            vertical_tile * height_of_square,
                            width_of_square + (horizontal_tile * width_of_square),
                            height_of_square + (height_of_square * vertical_tile),
                        ]
                    )

                sm_h = (
                    min(
                        self.xmagnitude,
                        width_of_square - (self.min_h_sep + shift[vertical_tile][horizontal_tile - 1][0]),
                    )
                    if horizontal_tile > 0
                    else self.xmagnitude
                )
                sm_v = (
                    min(
                        self.ymagnitude,
                        height_of_square - (self.min_v_sep + shift[vertical_tile - 1][horizontal_tile][1]),
                    )
                    if vertical_tile > 0
                    else self.ymagnitude
                )

                dx = random.randint(-sm_h, self.xmagnitude)
                dy = random.randint(-sm_v, self.ymagnitude)
                shift[vertical_tile][horizontal_tile] = (dx, dy)

        shift = list(itertools.chain.from_iterable(shift))
        last_column = [(horizontal_tiles - 1) + horizontal_tiles * i for i in range(vertical_tiles)]
        last_row = range((horizontal_tiles * vertical_tiles) - horizontal_tiles, horizontal_tiles * vertical_tiles)

        polygons = []
        for x1, y1, x2, y2 in dimensions:
            polygons.append([x1, y1, x1, y2, x2, y2, x2, y1])

        polygon_indices = []
        for i in range((vertical_tiles * horizontal_tiles) - 1):
            if i not in last_row and i not in last_column:
                polygon_indices.append([i, i + 1, i + horizontal_tiles, i + 1 + horizontal_tiles])

        for idx, (a, b, c, d) in enumerate(polygon_indices):
            dx = shift[idx][0]
            dy = shift[idx][1]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[a]
            polygons[a] = [x1, y1, x2, y2, x3 + dx, y3 + dy, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[b]
            polygons[b] = [x1, y1, x2 + dx, y2 + dy, x3, y3, x4, y4]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[c]
            polygons[c] = [x1, y1, x2, y2, x3, y3, x4 + dx, y4 + dy]

            x1, y1, x2, y2, x3, y3, x4, y4 = polygons[d]
            polygons[d] = [x1 + dx, y1 + dy, x2, y2, x3, y3, x4, y4]

        generated_mesh = []
        for i in range(len(dimensions)):
            generated_mesh.append([dimensions[i], polygons[i]])

        return x.transform(x.size, Image.MESH, generated_mesh, resample=Image.BICUBIC)


class RandomTransform:
    def __init__(self, val):
        self.val = val

    def __call__(self, x):
        w, h = x.size
        dw, dh = (self.val, 0) if random.randint(0, 2) == 0 else (0, self.val)

        def rd(d):
            return random.uniform(-d, d)

        def fd(d):
            return random.uniform(-dw, d)

        tl_top = rd(dh)
        tl_left = fd(dw)
        bl_bottom = rd(dh)
        bl_left = fd(dw)
        tr_top = rd(dh)
        tr_right = fd(min(w * 3 / 4 - tl_left, dw))
        br_bottom = rd(dh)
        br_right = fd(min(w * 3 / 4 - bl_left, dw))

        tform = stf.ProjectiveTransform()
        tform.estimate(
            np.array(
                (
                    (tl_left, tl_top),
                    (bl_left, h - bl_bottom),
                    (w - br_right, h - br_bottom),
                    (w - tr_right, tr_top),
                )
            ),
            np.array(([0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0])),
        )

        corners = np.array([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]])
        corners = tform.inverse(corners)
        minc = corners[:, 0].min()
        minr = corners[:, 1].min()
        maxc = corners[:, 0].max()
        maxr = corners[:, 1].max()
        out_rows = maxr - minr + 1
        out_cols = maxc - minc + 1
        output_shape = np.around((out_rows, out_cols))

        translation = (minc, minr)
        tform4 = stf.SimilarityTransform(translation=translation)
        tform = tform4 + tform
        tform.params /= tform.params[2, 2]

        x = stf.warp(np.array(x), tform, output_shape=output_shape, cval=255, preserve_range=True)
        x = stf.resize(x, (h, w), preserve_range=True).astype(np.uint8)
        return Image.fromarray(x)
