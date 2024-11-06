import plistlib, cv2
import numpy as np
from pydicom import dcmread
from pydicom.pixel_data_handlers import apply_voi_lut
from skimage.draw import polygon


def load_inbreast_mask(mask_path, imshape=(4084, 3328), n_class=1):
    """
    This function loads a osirix xml region as a binary numpy array for INBREAST
    dataset
    @mask_path : Path to the xml file
    @imshape : The shape of the image as an array e.g. [4084, 3328]
    return: numpy array where each mass has a different number id.
    """

    def draw_mask(mask, points, imshape, pixel_value):
        if len(points) <= 2:
            for point in points:
                mask[int(point[0]), int(point[1])] = pixel_value
        else:
            x, y = zip(*points)
            x, y = np.array(x), np.array(y)
            poly_x, poly_y = polygon(x, y, shape=imshape)
            mask[poly_x, poly_y] = pixel_value

    def load_point(point_string):
        x, y = tuple([float(num) for num in point_string.strip("()").split(",")])
        return y, x

    i = 0
    mask = np.zeros(imshape)
    with open(mask_path, "rb") as mask_file:
        plist_dict = plistlib.load(mask_file, fmt=plistlib.FMT_XML)["Images"][0]
        numRois = plist_dict["NumberOfROIs"]
        rois = plist_dict["ROIs"]
        assert len(rois) == numRois
        for roi in rois:
            numPoints = roi["NumberOfPoints"]
            i += 1
            points = roi["Point_px"]
            assert numPoints == len(points)
            points = [load_point(point) for point in points]
            if n_class == 1:
                if roi["Name"] == "Mass":
                    draw_mask(mask, points, imshape, 1)
            else:
                if roi["Name"] in ["Mass", "Calcification"]:
                    pixel_value = 1 if roi["Name"] == "Mass" else 2
                    draw_mask(mask, points, imshape, pixel_value)
    if not np.all(mask == 0):
        return mask
    else:
        return None


def load_dicom_image(path):
    ds = dcmread(path)
    img2d = ds.pixel_array
    img2d = apply_voi_lut(img2d, ds)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        img2d = np.amax(img2d) - img2d
    img2d = cv2.normalize(
        img2d, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    ).astype(np.uint8)
    return img2d
