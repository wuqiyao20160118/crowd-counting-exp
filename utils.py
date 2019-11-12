import numpy as np
import os
from xml.etree.ElementTree import ElementTree
from glob import glob
import cv2
from copy import deepcopy
import warnings


def rect_dist(I, J):
    if len(I.shape) == 1:
        I = I[np.newaxis, :]
        J = J[np.newaxis, :]

    # area of boxes
    aI = (I[:, 2] - I[:, 0]) * (I[:, 3] - I[:, 1])
    aJ = (J[:, 2] - J[:, 0]) * (J[:, 3] - J[:, 1])

    x1 = np.maximum(I[:, 0], J[:, 0])
    y1 = np.maximum(I[:, 1], J[:, 1])
    x2 = np.minimum(I[:, 2], J[:, 2])
    y2 = np.minimum(I[:, 3], J[:, 3])

    aIJ = (x2-x1) * (y2-y1) * (np.logical_and(x2 > x1, y2 > y1))

    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            iou = aIJ / (aI + aJ - aIJ)
        except (RuntimeWarning, Exception):
            iou = np.zeros(aIJ.shape)

    # try:
    #     iou = aIJ / (aI + aJ - aIJ)
    # except (RuntimeWarning, Exception):
    #     print("Error in rect dist!")
    #     print(I)
    #     print(J)
    #     print(aIJ)
    #     exit(0)

    # set NaN, inf, and -inf to 0
    iou[np.isnan(iou)] = 0
    iou[np.isinf(iou)] = 0

    dist = np.maximum(np.zeros(iou.shape), np.minimum(np.ones(iou.shape), 1 - iou))

    return dist


def crop_image(img, img2, bboxes, bboxes2, input_size=(471, 471)):
    """
    Crop a 471x471 patch from the image, taking care for smaller images.
    bboxes is the np.array of all bounding boxes [x1, y1, x2, y2]
    """
    # randomly pick a cropping window for the image
    # We keep the second arg to randint at least 1 since randint is [low, high)
    crop_x1 = np.random.randint(0, np.max([1, (img.shape[1] - input_size[1] + 1)]))
    crop_y1 = np.random.randint(0, np.max([1, (img.shape[0] - input_size[0] + 1)]))
    crop_x2 = min(img.shape[1], crop_x1 + input_size[1])
    crop_y2 = min(img.shape[0], crop_y1 + input_size[0])
    crop_h = crop_y2 - crop_y1
    crop_w = crop_x2 - crop_x1

    # place the cropped image in a random location in a `input_size` image
    paste_box = [0, 0, 0, 0]  # x1, y1, x2, y2
    paste_box[0] = np.random.randint(0, input_size[1] - crop_w + 1)
    paste_box[1] = np.random.randint(0, input_size[0] - crop_h + 1)
    paste_box[2] = paste_box[0] + crop_w
    paste_box[3] = paste_box[1] + crop_h

    # set this to average image colors
    # this will later be subtracted in mean image subtraction
    img_buf = np.zeros((input_size + (3,)))
    img_buf_2 = np.zeros((input_size + (3,)))

    # add the average image so it gets subtracted later.
    for i, c in enumerate([0.485, 0.456, 0.406]):
        # img is a int8 array, so we need to scale the values accordingly
        img_buf[:, :, i] = int(c * 255)
        img_buf_2[:, :, i] = int(c * 255)

    img_buf[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2], :] = img[crop_y1:crop_y2, crop_x1:crop_x2, :]
    img_buf_2[paste_box[1]:paste_box[3], paste_box[0]:paste_box[2], :] = img2[crop_y1:crop_y2, crop_x1:crop_x2, :]

    if bboxes.shape[0] > 0:
        # check if overlap is above negative threshold
        tbox = deepcopy(bboxes)
        tbox[:, 0] = np.maximum(tbox[:, 0], crop_x1)
        tbox[:, 1] = np.maximum(tbox[:, 1], crop_y1)
        tbox[:, 2] = np.minimum(tbox[:, 2], crop_x2)
        tbox[:, 3] = np.minimum(tbox[:, 3], crop_y2)

        overlap = 1 - rect_dist(tbox, bboxes)

        # adjust the bounding boxes - first for crop and then for random placement
        bboxes[:, 0] = bboxes[:, 0] - crop_x1 + paste_box[0]
        bboxes[:, 1] = bboxes[:, 1] - crop_y1 + paste_box[1]
        bboxes[:, 2] = bboxes[:, 2] - crop_x1 + paste_box[0]
        bboxes[:, 3] = bboxes[:, 3] - crop_y1 + paste_box[1]

        # correct for bbox to be within image border
        bboxes[:, 0] = np.minimum(input_size[1], np.maximum(0, bboxes[:, 0]))
        bboxes[:, 1] = np.minimum(input_size[0], np.maximum(0, bboxes[:, 1]))
        bboxes[:, 2] = np.minimum(input_size[1], np.maximum(1, bboxes[:, 2]))
        bboxes[:, 3] = np.minimum(input_size[0], np.maximum(1, bboxes[:, 3]))

        # check to see if the adjusted bounding box is invalid
        invalid = np.logical_or(np.logical_or(bboxes[:, 2] <= bboxes[:, 0], bboxes[:, 3] <= bboxes[:, 1]),
                                overlap < 0.3)

        # remove invalid bounding boxes
        ind = np.where(invalid)
        bboxes = np.delete(bboxes, ind, 0)

    if bboxes2.shape[0] > 0:
        # check if overlap is above negative threshold
        tbox2 = deepcopy(bboxes2)
        tbox2[:, 0] = np.maximum(tbox2[:, 0], crop_x1)
        tbox2[:, 1] = np.maximum(tbox2[:, 1], crop_y1)
        tbox2[:, 2] = np.minimum(tbox2[:, 2], crop_x2)
        tbox2[:, 3] = np.minimum(tbox2[:, 3], crop_y2)

        overlap2 = 1 - rect_dist(tbox2, bboxes2)

        # adjust the bounding boxes - first for crop and then for random placement
        bboxes2[:, 0] = bboxes2[:, 0] - crop_x1 + paste_box[0]
        bboxes2[:, 1] = bboxes2[:, 1] - crop_y1 + paste_box[1]
        bboxes2[:, 2] = bboxes2[:, 2] - crop_x1 + paste_box[0]
        bboxes2[:, 3] = bboxes2[:, 3] - crop_y1 + paste_box[1]

        # correct for bbox to be within image border
        bboxes2[:, 0] = np.minimum(input_size[1], np.maximum(0, bboxes2[:, 0]))
        bboxes2[:, 1] = np.minimum(input_size[0], np.maximum(0, bboxes2[:, 1]))
        bboxes2[:, 2] = np.minimum(input_size[1], np.maximum(1, bboxes2[:, 2]))
        bboxes2[:, 3] = np.minimum(input_size[0], np.maximum(1, bboxes2[:, 3]))

        # check to see if the adjusted bounding box is invalid
        invalid2 = np.logical_or(np.logical_or(bboxes2[:, 2] <= bboxes2[:, 0], bboxes2[:, 3] <= bboxes2[:, 1]),
                                 overlap2 < 0.3)

        # remove invalid bounding boxes
        ind2 = np.where(invalid2)
        bboxes2 = np.delete(bboxes2, ind2, 0)

    return img_buf, img_buf_2, bboxes, bboxes2, paste_box


def read_xml(in_path):
    """
    parse xml files
    in_path: xml path
    return: ElementTree
    """
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def get_frame_annotation(data_dir, head=True):
    xml = data_dir[:-4] + '.xml'
    frame_dict = {}
    tree = read_xml(xml)
    root = tree.getroot()
    temp = []
    ori_temp = []
    filename = root.find('filename').text
    if ".png" not in filename:
        filename += ".png"
    img_dir = os.path.join(data_dir, filename)
    frame_dict['img'] = img_dir
    for object in root.iter('object'):
        anno = object.find('bndbox')
        annotation = [float(anno.find('xmin').text), float(anno.find('xmax').text), \
                     float(anno.find('ymin').text), float(anno.find('ymax').text)]
        if head:
            ori_temp.append(list(deepcopy(annotation)))
            annotation[0], annotation[1] = (annotation[0] + annotation[1]) / 2, (annotation[2] + annotation[3]) / 2
            temp.append(list(annotation)[:2])
        else:
            ori_temp.append(list(annotation))
            temp.append(list(annotation))
    frame_dict['annotation'] = np.array(temp)
    frame_dict['ori_annotation'] = np.array(ori_temp)
    return frame_dict


def _gaussian_kernel(sigma=1.0, kernel_size=None):
    '''
    Returns gaussian kernel if sigma > 0.0, otherwise dot kernel.
    '''
    if sigma <= 0.0:
        return np.array([[0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0],
                         [0.0, 0.0, 0.0]], dtype=np.float32)
    if kernel_size is None:
        kernel_size = int(3.0 * sigma)
    if kernel_size % 2 == 0:
        kernel_size += 1
        print('In data_reader.gaussian_kernel: Kernel size even; increased by 1.')
    if kernel_size < 3:
        kernel_size = 3
        print('In data_reader.gaussian_kernel: Kernel size less than 3; set as 3.')
    tmp = np.arange((-kernel_size // 2) + 1.0, (kernel_size // 2) + 1.0)
    xx, yy = np.meshgrid(tmp, tmp)
    kernel = np.exp(-((xx ** 2) + (yy ** 2)) / (2.0 * (sigma ** 2)))
    kernel_sum = np.sum(kernel)
    assert (kernel_sum > 1e-3)
    return kernel / kernel_sum


def _create_heatmap(image_shape, heatmap_shape, annotation_points, kernel):
    """
    Creates density map.
    annotation_points : ndarray Nx2,
                        annotation_points[:, 0] -> x coordinate
                        annotation_points[:, 1] -> y coordinate
    """
    assert (kernel.shape[0] == kernel.shape[1] and kernel.shape[0] % 2
            and kernel.shape[0] > 1)
    indices = (annotation_points[:, 0] < image_shape[1]) & \
              (annotation_points[:, 0] >= 0) & \
              (annotation_points[:, 1] < image_shape[0]) & \
              (annotation_points[:, 1] >= 0)
    annot_error_count = len(annotation_points)
    annotation_points = annotation_points[indices, :]

    hmap_height, hmap_width = heatmap_shape
    annotation_points[:, 0] *= (float(heatmap_shape[1]) / image_shape[1])
    annotation_points[:, 1] *= (float(heatmap_shape[0]) / image_shape[0])
    annotation_points = annotation_points.astype(np.int32)
    annot_error_count -= np.sum(indices)
    if annot_error_count:
        print('In data_reader.create_heatmap: Error in annotations; %d point(s) skipped.' % annot_error_count)
    indices = (annotation_points[:, 0] >= heatmap_shape[1]) & \
              (annotation_points[:, 0] < 0) & \
              (annotation_points[:, 1] >= heatmap_shape[0]) & \
              (annotation_points[:, 1] < 0)
    assert(np.sum(indices) == 0)

    prediction_map = np.zeros(heatmap_shape, dtype = np.float32)
    kernel_half_size = kernel.shape[0] // 2
    kernel_copy = np.empty_like(kernel)

    for x, y in annotation_points:
        y_start = y - kernel_half_size
        y_end = y_start + kernel.shape[0]
        x_start = x - kernel_half_size
        x_end = x_start + kernel.shape[1]
        kernel_copy[:] = kernel[:]
        kernel_tmp = kernel_copy
        if y_start < 0:
            i = -y_start
            kernel_tmp[i: 2 * i, :] += kernel_tmp[i - 1:: -1, :]
            kernel_tmp = kernel_tmp[i:, :]
            y_start = 0
        if x_start < 0:
            i = -x_start
            kernel_tmp[:, i: 2 * i] += kernel_tmp[:, i - 1:: -1]
            kernel_tmp = kernel_tmp[:, i:]
            x_start = 0
        if y_end > hmap_height:
            i = (hmap_height - y - 1) - kernel_half_size
            kernel_tmp[2 * i: i, :] += kernel_tmp[-1: i - 1: -1, :]
            kernel_tmp = kernel_tmp[: i, :]
            y_end = hmap_height
        if x_end > hmap_width:
            i = (hmap_width - x - 1) - kernel_half_size
            kernel_tmp[:, 2 * i: i] += kernel_tmp[:, -1: i - 1: -1]
            kernel_tmp = kernel_tmp[:, : i]
            x_end = hmap_width
        prediction_map[y_start: y_end, x_start: x_end] += kernel_tmp
    return prediction_map


def round_up(value):
    return round(value + 1e-6 + 1000) - 1000


def save_density_map(input_img, density_map, output_dir, fname='results.png'):
    if np.max(density_map) > 0:
        density_map = 255*density_map/np.max(density_map)
    if len(density_map.shape) != 2:
        #density_map = density_map[0][0]
        density_map = density_map[0]
    if density_map.shape[1] != input_img.shape[1]:
        density_map = cv2.resize(density_map, (input_img.shape[2],input_img.shape[1]))
    cv2.imwrite(os.path.join(output_dir, fname), density_map)


if __name__ == "__main__":
    root_dir = "./data/scvd/train"
    datas = os.listdir(root_dir)
    test_data_dir = os.path.join(root_dir, datas[0])
    xml_list = glob(test_data_dir + '/*.xml')
    xml_test = xml_list[0]
    tr = read_xml(xml_test)
    root = tr.getroot()
    object = root.find('object')
    anno = object.find('bndbox')
    annos = int(anno.find('xmin').text), int(anno.find('xmax').text), \
                             int(anno.find('ymin').text), int(anno.find('ymax').text)
    print(annos)
