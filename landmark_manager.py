##  coding: utf-8

import os
import sys
import csv
import json

import cv2
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


def load_pts(path):
    """Landmark commonly used file like following format:

    version:2 
    n_points:5 
    {
    1.0 2.0
    3.0 4.0
    5.0 6.0
    7.0 8.0
    9.0 1.0
    }
    """
    if not os.path.exists(path):
        # raise IOError("file not exist")
        return None

    file_obj = open(path, 'r')
    ret = {}
    for line in file_obj:
        line = line.strip()
        if line == '{':
            break
        data = line.split(':')
        ret[data[0]] = int(data[1])
    ret['points'] = []
    for line in file_obj:
        line = line.strip()
        if line == '}':
            break
        # data = list(map(float, line.split()))
        data = [c for c in map(float, line.split())]
        ret['points'].append(data)
    file_obj.close()
    return ret


def dump_pts(obj, path):
    """Landmark commonly used format(pairs to load_pts)"""
    if not obj:
        return False

    file_obj = open(path, 'w')
    for key in ['version', 'n_points']:
        line = key + ': ' + str(obj[key]) + '\n'
        file_obj.write(line)
    file_obj.write('{\n')
    for point in obj['points']:
        line = str(point[0]) + ' ' + str(point[1]) + '\n'
        file_obj.write(line)
    file_obj.write('}\n')
    file_obj.close()
    return True


def load_3d_landmark(path):
    if not os.path.exists(path):
        # raise IOError("file not exist")
        return None
    with open(path) as f:
        reader = csv.reader(f)
        _ret = list(reader)
        _ret = [float(item[0]) for item in _ret]
        ret = []
        for i in range(len(_ret)//3):
            ret.append([_ret[i], _ret[i+68], _ret[i+136]])
    return ret


def dump_list_to_csv(obj, path):
    if not obj:
        return False
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(obj)
    return True


def load_bbox(path):
    """bbox[xmin, ymin, xmax, ymax]"""
    if not os.path.exists(path):
        # raise IOError("file not exist")
        return None
    with open(path, 'r') as f:
        line = f.readline().strip()
        bbox = [c for c in map(float, line.split())]
        bbox[2] = bbox[0] + bbox[2]
        bbox[3] = bbox[1] + bbox[3]
    return bbox


def dump_bbox(bbox, path):
    """bbox[xmin, ymin, xmax, ymax]"""
    if not bbox:
        return False
    bbox[2] = bbox[2] - bbox[0]
    bbox[3] = bbox[3] - bbox[1]
    file_obj = open(path, 'w')
    line = " ".join(["%d" % int(v) for v in bbox])
    file_obj.write(line+'\n')
    file_obj.close()
    return True


def load_openface_csv(path):
    if not os.path.exists(path):
        # raise IOError("file not exist")
        return None
    with open(path) as f:
        reader = csv.reader(f)
        ret = list(reader)
        ret[1] = list(map(float, ret[1]))
    return ret


def parse_openface_csv(obj):
    if not obj:
        return False
    ret = {}
    index_start = 13
    index_length = 56
    ret['eye_lmk'] = []
    for i in range(index_length):
        ret['eye_lmk'].append([obj[1][index_start+i],
                               obj[1][index_start+index_length+i]])
    # p_scale = obj[1][639]
    p_scale = 1.0
    index_start = 300-1
    index_length = 68
    ret['face_lmk'] = []
    for i in range(index_length):
        ret['face_lmk'].append([obj[1][index_start+i]*p_scale,
                                obj[1][index_start+index_length+i]*p_scale])
    ret['gaze_angle_x'] = obj[1][11]
    ret['gaze_angle_y'] = obj[1][12]
    return ret


def parse_openface_csv_gaze(obj):
    if not obj:
        return False
    ret = {}
    ret['gaze_x'] = []
    ret['gaze_y'] = []
    index_start = 1
    index_length = len(obj)
    for i in range(1,index_length):
        ret['gaze_x'].append(obj[i][11])
        ret['gaze_y'].append(obj[i][12])
    return ret


def draw_points(img, path_dst, points, bboxs=None, color=None):
    """draw face bbox in single image

    Args:
        img: numpy array, bgr order of shape (1, 3, n, m) input image, provide by cv2.imread()
    """
    if img is None:
        return None

    if color is None and len(img.shape) == 1:
        color = (128)
    elif color is None and len(img.shape) == 3:
        color = (128, 128, 128)

    _img_draw = img.copy()
    if bboxs:
        for bbox in bboxs:
            cv2.rectangle(_img_draw, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color)

    for p in points:
        for i in range(len(p)):
            cv2.circle(_img_draw, (int(p[i][0]), int(p[i][1])), 1, color, 2)

    cv2.imwrite(path_dst, _img_draw)


def draw_points_with_index(img, path_dst, points, bboxs=None, color=None):
    """draw face bbox in single image

    Args:
        img: numpy array, bgr order of shape (1, 3, n, m) input image, provide by cv2.imread()
    """
    if img is None:
        return None

    if color is None and len(img.shape) == 1:
        color = (128)
    elif color is None and len(img.shape) == 3:
        color = (128, 128, 128)

    _img_draw = img.copy()
    if bboxs:
        for bbox in bboxs:
            cv2.rectangle(_img_draw, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for index, p in enumerate(points):
        for i in range(len(p)):
            imgzi = cv2.putText(_img_draw, str(
                i), (int(p[i][0]), int(p[i][1])), font, 0.4, color, 1)
            cv2.circle(_img_draw, (int(p[i][0]), int(p[i][1])), 1, color, 2)

    cv2.imwrite(path_dst, _img_draw)


def draw_openface_eye_points_with_index_and_line(img, path_dst, points, bboxs=None, color=None):
    """draw face bbox in single image

    Args:
        img: numpy array, bgr order of shape (1, 3, n, m) input image, provide by cv2.imread()
    """
    if img is None:
        return None

    if color is None and len(img.shape) == 1:
        color = (128)
    elif color is None and len(img.shape) == 3:
        color = (128, 128, 128)

    _img_draw = img.copy()
    if bboxs:
        for bbox in bboxs:
            cv2.rectangle(_img_draw, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for p in points:
        for i in range(len(p)):
            next_point = i + 1
            if i == 7:
                next_point = 0
            if i == 19:
                next_point = 8
            if i == 27:
                next_point = 20

            if i == 7 + 28:
                next_point = 0 + 28
            if i == 19 + 28:
                next_point = 8 + 28
            if i == 27 + 28:
                next_point = 20 + 28
            featurePoint = (int(p[i][0]), int(p[i][1]))
            cv2.putText(_img_draw, str(
                i), (int(p[i][0]), int(p[i][1])), font, 0.4, color, 1)
            cv2.circle(_img_draw, (int(p[i][0]), int(p[i][1])), 1, color, 2)
            nextFeaturePoint = (int(p[next_point][0]), int(p[next_point][1]))
            if ((i < 28 and (i < 8 or i > 19)) or (i >= 28 and (i < 8 + 28 or i > 19 + 28))):
                cv2.line(_img_draw, featurePoint,
                         nextFeaturePoint, color, 2)
            else:
                cv2.line(_img_draw, featurePoint, nextFeaturePoint,
                         color, 2)
    cv2.imwrite(path_dst, _img_draw)


def draw_label_tool_eye_points_with_index_and_line(img, path_dst, points, bboxs=None, color=None):
    """draw face bbox in single image

    Args:
        img: numpy array, bgr order of shape (1, 3, n, m) input image, provide by cv2.imread()
    """
    if img is None:
        return None

    if color is None and len(img.shape) == 1:
        color = (128)
    elif color is None and len(img.shape) == 3:
        color = (128, 128, 128)

    _img_draw = img.copy()
    if bboxs:
        for bbox in bboxs:
            cv2.rectangle(_img_draw, (int(bbox[0]), int(bbox[1])),
                          (int(bbox[2]), int(bbox[3])), color)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for p in points:
        for i in range(len(p)):
            next_point = i + 1
            if i == 11:
                next_point = 0
            if i == 19:
                next_point = 12
            if i == 23:
                next_point = 20

            if i == 11 + 25:
                next_point = 0 + 25
            if i == 19 + 25:
                next_point = 12 + 25
            if i == 23 + 25:
                next_point = 20 + 25
            if i == 24 or i == 49 or i >= 50:
                next_point = i
            featurePoint = (int(p[i][0]), int(p[i][1]))
            cv2.putText(_img_draw, str(
                i), (int(p[i][0]), int(p[i][1])), font, 0.4, color, 1)
            cv2.circle(_img_draw, (int(p[i][0]), int(p[i][1])), 1, color, 2)
            nextFeaturePoint = (int(p[next_point][0]), int(p[next_point][1]))
            if ((i < 25 and (i < 8 or i > 19)) or (i >= 25 and (i < 8 + 25 or i > 19 + 25))):
                cv2.line(_img_draw, featurePoint,
                         nextFeaturePoint, color, 2)
            else:
                cv2.line(_img_draw, featurePoint, nextFeaturePoint,
                         color, 2)
    cv2.imwrite(path_dst, _img_draw)


def get_bbox_from_landmark68(point):
    """仅支持单个人脸的特征点生成人脸框"""
    xmin, ymin = point[0]
    xmax, ymax = point[0]
    for p in point:
        xmin = min(xmin, p[0])
        ymin = min(ymin, p[1])
        xmax = max(xmax, p[0])
        ymax = max(ymax, p[1])
    scale = 1.05
    xoffset = xmin - xmin/scale
    yoffset = ymin - ymin/(scale+0.1)
    bbox = [xmin - xoffset, ymin-yoffset, xmax+xoffset, ymax+yoffset/2]
    return bbox


def convert_openface_eye_points_to_label_tool_pts(ldmks):
    if not ldmks:
        return False
    ret = {}
    ret['version'] = 1
    ret['n_points'] = 56
    ret['points'] = []

    # 左眼
    eye_lid_len = 12
    eye_lid_idx_in_openface = 8
    for i in range(eye_lid_len):
        ret['points'].append(ldmks['eye_lmk'][eye_lid_idx_in_openface+i])
    eye_ball_len = 8
    eye_ball_idx_in_openface = 0
    for i in range(eye_ball_len):
        ret['points'].append(ldmks['eye_lmk'][eye_ball_idx_in_openface+i])
    eye_pupil_len = 4
    eye_pupil_idx_in_openface = 27
    for i in range(eye_pupil_len):
        ret['points'].append(ldmks['eye_lmk'][eye_pupil_idx_in_openface-2*i])
    # eye_center_len = 1
    # eye_center_idx_in_openface = -1
    x_sum, y_sum = 0, 0
    for i in range(8):
        x_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][0]
        y_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][1]
    ret['points'].append([x_sum // 8, y_sum//8])

    # 右眼
    eye_lid_len = 12
    eye_lid_idx_in_openface = 36
    for i in range(eye_lid_len):
        ret['points'].append(ldmks['eye_lmk'][eye_lid_idx_in_openface+i])
    eye_ball_len = 8
    eye_ball_idx_in_openface = 28
    for i in range(eye_ball_len):
        ret['points'].append(ldmks['eye_lmk'][eye_ball_idx_in_openface+i])
    eye_pupil_len = 4
    eye_pupil_idx_in_openface = 55
    for i in range(eye_pupil_len):
        ret['points'].append(ldmks['eye_lmk'][eye_pupil_idx_in_openface-2*i])
    # eye_center_len = 1
    # eye_center_idx_in_openface = -1
    x_sum, y_sum = 0, 0
    for i in range(8):
        x_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][0]
        y_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][1]
    ret['points'].append([x_sum // 8, y_sum//8])

    ret['points'].append(ldmks['face_lmk'][31])
    ret['points'].append(ldmks['face_lmk'][35])
    ret['points'].append(ldmks['face_lmk'][48])
    ret['points'].append(ldmks['face_lmk'][54])
    ret['points'].append(ldmks['face_lmk'][0])
    ret['points'].append(ldmks['face_lmk'][16])
    return ret


def convert_label_tool_pts_to_openface_eye_points(ldmks):
    if not ldmks:
        return False
    ret = {}
    ret['version'] = 1
    ret['n_points'] = 56
    ret['points'] = []

    # 左眼
    eye_lid_len = 12
    eye_lid_idx_in_openface = 8
    for i in range(eye_lid_len):
        ret['points'].append(ldmks['eye_lmk'][eye_lid_idx_in_openface+i])
    eye_ball_len = 8
    eye_ball_idx_in_openface = 0
    for i in range(eye_ball_len):
        ret['points'].append(ldmks['eye_lmk'][eye_ball_idx_in_openface+i])
    eye_pupil_len = 4
    eye_pupil_idx_in_openface = 27
    for i in range(eye_pupil_len):
        ret['points'].append(ldmks['eye_lmk'][eye_pupil_idx_in_openface-2*i])
    # eye_center_len = 1
    # eye_center_idx_in_openface = -1
    x_sum, y_sum = 0, 0
    for i in range(8):
        x_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][0]
        y_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][1]
    ret['points'].append([x_sum // 8, y_sum//8])

    # 右眼
    eye_lid_len = 12
    eye_lid_idx_in_openface = 36
    for i in range(eye_lid_len):
        ret['points'].append(ldmks['eye_lmk'][eye_lid_idx_in_openface+i])
    eye_ball_len = 8
    eye_ball_idx_in_openface = 28
    for i in range(eye_ball_len):
        ret['points'].append(ldmks['eye_lmk'][eye_ball_idx_in_openface+i])
    eye_pupil_len = 4
    eye_pupil_idx_in_openface = 55
    for i in range(eye_pupil_len):
        ret['points'].append(ldmks['eye_lmk'][eye_pupil_idx_in_openface-2*i])
    # eye_center_len = 1
    # eye_center_idx_in_openface = -1
    x_sum, y_sum = 0, 0
    for i in range(8):
        x_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][0]
        y_sum += ldmks['eye_lmk'][eye_pupil_idx_in_openface-i][1]
    ret['points'].append([x_sum // 8, y_sum//8])

    ret['points'].append(ldmks['face_lmk'][31])
    ret['points'].append(ldmks['face_lmk'][35])
    ret['points'].append(ldmks['face_lmk'][48])
    ret['points'].append(ldmks['face_lmk'][54])
    ret['points'].append(ldmks['face_lmk'][0])
    ret['points'].append(ldmks['face_lmk'][16])
    return ret


def convert_openface_eye_points_and_rect_to_label_tool_pts(path):
    if not os.path.exists(path):
        return None
    path_dst = path.replace('.csv', '.pts')
    ret_csv = load_openface_csv(path)
    ldmks = parse_openface_csv(ret_csv)
    label_tool_lmks = convert_openface_eye_points_to_label_tool_pts(ldmks)
    dump_pts(label_tool_lmks, path_dst)
    bbox = get_bbox_from_landmark68(label_tool_lmks['points'])
    path_dst = path.replace('.csv', '.rect')
    dump_bbox(bbox, path_dst)


def example_label_tool_pts_rect_show():
    root_dir = "cache/face_hd/"
    file_path = os.path.join(
        root_dir, "eye_data_20180613_dong.yang/豫Q-C8936(2655)-20170717124529-分神提醒.pts")
    image_path = root_dir + "white_6400_4800.png"
    image_path_dst = root_dir + "face_hd_dst.jpg"

    img = cv2.imread(image_path)

    ret = load_pts(file_path)
    bbox = load_bbox(file_path.replace('.pts', '.rect'))

    draw_label_tool_eye_points_with_index_and_line(
        img, image_path_dst+'_label_tool.jpg', [ret['points']], [bbox])
    print("-------------------------------")


def example_openface_pts_rect_show():
    root_dir = "cache/face_hd/"
    file_path = os.path.join(
        root_dir, "eye_data_20180613_dong.yang/豫Q-C8936(2655)-20170717124529-分神提醒.pts")
    image_path = root_dir + "white_6400_4800.png"
    image_path_dst = root_dir + "face_hd_dst.jpg"

    img = cv2.imread(image_path)

    ret = load_pts(file_path)
    bbox = load_bbox(file_path.replace('.pts', '.rect'))

    draw_label_tool_eye_points_with_index_and_line(
        img, image_path_dst+'_label_tool.jpg', [ret['points']], [bbox])
    print("-------------------------------")


def main(argv):
    example_label_tool_pts_rect_show()
    # file_path = "cache/face_hd/eye_data_20180613_dong.yang/19700101_010607 0571.pts"
    # file_path_dst = "cache/face_hd/face_hd_dst.pts"
    # file_path_csv = "cache/face_hd/face_hd.csv"
    # image_path = "cache/face_hd/white_6400_4800.png"
    # image_path_dst = "cache/face_hd/face_hd_dst.jpg"
    # file_path_3d_csv = "cache/face_hd/face_3d.csv"
    # ret = load_3d_landmark(file_path_3d_csv)
    # dump_list_to_csv(ret, file_path_3d_csv+'.csv')
    # img = cv2.imread(image_path)
    # draw_points(img, image_path_dst, [ret])
    # # ret = load_pts(file_path)
    # # dump_pts(ret, file_path_dst)
    # ret = load_openface_csv(file_path_csv)
    # lmks = parse_openface_csv(ret)
    # label_tool_lmks = select_openface_eye_points_to_label_tool_pts(lmks)
    # ret = load_pts(file_path)
    # bbox = load_bbox(file_path.replace('.pts', '.rect'))
    # draw_label_tool_eye_points_with_index_and_line(
    #     img, image_path_dst+'_label_tool.jpg', [ret['points']], [bbox])
    # # ret['version'] = 1
    # # ret['points'] = lmks['eye_lmk']
    # dump_pts(label_tool_lmks, file_path_dst)
    # ret = load_pts(file_path_dst)
    # draw_label_tool_eye_points_with_index_and_line(
    #     img, image_path_dst+'_label_tool2.jpg', [label_tool_lmks['points']])
    # draw_openface_eye_points_with_index_and_line(
    #     img, image_path_dst, [lmks['eye_lmk']])

    # bbox = get_bbox_from_landmark68(lmks['face_lmk'])
    # draw_points(img, image_path_dst+'.jpg', [lmks['face_lmk']], [bbox])
    file_dir = "cache/face_hd/eye_data_20180613_dong.yang/"
    for file_name in sorted(os.listdir(file_dir)):
        base, ext = os.path.splitext(file_name)
        if ext not in ['.csv']:
            continue
        file_path = os.path.join(file_dir, file_name)
        convert_openface_eye_points_to_label_tool_pts(file_path)


if __name__ == '__main__':
    app.run(main)
