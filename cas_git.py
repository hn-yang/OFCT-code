import os
import dlib
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from PyEMD import EMD, EEMD, CEEMDAN
import class_encoder

face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
font_type = cv2.FONT_HERSHEY_SIMPLEX
landmark_data = []


def temporal_ideal_filter(input_tensor, low_freq, high_freq, sampling_rate, axis=0):
    fft_result = fftpack.fft(input_tensor, axis=axis)
    frequencies = fftpack.fftfreq(input_tensor.shape[0], d=1.0 / sampling_rate)

    lower_bound = (np.abs(frequencies - low_freq)).argmin()
    upper_bound = (np.abs(frequencies - high_freq)).argmin()

    fft_result[upper_bound:-upper_bound] = 0
    filtered_result = fftpack.ifft(fft_result, axis=axis)

    return np.abs(filtered_result)


def perform_emd(signal, signal_data, output_path, index, sampling_freq):
    time = np.arange(len(signal_data) / sampling_freq)
    signal_data = np.array(signal_data)

    emd = EMD()
    imfs = emd.emd(signal_data, time)

    num_imfs = imfs.shape[0]

    imf_sum = np.zeros(imfs.shape[1])
    imf_sum_1 = np.zeros(imfs.shape[1])

    for idx, imf in enumerate(imfs):
        if idx != num_imfs - 1:
            imf_sum_1 = np.add(imf_sum_1, imf)
        if idx != 0:
            imf_sum = np.add(imf_sum, imf)

    return imf_sum, imf_sum_1


def crop_face_image(input_image, target_size):
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    detected_faces = face_detector(gray_image, 0)

    if len(detected_faces) == 0:
        print("Cannot find even one face")
        return None, None, None, None, None, None

    for face in detected_faces:
        landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(input_image, face).parts()])

    left_eye = landmarks[39]
    right_eye = landmarks[42]

    eye_width = int((right_eye[0, 0] - left_eye[0, 0]) / 2)
    center = [int((right_eye[0, 0] + left_eye[0, 0]) / 2), int((right_eye[0, 1] + left_eye[0, 1]) / 2)]

    cv2.rectangle(input_image,
                  (center[0] - int(4.5 * eye_width), center[1] - int(3.5 * eye_width)),
                  (center[0] + int(4.5 * eye_width), center[1] + int(5.5 * eye_width)),
                  (0, 0, 255), 2)

    top = max(center[1] - int(3 * eye_width), 0)
    bottom = center[1] + int(5 * eye_width)
    left = max(center[0] - int(4 * eye_width), 0)
    right = center[0] + int(4 * eye_width)

    cropped_image = input_image[top:bottom, left:right]
    resized_image = cv2.resize(cropped_image, (target_size, target_size))

    return landmarks, resized_image, top, bottom, left, right


def get_roi_bound(lower_index, upper_index, padding, landmarks):
    roi_points = landmarks[lower_index:upper_index]

    roi_high = roi_points[:, 0].argmax(axis=0)
    roi_low = roi_points[:, 0].argmin(axis=0)
    roi_left = roi_points[:, 1].argmin(axis=0)
    roi_right = roi_points[:, 1].argmax(axis=0)

    roi_high_value = roi_points[roi_high, 0]
    roi_low_value = roi_points[roi_low, 0]
    roi_left_value = roi_points[roi_left, 1]
    roi_right_value = roi_points[roi_right, 1]

    roi_high_extended = (roi_high_value + padding)[0, 0]
    roi_low_extended = (roi_low_value - padding)[0, 0]
    roi_left_extended = (roi_left_value - padding)[0, 0]
    roi_right_extended = (roi_right_value + padding)[0, 0]

    return roi_high_extended, roi_low_extended, roi_left_extended, roi_right_extended


def get_roi(flow_data, exclusion_percent):
    magnitudes, angles = cv2.cartToPolar(flow_data[:, :, 0], flow_data[:, :, 1], angleInDegrees=True)
    magnitudes = np.ravel(magnitudes)

    flow_x = np.ravel(flow_data[:, :, 0])
    flow_y = np.ravel(flow_data[:, :, 1])

    sorted_indices = np.argsort(magnitudes)
    threshold_index = int(len(magnitudes) * (1 - exclusion_percent))
    x_sum = 0
    y_sum = 0

    for i in range(threshold_index, len(sorted_indices)):
        index = sorted_indices[i]
        x_sum += flow_x[index]
        y_sum += flow_y[index]

    avg_x = x_sum / (len(sorted_indices) - threshold_index)
    avg_y = y_sum / (len(sorted_indices) - threshold_index)

    return avg_x, avg_y


def adjust_landmarks(gray_image, input_image, landmarks, top_frame, left_frame, width, height, target_size):
    detected_faces = face_detector(gray_image, 0)

    if len(detected_faces) == 0:
        landmarks[:, 0] = (landmarks[:, 0] - left_frame) * (target_size / width)
        landmarks[:, 1] = (landmarks[:, 1] - top_frame) * (target_size / height)
        adjusted_landmarks = landmarks
    else:
        adjusted_landmarks = np.matrix([[p.x, p.y] for p in landmark_predictor(input_image, detected_faces[0]).parts()])

    return adjusted_landmarks


def calculate_flow_magnitude(flow_vectors):
    flow_vectors = np.array(flow_vectors)
    flow_magnitudes = np.sum(flow_vectors ** 2, axis=1)
    flow_magnitudes = np.sqrt(flow_magnitudes)

    return flow_magnitudes



def analyze_flow(flow_data, imf_sum1, threshold1, threshold2):
    flow_data = np.array(flow_data)
    min_value = np.min(flow_data)
    flow_data = flow_data - min_value
    filtered_flow = []

    for idx in range(len(flow_data)):
        if flow_data[idx] >= threshold1:
            filtered_flow.append(idx)

    segmented_flow = []
    if len(filtered_flow) > 0:
        start = filtered_flow[0]
        end = filtered_flow[0]

        for i in range(len(filtered_flow)):
            if filtered_flow[i] >= end and filtered_flow[i] - end < 3:
                end = filtered_flow[i]
            else:
                segmented_flow.append([start, end])
                start = filtered_flow[i]
                end = filtered_flow[i]

        segmented_flow.append([start, end])

    filtered_flow = []
    segmented_flow = np.array(segmented_flow)

    for i in range(len(segmented_flow)):
        start = segmented_flow[i, 0]
        end = segmented_flow[i, 1]

        for j in range(start, end):
            left_bound = max(0, j - 30)
            right_bound = min(len(flow_data) - 1, j + 30)
            local_min = np.min(flow_data[left_bound:right_bound])
            local_min_imf = np.min(imf_sum1[left_bound:right_bound])

            if flow_data[j] - local_min > threshold2 and imf_sum1[j] - local_min_imf > 0.5:
                filtered_flow.append(j)

    final_segments = []
    if len(filtered_flow) > 0:
        start = filtered_flow[0]
        end = filtered_flow[0]

        for i in range(len(filtered_flow)):
            if filtered_flow[i] >= end and filtered_flow[i] - end < 3:
                end = filtered_flow[i]
            else:
                final_segments.append([start, end])
                start = filtered_flow[i]
                end = filtered_flow[i]

        final_segments.append([start, end])

    return np.array(final_segments)

def expand_flow_segments(flow_segments, edm_data):
    for i in range(len(flow_segments)):
        segment_start = flow_segments[i, 0]
        segment_end = flow_segments[i, 1]

        left_bound1 = max(0, segment_start - 30)
        right_bound1 = min(len(edm_data) - 1, segment_start + 30)
        left_bound2 = max(0, segment_end - 30)
        right_bound2 = min(len(edm_data) - 1, segment_end + 30)

        if segment_end > segment_start:
            high_value = np.max(edm_data[segment_start:segment_end])
        else:
            high_value = edm_data[segment_start]

        min_left = np.min(edm_data[left_bound1:right_bound1])
        min_left_index = np.argmin(edm_data[left_bound1:right_bound1]) + left_bound1
        min_right = np.min(edm_data[left_bound2:right_bound2])
        min_right_index = np.argmin(edm_data[left_bound2:right_bound2]) + left_bound2

        if min_left_index < segment_start:
            for j in range(segment_start - 1, -1, -1):
                if edm_data[j] - min_left < 0.33 * (high_value - min_left):
                    segment_start = j
                    break
                if edm_data[j] > edm_data[j + 1]:
                    segment_start = j + 2
                    break
        else:
            left_adj = max(segment_start - 10, 0)
            min_adj_left_index = np.argmin(edm_data[left_adj:segment_start + 1]) + left_adj
            if edm_data[segment_start] - edm_data[min_adj_left_index] > 0.3:
                segment_start = min_adj_left_index + 1

        if min_right_index > segment_end:
            for j in range(segment_end + 1, min_right_index):
                if edm_data[j] - min_right < 0.33 * (high_value - min_right):
                    segment_end = j
                    break
                if edm_data[j] > edm_data[j - 1]:
                    segment_end = j - 2
                    break
        else:
            right_adj = min(segment_end + 10, len(edm_data) - 1)
            min_adj_right_index = np.argmin(edm_data[segment_end:right_adj + 1]) + segment_end
            if edm_data[segment_end] - edm_data[min_adj_right_index] > 0.3:
                segment_end = min_adj_right_index - 1

        flow_segments[i, 0] = segment_start
        flow_segments[i, 1] = segment_end

    return flow_segments


def compute_mo_features(flow_data, percentage):
    r1, theta1 = cv2.cartToPolar(flow_data[:, :, 0], flow_data[:, :, 1], angleInDegrees=True)

    bin_edges = [0, 160, 180, 340, 360]
    hist, _ = np.histogram(theta1, bins=bin_edges)

    max_indices = np.argsort(hist)[-2:]

    max_bin1_values = r1[(theta1 >= bin_edges[max_indices[0]]) & (theta1 < bin_edges[max_indices[0] + 1])]
    max_bin2_values = r1[(theta1 >= bin_edges[max_indices[1]]) & (theta1 < bin_edges[max_indices[1] + 1])]

    r1 = np.ravel(max_bin1_values)
    r2 = np.ravel(max_bin2_values)

    x_new, y_new = 0, 0
    x1 = np.ravel(flow_data[:, :, 0])
    y1 = np.ravel(flow_data[:, :, 1])

    arg1 = np.argsort(r1)
    arg2 = np.argsort(r2)

    num1 = int(len(r2) * 0.8)
    num2 = int(len(r1) * 0.9)

    arg1 = arg1[num2:]
    arg2 = arg2[num1:]

    combined_arg = np.concatenate((arg1, arg2))

    for i in range(len(combined_arg)):
        a = combined_arg[i]
        x_new += x1[a]
        y_new += y1[a]

    x = x_new / len(combined_arg)
    y = y_new / len(combined_arg)

    return x, y


def process_flow_data(flow_data, lower_threshold, upper_threshold, position, index, k, offset, total_flow):
    fs = 1
    constant = 1
    lower_threshold += constant
    upper_threshold += constant

    flow_data = calculate_flow_magnitude(flow_data)  # Convert flow features to magnitude form
    flow_data = np.array(flow_data)

    text_features = class_encoder.class_encoder()
    flow_data_len = flow_data.shape[0]
    text_features_len = text_features.shape[0]

    flow_weight = 0.9
    text_weight = 1.1
    min_len = min(flow_data_len, text_features_len)
    flow_data[:min_len] = flow_weight * flow_data[:min_len] + text_weight * text_features[:min_len]

    position = f"{position}{index}----"

    filter_threshold = 2
    filtered_flow_data = temporal_ideal_filter(flow_data[offset:-offset], 1, filter_threshold, 30)  # Apply filter

    adjusted_len = len(filtered_flow_data) + 2  # Equivalent to 200

    flow_data_imf, imf_sum = perform_emd(flow_data[offset:-offset], filtered_flow_data, position, str(k - adjusted_len), fs)

    segmented_flow_data = analyze_flow(filtered_flow_data, imf_sum, lower_threshold, upper_threshold)
    segmented_flow_data = expand_flow_segments(segmented_flow_data, filtered_flow_data)  # Expand segments

    segmented_flow_data = segmented_flow_data + (k - adjusted_len) + offset
    for segment in segmented_flow_data:
        total_flow.append(segment)

    return total_flow


def non_maximum_suppression(flow_segments, threshold):
    flow_segments = np.array(flow_segments)
    merged_segments = [[0, 0]]

    for i in range(len(flow_segments)):
        is_new_segment = True
        if i == 0:
            merged_segments = np.vstack((merged_segments, [flow_segments[i, 0], flow_segments[i, 1]]))
            continue

        for j in range(1, len(merged_segments)):
            if flow_segments[i, 0] > merged_segments[j, 1] or flow_segments[i, 1] < merged_segments[j, 0]:
                iou = 0
            else:
                max_start = max(flow_segments[i, 0], merged_segments[j, 0])
                min_end = min(flow_segments[i, 1], merged_segments[j, 1])
                intersection_width = min_end - max_start
                np.seterr(divide="ignore", invalid="ignore")
                iou = max(intersection_width / (merged_segments[j, 1] - merged_segments[j, 0]),
                          intersection_width / (flow_segments[i, 1] - flow_segments[i, 0]))

            if iou > threshold:
                is_new_segment = False
                merged_segments[j, 1] = max(merged_segments[j, 1], flow_segments[i, 1])
                merged_segments[j, 0] = min(merged_segments[j, 0], flow_segments[i, 0])

        if is_new_segment:
            merged_segments = np.vstack((merged_segments, [flow_segments[i, 0], flow_segments[i, 1]]))

    return merged_segments




def detect_facial_movements(path1, path2, qian, hou, fs):  # 与16相比再增加两个位置眼睑部位

    path = path1 + path2 + '/'  # 视频图片文件夹的位置
    fileList1 = os.listdir(path)  # 图片路径

    # fileList1=sorted(fileList1, key=lambda x: int(x.split("_")[1].split(".")[0]))  #对提取的图片排序
    fileList1.sort()
    fileList = []
    l = 0
    for i in fileList1:
        if (l % fs == 0):
            fileList.append(i)
        l = l + 1
    k = 0  # 这里的k代表开始的位置
    start = k - 99  # 每一小段的开始和结束
    # end = k + 100
    end = k + 200
    move = 100  # 默认移动是100
    last = True  # 最后一段是否处理过
    label_vio = np.array([[0, 0]])
    while (k < len(fileList)):
        # while(k<len(fileList) and k<2000):
        start += move
        end += move
        if (end > len(fileList) and last == True):
            end = len(fileList) - 2  # 如果是最后一个，及没有200那么多，就调整end，   start不变
            last = False
        k = 0
        mid = False
        global start_1
        global end_1
        start_1 = start
        end_1 = end
        for i in fileList:
            k = k + 1
            if (k >= start):
                if (k == start):
                    flow1_total = [[0, 0]]  # 是存储了不同位置帧之间的光流
                    flow1_total1 = [[0, 0]]
                    flow1_total2 = [[0, 0]]
                    flow1_total3 = [[0, 0]]
                    flow2_total = [[0, 0]]
                    flow3_total = [[0, 0]]
                    flow3_total1 = [[0, 0]]
                    flow3_total2 = [[0, 0]]
                    flow3_total3 = [[0, 0]]
                    flow4_total = [[0, 0]]
                    flow4_total1 = [[0, 0]]
                    flow4_total2 = [[0, 0]]
                    flow4_total3 = [[0, 0]]
                    flow4_total4 = [[0, 0]]
                    flow4_total5 = [[0, 0]]
                    flow5_total1 = [[0, 0]]
                    flow5_total2 = [[0, 0]]
                    flow2_total1 = [[0, 0]]
                    flow6_total = [[0, 0]]
                    flow7_total = [[0, 0]]

                    img_rd = cv2.imread(path + i)  # D:/face_image_test/EP07_04/

                    img_size = 256
                    landmark0, img_rd, frame_shang, frame_xia, frame_left, frame_right = crop_face_image(img_rd, img_size)
                    # 记录框的位置，上下左右在整个图片中的坐标，和68点的位置。img_rd是被裁减之后的面部位置，并resize到256*256

                    gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)  # 变成灰度图
                    landmark0 = adjust_landmarks(gray, img_rd, landmark0, frame_shang, frame_left, frame_xia - frame_shang,
                                             frame_right - frame_left, img_size)  # 对人脸68个点的定位
                    # 相对与新图片的68点的位置。

                    round1 = 0
                    roil_right, roi1_left, roi1_low, roi1_high = get_roi_bound(17, 22, 0, landmark0)  # 左眉毛的位置

                    roi1_sma = []  # 存储了左眼的三个小的感兴趣区域，从里到外
                    roi1_sma.append([landmark0[20, 1] - (roi1_low - 15), landmark0[20, 0] - (roi1_left - 5)])
                    roi1_sma.append([landmark0[19, 1] - (roi1_low - 15), landmark0[19, 0] - (roi1_left - 5)])
                    roi1_sma.append([landmark0[18, 1] - (roi1_low - 15), landmark0[18, 0] - (roi1_left - 5)])

                    prevgray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left - 5:roil_right]

                    # 右眼
                    roi3_right, roi3_left, roi3_low, roi3_high = get_roi_bound(22, 27, 0, landmark0)
                    roi3_sma = []  # 存储了右眼的三个小的感兴趣区域，从里到外
                    roi3_sma.append([landmark0[23, 1] - (roi3_low - 15), landmark0[23, 0] - roi3_left])
                    roi3_sma.append([landmark0[24, 1] - (roi3_low - 15), landmark0[24, 0] - roi3_left])
                    roi3_sma.append([landmark0[25, 1] - (roi3_low - 15), landmark0[25, 0] - roi3_left])

                    prevgray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                    # print(prevgray_roi1.shape)

                    # 嘴巴处的四个
                    roi4_right, roi4_left, roi4_low, roi4_high = get_roi_bound(48, 67, 0, landmark0)
                    roi4_sma = []
                    roi4_sma.append([landmark0[48, 1] - (roi4_low - 15), landmark0[48, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[54, 1] - (roi4_low - 15), landmark0[54, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[51, 1] - (roi4_low - 15), landmark0[51, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[57, 1] - (roi4_low - 15), landmark0[57, 0] - (roi4_left - 20)])
                    roi4_sma.append([landmark0[62, 1] - (roi4_low - 15), landmark0[62, 0] - (roi4_left - 20)])

                    prevgray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]

                    # 鼻子两侧
                    roi5_right, roi5_left, roi5_low, roi5_high = get_roi_bound(30, 36, 0, landmark0)
                    roi5_sma = []
                    roi5_sma.append([landmark0[31, 1] - (roi5_low - 20), landmark0[31, 0] - (roi5_left - 30)])
                    roi5_sma.append([landmark0[35, 1] - (roi5_low - 20), landmark0[35, 0] - (roi5_left - 30)])

                    prevgray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]

                    roi2_right, roi2_left, roi2_low, roi2_high = get_roi_bound(29, 31, 13, landmark0)
                    prevgray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]

                else:

                    if (True):

                        img_rd1 = cv2.imread(path + i)  # D:/face_image_test/EP07_04/
                        # print(path+i)
                        img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]  # 按照第一个图的框切割出一个脸

                        img_rd = cv2.resize(img_crop, (img_size, img_size))
                        gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                        # 求全局的光流
                        gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                        # 使用Gunnar Farneback算法计算密集光流
                        flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow2 = np.array(flow2)

                        # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                        x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)
                        # print("全局运动为{}and{}".format(x1,y1))
                        flow2_total1.append([x1, y1])

                        # 进行面部对齐，移动切割框
                        l = 0
                        while ((x1 ** 2 + y1 ** 2) > 1):  # 移动比较大，相应移动脸的位置
                            l = l + 1
                            if (l > 3):
                                # print("ppp")
                                break
                            frame_left += int(round(x1))
                            frame_shang += int(round(y1))
                            frame_right += int(round(x1))
                            frame_xia += int(round(y1))

                            frame_left = max(0, frame_left)
                            frame_shang = max(0, frame_shang)

                            if frame_xia == 0:
                                print(path + i)
                                continue
                            img_rd1 = cv2.imread(path + i)

                            img_crop = img_rd1[frame_shang:frame_xia, frame_left:frame_right]
                            # cv2.imwrite('img_crop.jpg', img_crop)
                            img_rd = cv2.resize(img_crop, (img_size, img_size))
                            gray = cv2.cvtColor(img_rd, cv2.COLOR_RGB2GRAY)
                            # 求全局的光流
                            gray_roi2 = gray[roi2_low:roi2_high, roi2_left:roi2_right]
                            # 使用Gunnar Farneback算法计算密集光流
                            flow2 = cv2.calcOpticalFlowFarneback(prevgray_roi2, gray_roi2, None, 0.5, 3, 15, 5, 7, 1.5,
                                                                 0)
                            flow2 = np.array(flow2)

                            # him2, x1, y1 = get_roi_him(flow2[15:-10, 5:-5, :])
                            x1, y1 = get_roi(flow2[15:-10, 5:-5, :], 0.7)

                            # print("全局运动为{}and{}".format(x1, y1))
                            flow2_total1.append([x1, y1])
                        # 对齐完毕

                        gray_roi1 = gray[(roi1_low - 15):roi1_high + 5, roi1_left - 5:roil_right]
                        # 使用Gunnar Farneback算法计算密集光流
                        try:
                            flow1 = cv2.calcOpticalFlowFarneback(prevgray_roi1, gray_roi1, None, 0.5, 3, 15, 5, 7, 1.5,
                                                                 0)  # 计算整个左眉毛处的光流
                        except:
                            break
                        flow1[:, :, 0] = flow1[:, :, 0]
                        flow1[:, :, 1] = flow1[:, :, 1]
                        flow1=flow1-np.array([x1,y1])
                        # print("pppppp")
                        round1 = 10
                        roi1_sma = np.array(roi1_sma)
                        # print(roi1_sma)
                        a, b = compute_mo_features(flow1[round1:-round1, round1:-round1, :], 0.2)  # 去掉光流特征矩阵周边round大小的部分，求均值
                        a1, b1 = compute_mo_features(  # 一个感兴趣区域处的平均光流
                            flow1[roi1_sma[0, 0] - 10:roi1_sma[0, 0] + 10, roi1_sma[0, 1] - 10:roi1_sma[0, 1] + 10, :],
                            0.2)
                        a2, b2 = compute_mo_features(
                            flow1[roi1_sma[1, 0] - 10:roi1_sma[1, 0] + 10, roi1_sma[1, 1] - 10:roi1_sma[1, 1] + 10, :],
                            0.2)
                        a3, b3 = compute_mo_features(
                            flow1[roi1_sma[2, 0] - 10:roi1_sma[2, 0] + 10, roi1_sma[2, 1] - 10:roi1_sma[2, 1] + 10, :],
                            0.2)

                        flow1_total1.append([a1 - x1, b1 - y1])  # 局部区域减去全局光流
                        flow1_total2.append([a2 - x1, b2 - y1])
                        flow1_total3.append([a3 - x1, b3 - y1])
                        flow1_total.append([a - x1, b - y1])

                        gray_roi3 = gray[(roi3_low - 15):roi3_high + 5, roi3_left:roi3_right]
                        # 使用Gunnar Farneback算法计算密集光流
                        flow3 = cv2.calcOpticalFlowFarneback(prevgray_roi3, gray_roi3, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow3[:, :, 0] = flow3[:, :, 0]
                        flow3[:, :, 1] = flow3[:, :, 1]
                        round1 = 10

                        roi3_sma = np.array(roi3_sma)
                        # print(roi1_sma)
                        a, b = compute_mo_features(flow3[round1:-round1, round1:-round1, :], 0.3)  # 取0.3
                        a1, b1 = compute_mo_features(
                            flow3[roi3_sma[0, 0] - 10:roi3_sma[0, 0] + 10, roi3_sma[0, 1] - 10:roi3_sma[0, 1] + 10, :],
                            0.3)
                        a2, b2 = compute_mo_features(
                            flow3[roi3_sma[1, 0] - 10:roi3_sma[1, 0] + 10, roi3_sma[1, 1] - 10:roi3_sma[1, 1] + 10, :],
                            0.3)
                        a3, b3 = compute_mo_features(
                            flow3[roi3_sma[2, 0] - 10:roi3_sma[2, 0] + 10, roi3_sma[2, 1] - 10:roi3_sma[2, 1] + 10, :],
                            0.3)

                        flow3_total1.append([a1 - x1, b1 - y1])
                        flow3_total2.append([a2 - x1, b2 - y1])
                        flow3_total3.append([a3 - x1, b3 - y1])
                        flow3_total.append([a - x1, b - y1])

                        gray_roi4 = gray[(roi4_low - 15):roi4_high + 10, roi4_left - 20:roi4_right + 20]

                        # 使用Gunnar Farneback算法计算密集光流
                        flow4 = cv2.calcOpticalFlowFarneback(prevgray_roi4, gray_roi4, None, 0.5, 3, 15, 5, 7, 1.5, 0)
                        flow4[:, :, 0] = flow4[:, :, 0]
                        flow4[:, :, 1] = flow4[:, :, 1]
                        round1 = 10
                        roi4_sma = np.array(roi4_sma)
                        # print(roi1_sma)
                        a, b = compute_mo_features(flow4[round1:-round1, round1:-round1, :], 0.3)
                        a1, b1 = compute_mo_features(
                            flow4[roi4_sma[0, 0] - 10:roi4_sma[0, 0] + 10, roi4_sma[0, 1] - 10:roi4_sma[0, 1] + 20, :],
                            0.2)
                        a2, b2 = compute_mo_features(
                            flow4[roi4_sma[1, 0] - 10:roi4_sma[1, 0] + 10, roi4_sma[1, 1] - 20:roi4_sma[1, 1] + 10, :],
                            0.2)
                        a3, b3 = compute_mo_features(
                            flow4[roi4_sma[2, 0] - 10:roi4_sma[2, 0] + 10, roi4_sma[2, 1] - 10:roi4_sma[2, 1] + 10, :],
                            0.2)
                        a4, b4 = compute_mo_features(
                            flow4[roi4_sma[3, 0] - 10:roi4_sma[3, 0] + 10, roi4_sma[3, 1] - 10:roi4_sma[3, 1] + 10, :],
                            0.2)
                        a5, b5 = compute_mo_features(
                            flow4[roi4_sma[4, 0] - 10:roi4_sma[4, 0] + 10, roi4_sma[4, 1] - 10:roi4_sma[4, 1] + 10, :],
                            0.2)

                        flow4_total1.append([a1 - x1, b1 - y1])
                        flow4_total2.append([a2 - x1, b2 - y1])
                        flow4_total3.append([a3 - x1, b3 - y1])
                        flow4_total4.append([a4 - x1, b4 - y1])
                        flow4_total5.append([a5 - x1, b5 - y1])
                        flow4_total.append([a - x1, b - y1])

                        gray_roi5 = gray[(roi5_low - 20):roi5_high + 5, roi5_left - 30:roi5_right + 30]
                        # 使用Gunnar Farneback算法计算密集光流
                        flow5 = cv2.calcOpticalFlowFarneback(prevgray_roi5, gray_roi5, None, 0.5, 3, 15, 5, 7, 1.5, 0)

                        round1 = 10
                        roi5_sma = np.array(roi5_sma)

                        a1, b1 = compute_mo_features(
                            flow5[roi5_sma[0, 0] - 20:roi5_sma[0, 0] + 5, roi5_sma[0, 1] - 20:roi5_sma[0, 1] + 10, :],
                            0.2)

                        a2, b2 = compute_mo_features(
                            flow5[roi5_sma[1, 0] - 20:roi5_sma[1, 0] + 5, roi5_sma[1, 1] - 10:roi5_sma[1, 1] + 20, :],
                            0.2)

                        flow5_total1.append([a1 - x1, b1 - y1])
                        flow5_total2.append([a2 - x1, b2 - y1])
                        round1 = 5

            if (k == end):
                hh = end - start + 1

                # print( pathp+"/"+str(k)+ ".npy")
                # np.save( pathp+"/"+str(k)+ ".npy", flow)

                totalflow = []
                totalflowmic = []
                totalflowmac = []
                a = 1
                # totalflow=process(flow1_total,1.4,1.8,"left_eye",0,k,a,totalflow)

                totalflow = process_flow_data(flow1_total1, 1.5, 2.2, "left_eye", 1, k, a, totalflow)
                totalflow = process_flow_data(flow1_total2, 1.5, 2.2, "left_eye", 2, k, a, totalflow)
                totalflow = process_flow_data(flow1_total3, 1.5, 2.2, "left_eye", 3, k, a, totalflow)

                # totalflow=process(flow3_total,1.4,1.8,"right_eye",0,k,a,totalflow)
                totalflow = process_flow_data(flow3_total1, 1.5, 2.2, "right_eye", 1, k, a, totalflow)
                totalflow = process_flow_data(flow3_total2, 1.5, 2.2, "right_eye", 2, k, a, totalflow)
                totalflow = process_flow_data(flow3_total3, 1.5, 2.2, "right_eye", 3, k, a, totalflow)

                # totalflow=process(flow4_total,1.4,1.85,"mouth",0,k,a,totalflow)
                totalflow = process_flow_data(flow4_total1, 1.3, 1.85, "mouth", 1, k, a, totalflow)
                totalflow = process_flow_data(flow4_total2, 1.3, 1.85, "mouth", 2, k, a, totalflow)
                totalflow = process_flow_data(flow4_total3, 1.3, 1.85, "mouth", 3, k, a, totalflow)
                totalflow = process_flow_data(flow4_total4, 1.3, 1.85, "mouth", 4, k, a, totalflow)
                totalflow = process_flow_data(flow4_total5, 1.3, 1.85, "mouth", 5, k, a, totalflow)

                totalflow = process_flow_data(flow5_total1, 1.4, 2.1, "nose", 1, k, a, totalflow)
                totalflow = process_flow_data(flow5_total2, 1.4, 2.1, "nose", 2, k, a, totalflow)

                totalflow = np.array(non_maximum_suppression(totalflow, 0.55))  # 把所有通道融合起来
                totalflow = np.array(non_maximum_suppression(totalflow, 0.55))
                totalflow_1 = totalflow - (k - hh)
                move = 100
                for i in range(len(totalflow_1)):
                    # if (totalflow_1[i, 0] - (k - hh) < 175):
                    if (totalflow_1[i, 0] < 100 and totalflow_1[i, 1] > 100):
                        if (totalflow_1[i, 1] < 150):
                            move = totalflow_1[i, 1] + 20
                        elif (totalflow_1[i, 0] > 50):
                            move = totalflow_1[i, 0] - 20
                        else:
                            a = min(189, totalflow_1[i, 1])
                            move = a + 10

                label_vio = np.vstack((label_vio, totalflow))
                break
    print("全部：")
    # print(label_vio)

    label_video_update = []  # 去除一些太短的片段
    label_video_update1 = []
    for i in range(len(label_vio)):
        if (label_vio[i, 1] - label_vio[i, 0] >= 12 and label_vio[i, 1] - label_vio[i, 0] <= 200):
            label_video_update.append([label_vio[i, 0], label_vio[i, 1]])
    label_video_update.sort()
    label_video_update = np.array(non_maximum_suppression(label_video_update, 0.2))
    label_video_update = np.array(non_maximum_suppression(label_video_update, 0.2))
    start = 0
    end = 0
    for i in range(len(label_video_update)):
        if (label_video_update[i, 1] != 0):
            if start != 0:
                if label_video_update[i, 0] - end < 15 and (label_video_update[i, 1] - label_video_update[i, 0]) > 15:
                    c = 1
                    continue
            label_video_update1.append([label_video_update[i, 0], label_video_update[i, 1]])
            start = label_video_update[i, 0]
            end = label_video_update[i, 1]
    label_video_update1 = np.array(label_video_update1)
    # print(label_video_update1)
    return label_video_update1

