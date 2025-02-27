# -*- coding: utf-8 -*-
import os
import shutil

import samm_util as SAMM
import cas_util as CAS
from time import *
import pandas as pd
import xlrd
import xlwt


import os
import shutil
import numpy as np  # 数据处理的库 numpy
import samm_util as SAMM
import cas_git as CAS
from openpyxl import load_workbook
from time import *
import pandas as pd
import xlrd
import xlwt
import csv


path_label='/home/yanghn/work/ME/MEGC202201/casme2_annotation.csv'
def test_SAMM(path5,flow):
    # print("vio")
    # print(path5)
    flow = np.array(flow)
    # data1 = xlrd.open_workbook(path_label)
    # table = data1.sheets()[0]
    actual_df = pd.read_csv('/home/yanghn/work/ME/megc202201/casme2_annotationNoheader.csv',header=None)

    lable_vio = []
    num_micro = 0
    for j in range(len(flow)):
        start2 = flow[j, 0]
        end2 = flow[j, 1]
        if (end2 - start2 <= 15):
            # print("lll")
            num_micro += 1
    for rown in range(actual_df.shape[0]):
        vio = actual_df.iloc[rown, 0]

        if vio == path5:
            start = int(actual_df.iloc[rown, 1])
            # mid = int(actual_df.iloc[rown, 3])
            end = int(actual_df.iloc[rown, 2])


            lable_vio.append([start, end])


    lable_vio=np.array(lable_vio)
    true_lable=0
    true_lable_03=0
    true_lable_02=0
    true_lable_04=0
    true_lable_mic=0
    true_lable_mic_03 = 0
    true_lable_mic_02 = 0
    true_lable_mic_04 = 0
    FPPP = []
    for i in range(len(lable_vio)):
        start1 = lable_vio[i, 0]
        end1 = lable_vio[i, 1]
        percent=0

        for j in range(len(flow)):
            start2=flow[j,0]
            end2 = flow[j,1]
            if(not(end2<start1 or start2>end1)):
                min_start=min(start1,start2)
                max_start=max(start1,start2)
                min_end = min(end1, end2)
                max_end = max(end1, end2)
                percent=(float(min_end-max_start))/(max_end-min_start)
                # print(percent)


                if(percent>=0.5):
                    FPPP.append(j)
                    print("lable:{},{},{}  test:{},{},{} percent={}".format(start1,end1,end1-start1,start2,end2,end2-start2,percent))
                    true_lable += 1
                    true_lable_04 += 1
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if(end1-start1<=15):
                        true_lable_mic += 1
                        true_lable_mic_03+=1
                        true_lable_mic_02 += 1
                        true_lable_mic_04 += 1
                        # print("正确的微表情")
                    with open("my_cas.csv", "a", newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ path5, start1, end1, start2, end2, "TP"])


                    break
                if (percent >= 0.4):
                    true_lable_04 += 1
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if (end1 - start1 <= 15):
                        true_lable_mic_03 += 1
                        true_lable_mic_02 += 1
                        true_lable_mic_04+= 1
                    break
                if (percent >= 0.3):
                    true_lable_03 += 1
                    true_lable_02 += 1
                    if (end1 - start1 <= 15):

                        true_lable_mic_03 += 1
                        true_lable_mic_02 += 1

                    break
                if (percent >= 0.2):
                    true_lable_02 += 1
                    if (end1 - start1 <= 15):

                        true_lable_mic_02 += 1

                    break


        if(percent<0.5):
            with open("my_cas.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([path5, start1, end1, '', '', "FN"])
    FPPP=set(FPPP)
    FPPP = list(FPPP)
    for j in range(len(flow)):
        if j not in FPPP:
            start2 = flow[j, 0] + 1  # 预测的标签
            end2 = flow[j, 1] + 1
            with open("my_cas.csv", "a", newline='') as f:
                writer = csv.writer(f)
                writer.writerow([path5, '', '', start2, end2, "FP"])

    print("正确的数量：{}".format(true_lable))
    print("测试出的数量：{}".format(len(flow)))
    print("真实的表情标注数量：{}".format(len(lable_vio)))
    return num_micro,true_lable,true_lable_02,true_lable_03,true_lable_04,true_lable_mic,true_lable_mic_04,true_lable_mic_03,len(flow),len(lable_vio)


def detect_expressionCASME2(path,name,fps):
    fileList = os.listdir(path)
    fileList.sort()
    # fileList=[ '005_7_699']
    # fileList.sort(key=lambda x: int(x.replace("sub", "").split('-')[0]))
    # fileList=['sub01', 'sub02', 'sub03']
    print(fileList)

    allvio_rightlable = 0
    allvio_rightlable_02 = 0
    allvio_rightlable_03 = 0
    allvio_rightlable_04 = 0
    allvio_rightlable_mic = 0
    allvio_rightlable_mic_02 = 0
    allvio_rightlable_mic_03 = 0
    all_num_micro = 0
    allvio_test = 0
    alllable_num = 0

    for vio1 in fileList:
        path1=path+vio1+"/"
        fileList1 = os.listdir(path1)
        fileList1.sort()
        print(fileList1)
        for vio in fileList1:
            if name == "samm":
                # pp = SAMM.draw_roiline19(path, vio, 6, -4, 7)
                pp = SAMM.detect_facial_movements(path1, vio, 6, -4, 7)
                pp = pp * 7
            elif name == "cas":
                # pp = CAS.draw_roiline19_1(path, vio, 0, -4, 1)
                # pp = CAS.draw_roiline19_1(path1, vio, 0, -4, 1)
                pp = CAS.detect_facial_movements(path1, vio, 0, -4, 1)
                # pp = CAS.draw_roiline19_2(path, vio, 0, -4, 1)
                # pp= CAS.ab(path, vio, 0, -4, 1)
            pp = pp.tolist()
            pp.sort()
            print(f"{vio}:" + f"{pp}")

            num_micro, true_lable, true_lable_02, true_lable_03, true_lable_04, true_lable_mic, true_lable_mic_02, true_lable_mic_03, test_true, lable_num = test_SAMM(
                vio, pp)

            allvio_rightlable += true_lable
            allvio_rightlable_02 += true_lable_02
            allvio_rightlable_03 += true_lable_03
            allvio_rightlable_04 += true_lable_04
            allvio_rightlable_mic += true_lable_mic
            allvio_rightlable_mic_03 += true_lable_mic_03
            allvio_rightlable_mic_02 += true_lable_mic_02
            allvio_test += test_true
            alllable_num += lable_num
            all_num_micro += num_micro
            print("在" + vio + "视频中有{}个正确的检测结果".format(true_lable))

    print("-------------------------")
    print("-------------------------")
    print("-------------------------")
    print("共有{}个正确的微表情的检测结果".format(allvio_rightlable_mic))
    print("共有{}个正确的宏表情的检测结果".format(allvio_rightlable - allvio_rightlable_mic))
    print("共有{}个正确的检测结果".format(allvio_rightlable))
    print("共预测出{}个微表情".format(all_num_micro))
    print("共预测出{}个宏表情".format(allvio_test - all_num_micro))
    print("总共预测出{}个结果".format((allvio_test)))
    print("总共有{}个真实的表情标注".format(alllable_num))

    print("------------------")
    print("------------------")
    print("------------------")
    print("iou=0.5")
    P=(allvio_rightlable)/(allvio_test)#准确率
    R=(allvio_rightlable)/(alllable_num)#召回率
    F=(2*P*R)/(P+R)
    print("计算P系数,准确率："+str(P))
    print("计算R系数,召回率："+str(R))
    print("计算F系数,综合评价："+str(F))

    print("iou=0.5 for mic")
    P = (allvio_rightlable_mic) / (all_num_micro)  # 准确率
    R = (allvio_rightlable_mic) / (57)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("iou=0.5 for mac")
    P = (allvio_rightlable-allvio_rightlable_mic) / (allvio_test-all_num_micro)  # 准确率
    R = (allvio_rightlable-allvio_rightlable_mic) / (300)  # 召回率
    F = (2 * P * R) / (P + R)
    print("计算P系数,准确率：" + str(P))
    print("计算R系数,召回率：" + str(R))
    print("计算F系数,综合评价：" + str(F))

    print("------------------")
    print("------------------")
    print("------------------")

    # 打开一个文件用于写入（如果文件不存在会自动创建）
    with open('/data/databases/yhndataset/cas_result/aoutput.txt', 'a') as f:
        # 将内容写入文件
        f.write("-------------------------\n")
        f.write("共有{}个正确的微表情的检测结果\n".format(allvio_rightlable_mic))
        f.write("共有{}个正确的宏表情的检测结果\n".format(allvio_rightlable - allvio_rightlable_mic))
        f.write("共有{}个正确的检测结果\n".format(allvio_rightlable))
        f.write("共预测出{}个微表情\n".format(all_num_micro))
        f.write("共预测出{}个宏表情\n".format(allvio_test - all_num_micro))
        f.write("总共预测出{}个结果\n".format(allvio_test))
        f.write("总共有{}个真实的表情标注\n".format(alllable_num))

        f.write("------------------\n")
        f.write("------------------\n")
        f.write("------------------\n")
        f.write("iou=0.5\n")

        # 计算并写入准确率、召回率、F系数
        P = (allvio_rightlable) / (allvio_test)  # 准确率
        R = (allvio_rightlable) / (alllable_num)  # 召回率
        F = (2 * P * R) / (P + R)
        f.write("计算P系数,准确率：" + str(P) + "\n")
        f.write("计算R系数,召回率：" + str(R) + "\n")
        f.write("计算F系数,综合评价：" + str(F) + "\n")

        f.write("iou=0.5 for mic\n")
        P = (allvio_rightlable_mic) / (all_num_micro)  # 微表情准确率
        R = (allvio_rightlable_mic) / (57)  # 微表情召回率
        F = (2 * P * R) / (P + R)
        f.write("计算P系数,准确率：" + str(P) + "\n")
        f.write("计算R系数,召回率：" + str(R) + "\n")
        f.write("计算F系数,综合评价：" + str(F) + "\n")

        f.write("iou=0.5 for mac\n")
        P = (allvio_rightlable - allvio_rightlable_mic) / (allvio_test - all_num_micro)  # 宏表情准确率
        R = (allvio_rightlable - allvio_rightlable_mic) / (300)  # 宏表情召回率
        F = (2 * P * R) / (P + R)
        f.write("计算P系数,准确率：" + str(P) + "\n")
        f.write("计算R系数,召回率：" + str(R) + "\n")
        f.write("计算F系数,综合评价：" + str(F) + "\n")
        f.write("-------------------------------")
        f.write("-------------------------------")
        f.write("\n")
        f.write("\n")


    # print("iou=0.4")
    # P = (allvio_rightlable_04) / (allvio_test)  # 准确率
    # R = (allvio_rightlable_04) / (alllable_num)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))
    #
    # print("iou=0.4 for mic")
    # P = (allvio_rightlable_mic_02) / (all_num_micro)  # 准确率
    # R = (allvio_rightlable_mic_02) / (57)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))
    #
    # print("iou=0.4 for mac")
    # P = (allvio_rightlable_04-allvio_rightlable_mic_02) / (allvio_test - all_num_micro)  # 准确率
    # R = (allvio_rightlable_04-allvio_rightlable_mic_02) / (300)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))
    #
    # print("------------------")
    # print("------------------")
    # print("------------------")
    # print("iou=0.3")
    # P = (allvio_rightlable_03) / (allvio_test)  # 准确率
    # R = (allvio_rightlable_03) / (alllable_num)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))
    #
    # print("iou=0.3 for mic")
    # P = (allvio_rightlable_mic_03) / (all_num_micro)  # 准确率
    # R = (allvio_rightlable_mic_03) / (57)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))
    #
    # print("iou=0.3 for mac")
    # P = (allvio_rightlable_03 - allvio_rightlable_mic_03) / (allvio_test - all_num_micro)  # 准确率
    # R = (allvio_rightlable_03 - allvio_rightlable_mic_03) / (300)  # 召回率
    # F = (2 * P * R) / (P + R)
    # print("计算P系数,准确率：" + str(P))
    # print("计算R系数,召回率：" + str(R))
    # print("计算F系数,综合评价：" + str(F))


def detect_expression(path,name,fps):
    fileList = os.listdir(path)
    fileList.sort()
    # fileList=[ '005_7_699']

    print(fileList)
    data = pd.DataFrame(columns=["vid","onset","offset","type"])

    for vio in fileList:

        # vio= "sub02_2250"
        print(vio)
        if name == "samm":
            # pp = SAMM.draw_roiline19(path, vio, 6, -4, 7)
            pp = SAMM.draw_roiline19_2(path, vio, 6, -4, 7)
            pp = pp * 7
        elif name == "cas":
            # pp = CAS.draw_roiline19_1(path, vio, 0, -4, 1)
            pp = CAS.draw_roiline19_1(path, vio, 0, -4, 1)
            # pp = CAS.draw_roiline19_2(path, vio, 0, -4, 1)
            # pp= CAS.ab(path, vio, 0, -4, 1)
        pp = pp.tolist()
        pp.sort()
        print(f"{vio}:" + f"{pp}")




def main():
    path_cas = "/data/databases/yhndataset/CASme2/rawpic/"
    # path_cas23 = "F:/MEdataset/CAS(ME)2/rawpic/"
    # path_cas23 = "/home/yanghn/work/ME/megc202201/megc2201v/CAS_Test_cropped/"
    # detect_expression(path_cas23, 'cas', 30)
    # path_cas ="/home/yanghn/work/ME/megc202201/megc2201v/divide22cas/"

    detect_expressionCASME2(path_cas, 'cas', 30)


if __name__ == '__main__':
    main()
