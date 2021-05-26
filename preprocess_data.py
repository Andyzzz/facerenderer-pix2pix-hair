
# -*- coding: utf-8 -*-
import numpy as np
import cv2
from PIL import Image
import dlib
import csv
import os
import glob

# 生成合适的csv文件
def get_csv(image_dir, csv_name):
    f = open(csv_name, 'w', encoding='utf-8')
    csv_writer = csv.writer(f)

    for i in range(0, 10000):
        image_name = str(i).zfill(5) + ".jpg"
        full_path = image_dir + image_name
        if os.path.isfile(full_path):
            csv_writer.writerow([image_name])
        else:
            print(image_name)

    f.close()



# dlib检测关键点
def get_landmark_from_img(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./dataset/shape_predictor_68_face_landmarks.dat")
    POINTS_NUM_LANDMARK = 68

    dets = detector(img, 1)
    if len(dets) == 0:
        print("no face in the fake image")
        return None, -1
    ret = 0
    rectangle = dets[0]

    # left, top, right, bottom = max(0, rectangle.left()), max(0, rectangle.top()), min(w, rectangle.right()), min(h, rectangle.bottom())
    landmark_shape = predictor(img, rectangle)
    landmark_arr = np.zeros((68, 2))
    for i in range(0, POINTS_NUM_LANDMARK):
        landmark_arr[i, 0] = landmark_shape.part(i).x  # x
        landmark_arr[i, 1] = landmark_shape.part(i).y  # y

    # # convert to relative coordinate
    # x = landmark_arr[0, 0]
    # y = landmark_arr[1, 0]
    # landmark_arr[0, :] = landmark_arr[0, :] - x
    # landmark_arr[1, :] = landmark_arr[1, :] - y

    return landmark_arr, ret


def draw_landmark(img, landmark_arr):
    point_size = 2
    point_color = (0, 255, 0)
    thickness = 4
    r, c = landmark_arr.shape
    if r > c:
        for i in range(0, r):
            x = landmark_arr[i, 0]
            y = landmark_arr[i, 1]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
    elif r < c:
        for i in range(0, c):
            x = landmark_arr[0, i]
            y = landmark_arr[1, i]
            cv2.circle(img, (int(x), int(y)), point_size, point_color, thickness)
            cv2.putText(img, str(i), (int(x) + 1, int(y) + 1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

    cv2.imshow("image", img)
    cv2.waitKey()


# 验证contour格式，用dlib的关键点代替contour
def draw_landmark_contour(img_ori, landmark):

    img = img_ori.copy()
    # landmark2 = []
    # for i in range(0, 17):
    #     landmark2.append([landmark[i, 0], landmark[i, 1]])
    #
    # for i in range(26, 16, -1):
    #     landmark2.append([landmark[i, 0], landmark[i, 1]])
    # landmark2 = np.array(landmark2)

    landmark2 = np.zeros((27, 2))
    for i in range(0, 17):
        landmark2[i, 0] = landmark[i, 0]
        landmark2[i, 1] = landmark[i, 1]

    ind = 17
    for i in range(26, 16, -1):
        landmark2[ind, 0] = landmark[i, 0]
        landmark2[ind, 1] = landmark[i, 1]
        ind += 1


    # landmark2 = np.expand_dims(landmark2, axis=1)
    landmark2 = landmark2.reshape((-1, 1, 2)).astype(np.int32)
    print(landmark2.shape)

    contours = [landmark2]
    ind = 0
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, ind, (255, 255, 255), -1)
    # 腐蚀
    kernal = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernal, iterations=1)

    # cv2.imshow("image", img)
    # cv2.waitKey()

    # img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # mask = np.ones_like(img_gray)*255
    # mask[np.where(img_gray > 0.0)] = 0

    # mask2 = np.expand_dims(mask, axis=2)
    # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)
    # cv2.imshow("image", mask2)
    # cv2.waitKey()

    return mask


# 抠出人脸区域
def get_face_img(img):
    landmark, ret = get_landmark_from_img(img)
    if ret < 0:
        return img, ret
    mask = draw_landmark_contour(img, landmark)
    # mask2 = np.expand_dims(mask, axis=2)
    # mask2 = np.concatenate((mask2, mask2, mask2), axis=2)/255
    img = img*np.uint8(mask/255)

    img[np.where(mask == 0)] = 255
    # img = img.astype(np.uint8)
    # cv2.imshow("image", img)
    # cv2.waitKey()


    return img, ret


def get_bbox(img_gray):
    index = np.where(img_gray < 240)
    left, right, top, bottom = np.min(index[1]), np.max(index[1]), np.min(index[0]), np.max(index[0])
    h, w = bottom-top, right-left
    return [top, left, h, w]


def get_img_512x512(img_name):
    PIL_img = Image.open(img_name)
    img = np.array(PIL_img)
    h, w, c = img.shape

    img_gray = np.array(PIL_img.convert('L'))
    bbox = get_bbox(img_gray)

    face_rec = 480.0

    if bbox[2] > face_rec or bbox[3] > face_rec:
        scale = min(face_rec/bbox[2], face_rec/bbox[3])
        PIL_img = PIL_img.resize((int(w*scale), int(h*scale)))
        img = np.array(PIL_img)
        img_gray = np.array(PIL_img.convert('L'))
        bbox = get_bbox(img_gray)

    img_ret = np.ones((512, 512, 3), dtype='uint8')*255
    top, left, h, w = bbox[0], bbox[1], bbox[2], bbox[3]
    img_ret[256-h//2:256-h//2+h, 256-w//2:256-w//2+w, :] = img[top:top+h, left:left+w, :]

    return img_ret


def batch_process():

    image_dir = "/home/zhang/zydDataset/faceRendererData/data512x512/"
    csv_name = "./dataset/celeba_train3.csv"
    # f = open(csv_name, 'w', encoding='utf-8')
    # csv_writer = csv.writer(f)
    #
    # for i in range(1, 1151):
    #     image_name = str(i).zfill(5) + ".jpg"
    #     full_path = image_dir + image_name
    #     if os.path.isfile(full_path):
    #         csv_writer.writerow([image_name])
    #     else:
    #         print(image_name)
    #
    # f.close()

    path_csv_file = "./dataset/celeba_train_crop3.csv"

    tgt_dir = "/home/zhang/zydDataset/faceRendererData/data512x512_crop3/"

    with open(path_csv_file, 'r') as f:
        reader = csv.reader(f)
        image_paths = list(reader)
    print(len(image_paths))
    for i in range(0, len(image_paths)):
        imagename = image_paths[i]
        img = cv2.imread(os.path.join(image_dir, imagename[0]))

        new_img, ret = get_face_img(img)
        if ret < 0:
            continue

        cv2.imwrite(os.path.join(tgt_dir, imagename[0]), new_img)


def get_quality(name):
    gray = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    res = cv2.Laplacian(gray, cv2.CV_64F).var()
    return res


if __name__ == "__main__":
    # img = cv2.imread("/home/zhang/zydDataset/faceRendererData/rawscan_masked/1_340/2_smile/1.jpg")
    # landmark, rec = get_landmark_from_img(img)
    # draw_landmark(img, landmark)
    # draw_landmark_contour(img, landmark)

    # img = get_img_512x512("/home/zhang/zydDataset/faceRendererData/1.jpg")
    # # img = cv2.imread("/home/zhang/zydDataset/faceRendererData/00914.jpg")
    # img2, ret = get_face_img(img)
    # cv2.imwrite("/home/zhang/zydDataset/faceRendererData/1_1.jpg", img2[:,:,::-1])

    # img = np.array(Image.open("/home/zhang/zydDataset/faceRendererData/data512x512_2000/00001.jpg"))
    # img = img[:, :, ::-1]


    # image_dir = "/home/zhang/zydDataset/faceRendererData/data512x512_2000/"
    # csv_name = "./dataset/celeba_train.csv"
    # # get_csv(image_dir, csv_name)
    #
    # path_csv_file = "./dataset/celeba_train.csv"
    #
    # tgt_dir = "/home/zhang/zydDataset/faceRendererData/data512x512_crop/"

    # with open(path_csv_file, 'r') as f:
    #     reader = csv.reader(f)
    #     image_paths = list(reader)
    #
    # for imagename in image_paths:
    #     img = cv2.imread(os.path.join(image_dir, imagename[0]))
    #
    #     new_img, ret = get_face_img(img)
    #     if ret < 0:
    #         continue
    #
    #     cv2.imwrite(os.path.join(tgt_dir, imagename[0]), new_img)


    # get_csv("/home/zhang/zydDataset/faceRendererData/data512x512_crop3/", "./dataset/celeba_train_crop3.csv")

    # batch_process()

    lis = sorted(glob.glob("/home/zhang/zydDataset/asianFaces/foreground_pachong/*.jpg"))
    q = []
    for i in range(0, len(lis)):
        # print(i, lis[i])
        r = get_quality(lis[i])
        q.append(r)
        if r < 529:
            print(lis[i])
        # print("res: ", r)
    print((max(q)+min(q))/2)
