import numpy as np
from PIL import Image


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    img = img.resize((512, 512), Image.BICUBIC)
    return img


def save_img(image_tensor, filename):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    image_numpy = image_numpy.astype(np.uint8)
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(filename)
    print("Image saved as {}".format(filename))


def celeba_label2color(label):
    color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0],
                  [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204],
                  [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0],
                  [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204],
                  [0, 51, 0], [255, 153, 51], [0, 204, 0]]
    res = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.uint8)
    for idx, color in enumerate(color_list):
        res[label == idx] = color

    return res


def get_img(image_tensor):
    image_numpy = image_tensor.float().numpy()
    image_numpy = (image_numpy + 1) / 2.0 * 18.0
    image_numpy = image_numpy.clip(0, 18)
    # image_numpy = image_numpy.astype(np.uint8)
    image_numpy = np.around(image_numpy)
    image_numpy = image_numpy[0, :, :]
    return image_numpy


def display(real_a_, real_b_, fake_b_, epoch_):
    a = real_a_.detach().cpu()[0, :, :, :]
    a = get_img(a)
    color = celeba_label2color(a)
    Image.fromarray(color).save("checkpoint/real_a_{}.png".format(epoch_))

    b = real_b_.detach().cpu()[0, :, :, :]
    b = get_img(b)
    color = celeba_label2color(b)
    Image.fromarray(color).save("checkpoint/real_b_{}.png".format(epoch_))

    b1 = fake_b_.detach().cpu()[0, :, :, :]
    b1 = get_img(b1)
    color = celeba_label2color(b1)
    Image.fromarray(color).save("checkpoint/fake_b_{}.png".format(epoch_))