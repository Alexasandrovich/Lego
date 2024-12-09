import os
import random
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def get_class_id_from_filename(fname):
    # Предполагается, что имя файла формата: classId-instanceId.jpg
    # Пример: '1-1.jpg' -> class_id = 1
    return fname.split('-')[0]


def get_augmentation_transforms():
    # Трансформации, которые применяются после того, как объект уже вырезан и приведён к размеру 448x448.
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(degrees=15),
        # T.ColorJitter(contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.5], [0.5])
    ])


def find_closest_contour_to_center(img_bin, center_x, center_y):
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    min_dist = float('inf')
    chosen_contour = None
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            dist = (cx - center_x) ** 2 + (cy - center_y) ** 2
            if dist < min_dist:
                min_dist = dist
                chosen_contour = cnt
    return chosen_contour


def crop_and_resize_contour(img_gray, contour, output_size=448):
    x, y, w, h = cv2.boundingRect(contour)
    roi = img_gray[y:y + h, x:x + w]

    max_side = max(w, h)
    square = np.zeros((max_side, max_side), dtype=roi.dtype)

    start_x = (max_side - w) // 2
    start_y = (max_side - h) // 2
    square[start_y:start_y + h, start_x:start_x + w] = roi

    square_resized = cv2.resize(square, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return square_resized


class LegoDataset(Dataset):
    def __init__(self, root_dir, downscale_width=1024, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.downscale_width = downscale_width
        self.samples = []

        classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        for cls_name in classes:
            class_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(class_dir):
                continue
            for fname in os.listdir(class_dir):
                if fname.lower().endswith('.jpg'):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        # Читаем большое изображение в grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read the image: {img_path}")

        orig_h, orig_w = img.shape[:2]

        # Масштабируем для поиска контуров
        scale_factor = self.downscale_width / orig_w
        downscale_h = int(orig_h * scale_factor)
        img_small = cv2.resize(img, (self.downscale_width, downscale_h), interpolation=cv2.INTER_AREA)

        center_x_small, center_y_small = self.downscale_width // 2, downscale_h // 2

        # OTSU бинаризация
        _, img_bin = cv2.threshold(img_small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Инверсия
        img_bin = 255 - img_bin

        # Находим самый центрированный контур в уменьшенном изображении
        closest_contour_small = find_closest_contour_to_center(img_bin, center_x_small, center_y_small)
        if closest_contour_small is None:
            # Если не нашли контур, вернём просто пустое преобразование
            # Или можно бросить исключение
            raise ValueError("No contours found!")

        # Масштабируем контур обратно
        closest_contour = np.array([[[int(pt[0][0] / scale_factor), int(pt[0][1] / scale_factor)]]
                                    for pt in closest_contour_small])

        # Обрезаем и ресайзим
        obj_img = crop_and_resize_contour(img, closest_contour, output_size=448)

        # Превращаем в PIL
        obj_img_pil = Image.fromarray(obj_img)

        if self.transform:
            obj_tensor = self.transform(obj_img_pil)
        else:
            # Если вдруг не задан transform, просто ToTensor
            obj_tensor = T.ToTensor()(obj_img_pil)

        return obj_tensor, label