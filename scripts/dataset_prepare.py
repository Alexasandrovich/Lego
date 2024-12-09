import os
import random
import shutil
from collections import defaultdict


def split_dataset(
        src_dir: str,
        dst_dir: str,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42
):
    random.seed(seed)
    # Собираем все файлы
    all_files = [f for f in os.listdir(src_dir) if f.lower().endswith('.jpg')]

    # Группируем файлы по классам
    class_groups = defaultdict(list)
    for f in all_files:
        # Предполагаем имена вида classId-instanceId.JPG
        class_id = f.split('-')[0]
        class_groups[class_id].append(f)

    # Создаём папки назначения
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dst_dir, split), exist_ok=True)

    # Делим на тройку сплитов
    for class_id, files in class_groups.items():
        random.shuffle(files)
        total = len(files)
        train_count = int(total * train_ratio)
        val_count = int(total * val_ratio)
        test_count = total - train_count - val_count

        train_files = files[:train_count]
        val_files = files[train_count:train_count + val_count]
        test_files = files[train_count + val_count:]

        # Создаём директории для каждого класса
        for split, split_files in [('train', train_files), ('val', val_files), ('test', test_files)]:
            class_dir = os.path.join(dst_dir, split, class_id)
            os.makedirs(class_dir, exist_ok=True)
            for f in split_files:
                shutil.copy(os.path.join(src_dir, f), os.path.join(class_dir, f))


if __name__ == "__main__":
    split_dataset(
        src_dir='/data/alex/LEGO_dataset/full',
        dst_dir='/data/alex/LEGO_dataset/spilt',
        train_ratio=0.9,
        val_ratio=0.1,
        test_ratio=0.0
    )