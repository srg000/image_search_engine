import os
import glob
import random
import shutil
from PIL import Image

"""对所有图片进行RGB转化，并且统一调整到一致大小，但不让图片发生变形或扭曲，划分了训练集和测试集"""

if __name__ == '__main__':
    test_split_ratio = 0.05  # 以95:5的比例将数据分到train和test
    desired_size = 128  # 图片缩放后的统一大小
    raw_path = "./dataset_fruit_veg/raw"
    output_train_dir = "./dataset_fruit_veg/train"
    output_test_dir = "./dataset_fruit_veg/test"

    dirs = glob.glob(os.path.join(raw_path, '*'))  # 匹配raw_path文件下的所有文件，包括文件夹
    dirs = [d for d in dirs if os.path.isdir(d)]  # 只保留文件夹

    print(f'Totally {len(dirs)}classes:{dirs}')

    for path in dirs:
        # 对每个类别单独处理
        path = path.split('\\')[-1]
        # 创建test和train文件夹
        os.makedirs(f'{output_train_dir}/{path}', exist_ok=True)
        os.makedirs(f'{output_test_dir}/{path}', exist_ok=True)
        # 读取raw文件夹下所有类型的图片
        files = glob.glob(os.path.join(raw_path, path, '*.jpg'))
        files += glob.glob(os.path.join(raw_path, path, '*.JPG'))
        files += glob.glob(os.path.join(raw_path, path, '*.png'))
        # 打乱文件
        random.shuffle(files)
        # 找到训练集和测试集的边界
        boundary = int(len(files) * test_split_ratio)

        # 对每个图片进行预处理，并根据比例放入到对应的文件夹中
        for i, file in enumerate(files):
            # 将图像转换为RGB三通道格式
            img = Image.open(file).convert('RGB')
            old_size = img.size  # old size[0]is in (width,height)format
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])  # 对宽和高等比例缩放
            # 将图片resize新的大小比例
            im = img.resize(new_size, Image.Resampling.LANCZOS)
            new_im = Image.new("RGB", (desired_size, desired_size))
            new_im.paste(im, ((desired_size - new_size[0]) // 2,
                              (desired_size - new_size[1]) // 2))
            assert new_im.mode == 'RGB'
            if i < boundary:
                new_im.save(os.path.join(f'{output_test_dir}/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))
            else:
                new_im.save(os.path.join(f'{output_train_dir}/{path}', file.split('\\')[-1].split('.')[0] + '.jpg'))
    test_files = glob.glob(os.path.join(output_test_dir, '*''*.jpg'))
    train_files = glob.glob(os.path.join(output_train_dir, '*.jpg'))

    print(f'Totally {len(train_files)}files for training')
    print(f'Totally {len(test_files)}files for test')