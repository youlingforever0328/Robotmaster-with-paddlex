import argparse
import os
# 选择使用0号卡
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import paddlex as pdx

def cal_sensitivies_file(model_dir, dataset, save_file):
    # 加载模型
    print(dataset)
    model = pdx.load_model(model_dir)

    # 定义验证所用的数据集
    eval_dataset = pdx.datasets.VOCDetection(
        data_dir=dataset,
        file_list=os.path.join(dataset, 'val_list.txt'),
        label_list=os.path.join(dataset, 'labels.txt'),
        transforms=model.eval_transforms)

    pdx.slim.cal_params_sensitivities(
        model, save_file, eval_dataset, batch_size=10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model_dir",
        default="/home/aistudio/output/yolov3_mobilenet/best_model",
        type=str,
        help="The model path.")
    parser.add_argument(
        "--dataset", default="/home/aistudio/dataset", type=str, help="The model path.")
    parser.add_argument(
        "--save_file",
        default="/home/aistudio/sensitivities.data",
        type=str,
        help="The sensitivities file path.")

    args = parser.parse_args()
    cal_sensitivies_file(args.model_dir, "/home/aistudio/dataset", args.save_file)