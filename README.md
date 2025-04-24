# SAR-NeRF-tensorflow
## 使用准备
解压dataset下angle_10_train_128.zip文件
解压out下angle_10_train_128.zip文件

## 训练
python run_nerf.py --datadir ./dataset/angle_10_train_128 --scale 1.0

## 生成
python test.py --datadir ./dataset/angle_10_train_128 --scale 1.0
