# pycaffe-yolo
YOLO reimplement in caffe, written with python layer.
##Usage
###Complie
```
git clone --recursive https://github.com/Zehaos/pycaffe-yolo
```
```
cd caffe-yolo
```
```
make -j8
make pycaffe
```
###Train
**Create trainning list**
```
cd ../tools
python create_voc_data_list.py
```
**Train**
```
PYTHONPATH=../lib ./train_googlenet.sh
```
###Demo
```
./demo.sh
```

 ![image](https://github.com/Zehaos/pycaffe-yolo/demo/gnet_iter10000.png)
 ![image](https://github.com/Zehaos/pycaffe-yolo/demo/gnet_iter30000.png)
 ![image](https://github.com/Zehaos/pycaffe-yolo/demo/gnet_iter60000.png)