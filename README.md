# Код към дипломна работа 

## Основни файлове
data_preparation.py - скрипт за подговтяне на данните за трениране и тестване  
model.py - дефиниция на архитектурата на модела  
train.py - скрипт за трениране на модела и запазването му в папка trained_model  
checkpoints - папка с запазени междинни състояния от тренирането  
convert_to_onnx.sh - конвертиране на запазения модел от папка trained_model към отворения формат onnx  
evaluate.py - скрипт за тестване на тренирания модел  
output - папка с изображения резултат от тестването на модела  
други - помощни файлове и функции, които се изпозват за изгаждане на модела  

## Данни
Използвани са 2 набора от данни - 300W и WFLW достъпни от линковете по-долу.  
Те трябва да бъдат разрахивирани в папка data, след което да се изпълни data_preparation.py

Набор от данни 300W: https://ibug.doc.ic.ac.uk/resources/300-W/
Набор от данни WFLW: https://wywu.github.io/projects/LAB/WFLW.html

## Използвани версии на библиотеки
Кода е разработван на Ubuntu 20.04, с използване на Docker контейнер nvcr.io/nvidia/tensorflow:21.05-tf2-py3 в който са налични основните използвани библиотеки  

Версии на включените в контейнера библиотеки:  
tensorrt==7.2.3.4  
tensorflow==2.4 
pandas==1.2.5  
pycuda==2021.1  
opencv-python==4.5.2.54  
opencv-python-headless==4.5.2.54  
onnx==1.9.0  
onnx-simplifier==0.3.6  
onnxconverter-common==1.8.1  
onnxoptimizer==0.2.6  
onnxruntime==1.8.0  
matplotlib==3.4.2  

За тестване на платформата Nvidia Jetson TX2 на нея е инсталиран пакета Nvidia JetPack 4.4 базиран на Ubuntu 18.04  
