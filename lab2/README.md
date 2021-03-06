# Лабораторная работа №2
# Разработка полностью связанных нейронных сетей

+ [Входные данные](#Format_input)
+ [Архитектуры нейронных сетей](#NN_architecture)
+ [Результаты экспериментов](#Results)
+ [Описание программной реализации](#Description)


## <a name="Format_input"></a>	Входные данные
Загрузка и предобработка данных реализована в файле [dataset.py](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/master/lab2/dataset.py).


Для использования в качестве входного сигнала описанных ниже нейронных сетей, нормированные изображений 28x28 пикселей представляются в виде вектора длины 784.


## <a name="NN_architecture"></a>	Архитектуры нейронных сетей
В качестве архитектур использовались модели с 1-3 полносвязными скрытыми слоями. Количество нейронов варьировалось от 64 до 512
в каждом слое, функции активации, используемые на скрытых слоях: relu, сигмоид и линейная. На выходном слое всегда использовалась функция 
softmax, так как классов 10.

Используемые архитектуры сетей:

| Количество слоёв | Количество нейронов в слоях | Функция активации слоёв|
|:----------------:|:---------------------------:|:----------------------:|
| 1 | 256 | relu |
| 1 | 512 | relu |
| 2 | 512, 256 | relu |
| 3 | 512, 256, 64 | relu |
| 1 | 256 | linear |
| 1 | 512 | linear |
| 2 | 512, 256 | linear |
| 3 | 512, 256, 64 | linear |
| 1 | 256 | sigmoid |
| 1 | 512 | sigmoid |
| 2 | 512, 256 | sigmoid |
| 3 | 512, 256, 64 | sigmoid |

В качестве функции потерь использовалась перекрестная:
    
![](https://latex.codecogs.com/gif.latex?E%28w%29%3D-%5Csum%5Climits_%7Bj%3D1%7D%5EMy_j%5Cln%7Bu_j%7D)
    
где ![](https://latex.codecogs.com/gif.latex?y_j) – ожидаемый выход (метки),

![](https://latex.codecogs.com/gif.latex?u_j) – выход сети.

Качество сети оценивалось через точность - отношение количества меток, совпавших с предсказанными к числу примеров:

![](https://latex.codecogs.com/gif.latex?\frac{I(y_j=u_j)}{N},j=\overline{1,N})

## <a name="Description"></a>	Описание программной реализации

### [dataset.py](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/master/lab2/dataset.py) содержит методы для обработки входных данных:

+ load_dataset загружает набор данных, нормирует, приводит x к векторному виду, а y к one-hot кодированию

### [models.py](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/master/lab2/models.py) содержит методы для создания моделей:

+ build_dense_model_1 строит полносвязную модель в соответствии с параметрами из аргументов

+ generate_model_zoo создаёт список моделей разных архитектур для экспериментов

### [experiments.py](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/master/lab2/experiments.py) содержит методы для проведения экспериментов на моделях

+ save_history_img сохраняет график обучения

+ calculate_accuracy считает точность модели

+ train_models обучает список моделей. Возвращает модели с весами, которые показали лучшую точность на валидации

+ test_models тестирует список моделей

## <a name="Results"></a>	Результаты экспериментов

| Архитектура сети | Время обучения, эпохи | Качество решения, точность|
|:----------------:|:---------------------------:|:----------------------:|
| 256, relu | 150 | 0.8328 |
| 512, relu | 150 | 0.8355 |
| 512, 256, relu | 150 | 0.8472 |
| 512, 256, 64, relu | 150 | 0.8356 |
| 256, linear | 150 | 0.8302 |
| 512, linear | 150 | 0.8312 |
| 512, 256, linear | 150 | 0.8376 |
| 512, 256, 64, linear | 150 | 0.8405 |
| 256, sigmoid | 150 | 0.7691 |
| 512, sigmoid | 150 | 0.7744 |
| 512, 256, sigmoid | 150 | 0.7194 |
| 512, 256, 64, sigmoid | 150 | 0.5011 |

По результатам можно видеть, что худший результат имеют сети с функцией активации сигмоид, вне зависимости от количества слоёв и нейронов. Также, можно видеть, что relu показывает лучшие результаты, чем линейная функция активации на всех моделях, кроме архитектуры с тремя слоями.
