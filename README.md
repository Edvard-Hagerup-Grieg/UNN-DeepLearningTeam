# Лабораторные работы по курсу "Глубокое обучение"
## Содержание
+ [Постановка задачи](#Task)
+ [Набор данных](#Dataset)
+ [Метрика качества](#Metric)
+ [Формат хранения данных](#Format)


## <a name="Task"></a>	Постановка задачи
Пусть <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;X&space;\in&space;\mathbb{R}^{N=h*w}"></a> - множество объеков (входов), где каждый объект описывает изображение размером <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;h*w"></a>, а <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y\in&space;\mathbb{R}^{M}"></a> - множество меток (выходов) соответствующих этим изображениям, где <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;M"></a> - количество классов, к которым каждое изображение может относиться. Для каждого элемента множества <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;Y"></a> справедливо  

<a><img src="https://latex.codecogs.com/gif.latex?\inline&space;\forall&space;y\in&space;Y:\sum_{i=1}^{M}{y^{(i)}}&space;=&space;1"></a>

Для набора данных <a><img src="https://latex.codecogs.com/gif.latex?\inline&space;\left&space;\{&space;\left&space;(&space;x^{(i)},&space;y^{(i)}&space;\right&space;):x^{(i)}\in&space;X,&space;y^{(i)}\in&space;Y,&space;i&space;=&space;1,...,L\right&space;\}"></a> предполагается подбор архитекрур [полносвязной](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/tree/master/lab2) и [сверточной](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/tree/master/lab3) нейронных сетей для решения задачи классификации, их обучение и тестирование, [начальная настройка весов](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/tree/master/lab4), а так же реализация [переноса знаний](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/tree/master/lab5).


## <a name="Dataset"></a>	Набор данных
В качестве исходных данных используется набор данных [Fashion MNIST](https://www.kaggle.com/zalando-research/fashionmnist), содержащий 70 000 изображений одежды 10 разных категорий:

| Класс | Представленность в наборе |
|:-----:|:-------------------------:|
|Футболка / топ | 7000 |
|         Шорты | 7000 |
|        Свитер | 7000 | 
|        Платье | 7000 |
|          Плащ | 7000 |
|       Сандали | 7000 |
|       Рубашка | 7000 |
|     Кроссовок | 7000 |
|         Сумка | 7000 | 
|       Ботинок | 7000 |

Все классы взаимоисключающие и сбалансированные.


## <a name="Metric"></a>	Метрика качества
При обучении модели и при ее тестировании фиксируются точность:

<a><img src="https://latex.codecogs.com/gif.latex?\inline&space;Accuracy&space;=&space;\frac{TP&space;&plus;&space;TN}{P&space;&plus;&space;N}"></a>

и ошибка классификации:

<a><img src="https://latex.codecogs.com/gif.latex?\inline&space;Error&space;=&space;\frac{FP&space;&plus;&space;FN}{P&space;&plus;&space;N}"></a>.


## <a name="Format"></a>	Формат хранения данных
Используемая при выполнении лабораторных работ библиотека Keras, позволяет автоматически загружать стандартные наборы данных, в том числе и Fashion MNIST, и сохранять их в каталоге ~/.keras/datasets. Изображения в автоматически загруженном наборе данных представлены в оттенках серого и имеют разрешение 28x28 пикселей:
![Пример изображений из набора данных](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/general_report/images/data_example.png)
Поскольку каждый пиксель принимает значение в диапазоне от 0 до 255, изображения во всех дальнейших экспериментах будут нормироваться:
![Пример нормированного изображения](https://github.com/Edvard-Hagerup-Grieg/UNN-DeepLearningTeam/blob/general_report/images/data_norm_example.png)


Кроме того, метка для каждого изображения приводится к вектору длины 10 методом One-Hot Encoding:

| Номер изображения | Футболка / топ | Шорты | Свитер | Платье | Плащ | Сандали | Рубашка | Кроссовок | Сумка | Ботинок |
|:-----------------:|:--------------:|:-----:|:------:|:------:|:----:|:-------:|:-------:|:---------:|:-----:|:-------:|
|  1   |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  1  |
|  2   |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  | 
|  3   |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |
| ...  | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |
|69 999|  0  |  0  |  0  |  0  |  0  |  1  |  0  |  0  |  0  |  0  |
|70 000|  0  |  1  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |  0  |


Следует отметить, что при разбиении на обучающий (60 000 изображений) и тестовый (10 000 изображений) наборы сохраняется сбалансированность классов:

| Класс | Представленность в тренировочном наборе | Представленность в тестовом наборе |
|:-----:|:---------------------------------------:|:----------------------------------:|
|  Футболка / топ | 6000 | 1000 |
|           Шорты | 6000 | 1000 |
|          Свитер | 6000 | 1000 |
|          Платье | 6000 | 1000 |
|            Плащ | 6000 | 1000 |
|         Сандали | 6000 | 1000 |
|         Рубашка | 6000 | 1000 |
|       Кроссовок | 6000 | 1000 |
|           Сумка | 6000 | 1000 |
|         Ботинок | 6000 | 1000 |
