# Reconocimiento facial mediante Transfer Learning con CelebA

## Descripción

Este proyecto implementa un experimento de transferencia de aprendizaje para clasificación facial utilizando una red neuronal convolucional (CNN) entrenada inicialmente con el conjunto de datos **CelebA**.

El objetivo es reutilizar las características faciales aprendidas por la CNN para construir un clasificador binario capaz de distinguir entre dos clases:

* `yo`: imágenes propias.
* `otros`: imágenes de otras personas provenientes de CelebA.

El experimento incluye una etapa de entrenamiento con las capas convolucionales congeladas y una segunda etapa de **fine-tuning**, en la que se descongela la última capa convolucional.

## Tecnologías utilizadas

* Python
* TensorFlow 2.20.0
* TensorFlow Datasets 4.9.10
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab

## Dataset CelebA

CelebA contiene 202,599 imágenes de rostros y 40 atributos faciales binarios.

Para el entrenamiento inicial de la CNN se utilizaron las particiones proporcionadas por el dataset:

| Conjunto      |    Imágenes |
| ------------- | ----------: |
| Entrenamiento |     162,770 |
| Validación    |      19,867 |
| Prueba        |      19,962 |
| **Total**     | **202,599** |

Las imágenes fueron redimensionadas a `128 × 128` píxeles y normalizadas al intervalo `[0,1]`.

## Arquitectura

La CNN utilizada para aprender los atributos de CelebA está formada por:

```text
Input (128, 128, 3)
        ↓
Conv2D (32 filtros)
        ↓
MaxPooling2D
        ↓
Conv2D (64 filtros)
        ↓
MaxPooling2D
        ↓
Conv2D (128 filtros)
        ↓
GlobalAveragePooling2D
        ↓
Dense (128)
        ↓
Dense (40, sigmoid)
```

La salida de 40 neuronas permite realizar la predicción multietiqueta de los atributos faciales de CelebA.

## Transfer Learning

Después de entrenar la CNN con CelebA, se eliminaron las capas correspondientes al clasificador original de 40 atributos y se reutilizaron las capas convolucionales como extractor de características.

Sobre este extractor se incorporó:

```text
GlobalAveragePooling2D
        ↓
Dense (128, ReLU)
        ↓
Dense (1, sigmoid)
```

La salida binaria representa la clasificación entre `yo` y `otros`.

## Aumento de datos

Debido al reducido número de imágenes propias, se utilizó `ImageDataGenerator` exclusivamente sobre el conjunto de entrenamiento.

Se aplicaron:

* Rotación de hasta ±15°.
* Desplazamientos horizontales y verticales de hasta 10%.
* Zoom de hasta 10%.
* Reflexión horizontal.
* Normalización de los píxeles.

Las imágenes de validación no fueron aumentadas.

## Dataset para clasificación facial

Se utilizaron 31 imágenes propias y 100 imágenes de CelebA para la clase `otros`.

| Clase     | Entrenamiento | Validación |   Total |
| --------- | ------------: | ---------: | ------: |
| yo        |            25 |          6 |      31 |
| otros     |            80 |         20 |     100 |
| **Total** |       **105** |     **26** | **131** |

La división se realizó utilizando una semilla aleatoria de 42.

## Resultados

### Etapa b — Capas convolucionales congeladas

Durante esta etapa solamente se entrenó el nuevo clasificador binario.

| Métrica             | Entrenamiento | Validación |
| ------------------- | ------------: | ---------: |
| Binary Accuracy     |        0.9429 |     1.0000 |
| Binary Crossentropy |        0.2548 |     0.1978 |

### Etapa c — Fine-tuning

Para realizar el ajuste fino se descongeló únicamente la última capa convolucional y se utilizó Adam con una tasa de aprendizaje de `1e-5`.

| Métrica             | Entrenamiento | Validación |
| ------------------- | ------------: | ---------: |
| Binary Accuracy     |        0.9143 |     1.0000 |
| Binary Crossentropy |        0.2532 |     0.1938 |

El modelo final contiene 109,889 parámetros, de los cuales 90,497 son entrenables y 19,392 no entrenables.

### Matriz de confusión

La matriz de confusión obtenida sobre las 26 imágenes de validación fue:

```text
                 Predicción
              otros       yo
Real otros      20         0
     yo          0         6
```

No se presentaron errores de clasificación en las imágenes de validación utilizadas.

| Clase | Precision | Recall | F1-score |
| ----- | --------: | -----: | -------: |
| otros |    1.0000 | 1.0000 |   1.0000 |
| yo    |    1.0000 | 1.0000 |   1.0000 |

## Interpretación

La transferencia de aprendizaje permitió reutilizar las características aprendidas con CelebA para una nueva tarea de clasificación binaria.

El clasificador obtuvo una accuracy de validación de 100% desde las últimas épocas de la etapa b. El fine-tuning mantuvo esta accuracy y produjo una reducción pequeña de la pérdida de validación, de 0.1978 a 0.1938.

Por lo tanto, en este experimento las características aprendidas con CelebA resultaron útiles para la tarea específica de clasificación facial, mientras que el fine-tuning permitió realizar ajustes adicionales sobre la representación aprendida.

## Limitaciones

Los resultados deben interpretarse considerando el tamaño reducido del conjunto utilizado para la clasificación facial. Particularmente, el conjunto de validación contiene solamente seis imágenes propias.

Por esta razón, el resultado de 100% de accuracy describe el comportamiento del modelo sobre las 26 imágenes de validación disponibles y no permite concluir que el sistema tenga una precisión de 100% ante fotografías nuevas o en condiciones generales.

Factores como iluminación, orientación del rostro, expresión facial, distancia, resolución y fondo podrían modificar el desempeño del modelo.

El experimento tiene fines académicos y no representa un sistema de reconocimiento facial listo para aplicaciones reales.

## Privacidad

Las imágenes personales utilizadas durante el experimento **no forman parte de este repositorio**. Los datos personales se mantienen almacenados de manera local o en Google Drive.

El repositorio contiene únicamente el código, el notebook y los resultados necesarios para documentar el experimento.

## Estructura del repositorio

```text
tarea-celeba/
│
├── README.md
├── CelebA_transfer_learning.ipynb
│
└── resultados/
    ├── curva_accuracy.png
    ├── curva_loss.png
    └── matriz_confusion.png
```
