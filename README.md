# FastBurger BA - Pipeline Base de Entrenamiento en PyTorch

## Objetivo del proyecto

Este proyecto corresponde al primer checkpoint del proyecto integrador de Deep Learning.

El objetivo principal de esta entrega es construir una infraestructura técnica inicial que permita entrenar, validar y monitorear un clasificador base utilizando PyTorch.

En esta primera etapa no se busca maximizar el desempeño del modelo, sino validar que el pipeline completo funcione correctamente: carga de datos, preparación del texto, vectorización, definición de una arquitectura base, entrenamiento, validación y seguimiento de métricas.

## Contexto del dataset

Se trabajó con un dataset de opiniones de clientes de una cadena ficticia de comida rápida llamada **FastBurger BA**.

El dataset contiene reseñas asociadas a distintas sucursales, junto con una clasificación de sentimiento derivada de la calificación otorgada por los clientes.

Las principales variables utilizadas fueron:

- `resena_bruta`: texto original de la opinión del cliente.
- `sentimiento`: categoría textual del sentimiento.
- `label_sentimiento`: codificación numérica de la categoría objetivo.
- `sucursal`: sucursal asociada a la reseña.
- `barrio`: zona o ubicación asociada.
- `tema_principal`: temática dominante de la reseña.

La variable objetivo del modelo fue `label_sentimiento`, codificada de la siguiente manera:

- 0: negativo
- 1: neutro
- 2: positivo

## Preprocesamiento inicial

Para esta primera versión del pipeline se aplicó una limpieza ligera del texto, orientada a reducir ruido básico sin modificar en profundidad el contenido semántico de las reseñas.

El procesamiento incluyó:

- Conversión del texto a minúsculas.
- Eliminación de caracteres especiales.
- Normalización de espacios.
- Uso de una versión limpia de la reseña para la vectorización.

Este preprocesamiento permitió construir una primera representación textual adecuada para validar el flujo técnico del modelo.

## Representación del texto

Se utilizó `TfidfVectorizer` para transformar las reseñas en una representación numérica compatible con PyTorch.

Esta técnica permitió convertir el texto en una matriz de características basada en la importancia relativa de los términos dentro del corpus.

La configuración utilizada fue:

- Vectorizador: `TfidfVectorizer`
- Máximo de características: `2000`
- Variable de entrada: reseña limpia
- Variable objetivo: `label_sentimiento`

## Arquitectura del modelo

Se implementó una red neuronal multicapa simple utilizando `torch.nn.Module`.

La arquitectura utilizada fue un MLP básico compuesto por:

- Capa de entrada compatible con la dimensión del vector TF-IDF.
- Capa oculta de 128 neuronas.
- Activación ReLU.
- Capa oculta de 64 neuronas.
- Activación ReLU.
- Capa de salida de 3 neuronas, correspondiente a las tres clases de sentimiento.

La arquitectura fue definida como línea base para validar el funcionamiento del pipeline de entrenamiento y evaluación.

## Configuración de entrenamiento

La configuración inicial del entrenamiento fue:

- Framework: PyTorch
- Función de pérdida: CrossEntropyLoss
- Optimizador: Adam
- Learning rate: 0.001
- Épocas: 20
- Métrica principal: Accuracy
- División de datos: entrenamiento y validación

Además, se configuró la detección automática del dispositivo de ejecución:

- CUDA, si se encontraba disponible.
- MPS, si se encontraba disponible.
- CPU, en caso contrario.

También se fijaron semillas de aleatoriedad para favorecer la reproducibilidad del experimento.

## Pipeline de entrenamiento

Durante cada época se ejecutó el ciclo completo de entrenamiento:

1. Forward pass.
2. Cálculo de la pérdida.
3. Reinicio de gradientes mediante `optimizer.zero_grad()`.
4. Backward pass mediante `loss.backward()`.
5. Actualización de parámetros mediante `optimizer.step()`.
6. Evaluación sobre el conjunto de validación.

Este flujo permitió validar correctamente el uso de `torch.autograd` y del optimizador Adam.

## Evaluación y resultados

Durante el entrenamiento se monitorearon:

- Pérdida de entrenamiento.
- Pérdida de validación.
- Accuracy sobre validación.

El modelo alcanzó una accuracy final de validación cercana al **64.72%**.

La evolución de la función de pérdida mostró una tendencia descendente, lo que indica que el modelo logró ajustar sus parámetros y aprender patrones presentes en las reseñas.

La pérdida de validación acompañó razonablemente la evolución de la pérdida de entrenamiento, sin evidencias claras de sobreajuste en esta primera etapa.

## Interpretación del resultado

El resultado obtenido demuestra que incluso una arquitectura simple basada en TF-IDF y una red neuronal multicapa puede capturar información relevante para clasificar el sentimiento de reseñas de clientes.

Sin embargo, este modelo debe interpretarse como una primera línea base técnica. Su principal valor dentro del proyecto es validar la infraestructura de entrenamiento y evaluación, más que alcanzar el máximo desempeño posible.

Este checkpoint permite contar con una base funcional sobre la cual incorporar mejoras en futuras etapas, tales como:

- Limpieza textual más avanzada.
- Lematización.
- Eliminación de stopwords.
- Embeddings.
- Modelos recurrentes.
- Transformers.
- Métricas adicionales como Precision, Recall y F1-score.

## Conclusión

En esta primera pre-entrega se logró construir un pipeline base completo y funcional en PyTorch.

El proyecto incluye carga y preparación de datos, vectorización del texto, definición de una arquitectura neuronal simple, entrenamiento con Adam, validación y seguimiento de métricas.

El principal resultado de esta etapa es contar con una infraestructura reproducible que servirá como punto de partida para futuras iteraciones del proyecto integrador de NLP y Deep Learning.
