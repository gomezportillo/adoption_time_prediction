# Cambio el directorio de trabajo ----------------------------------------------
setwd("C:/Users/Pedro Manuel/Desktop/SIGE")
options(max.print = 999999)
set.seed(1234)

# Descargo el fichero .zip -----------------------------------------------------

download.file(
  "https://s3.us-east-2.amazonaws.com/pets-adoption/petfinder-adoption-prediction.zip",
  destfile = "petfinder-adoption-prediction.zip"
)
zip::unzip("petfinder-adoption-prediction.zip", exdir = "petfinder-adoption-prediction")

library(keras)

# Cargar y pre-procesar imágenes ----------------------------------------------
train_dir <- './petfinder-adoption-prediction_pruebas/train_images/'

# REF. https://tensorflow.rstudio.com/keras/reference/image_data_generator.html
# REF. https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
train_datagen <- image_data_generator(rescale=1/255, validation_split=0.2)

# REF. https://tensorflow.rstudio.com/keras/reference/flow_images_from_directory.html
train_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode='categorical',
  subset='training') # set as training data

test_generator <- flow_images_from_directory(
  directory = train_dir,
  generator = train_datagen,
  target_size = c(150, 150),   # (w, h) --> (150, 150)
  batch_size = 20,             # grupos de 20 imágenes
  class_mode='categorical',
  subset='validation') # set as validation data


# Creación del modelo ----------------------------------------------------------

## Definir arquitectura --------------------------------------------------------
## REF. https://tensorflow.rstudio.com/keras/articles/sequential_model.html
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32,  kernel_size = c(3, 3), activation = "relu", input_shape = c(150, 150, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64,  kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 5, activation = "softmax")

## Resumen del modelo ----------------------------------------------------------
summary(model)

## Compilar modelo --------------------------------------------------------------
# REF. https://tensorflow.rstudio.com/keras/reference/compile.html
model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 0.0001, decay = 1e-6),
  metrics = c("accuracy")
)

# Entrenamiento ----------------------------------------------------------------
# REF. https://tensorflow.rstudio.com/keras/reference/fit_generator.html
history <-  model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  validation_steps = 50,
  epochs = 15,
  verbose = 2,
  callbacks = list(
    # Guardar el mejor modelo tras cada época
    callback_model_checkpoint("./mejor_checkpoints.h5", save_best_only = TRUE),
    # Gardar los logs para tensorboard
    callback_tensorboard(log_dir = "./logs")
  ))

## Evaluar modelo --------------------------------------------------------------
# REF. https://tensorflow.rstudio.com/keras/reference/evaluate_generator.html
model %>% evaluate_generator(test_generator, steps = 25)

# Visualizar entrenamiento -----------------------------------------------------
plot(history)

# Ejecutar tensorboard ---------------------------------------------------------
tensorboard("./logs")

# Imprima la información de la versión sobre R , el SO y los paquetes usados ---
sessionInfo()

test_generator$filenames
predictions <- predict_generator(
  model,
  test_generator,
  steps = test_generator$n/test_generator$batch_size
)
for(i in 1:100) { print(predictions[i])}
