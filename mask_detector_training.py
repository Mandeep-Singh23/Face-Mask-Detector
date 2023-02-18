#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
from imutils import paths
import matplotlib.pyplot as plt
import os

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report


# In[2]:


directory = r"D:\Projects\Deep Learning\Face Mask Detection\Face Mask Detector Dataset\dataset"
categories = ['with_mask', 'without_mask']


# In[24]:


data = []
label = []


# In[25]:


for category in categories:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)
        
        data.append(image)
        label.append(category)


# In[26]:


lb = LabelBinarizer()
label = lb.fit_transform(label)
label = to_categorical(label)


# In[30]:


data = np.array(data, dtype = 'float32')
label = np.array(label)


# In[32]:


X_train, X_test, y_train, y_test = train_test_split(data, label, test_size = 0.2, 
                                                    stratify= label, random_state = 42)


# In[34]:


img_aug = ImageDataGenerator(rotation_range = 20, zoom_range = .2, width_shift_range = .2,
                            height_shift_range = 0.2, shear_range = 0.2, horizontal_flip = True,
                            fill_mode = 'nearest')


# In[38]:


base_model = MobileNetV2(weights='imagenet', include_top = False,
                         input_tensor = Input(shape = (224,224,3)))


# In[40]:


main_model = base_model.output
main_model = AveragePooling2D(pool_size = (7,7))(main_model)
main_model = Flatten(name = 'Flatter_layer')(main_model)
main_model = Dense(128, activation = 'relu')(main_model)
main_model = Dropout(0.5)(main_model)
main_model = Dense(2, activation = 'softmax')(main_model)


# In[41]:


model = Model(inputs = base_model.input, outputs = main_model)


# In[43]:


for layer in base_model.layers:
    layer.trainable = False


# In[44]:


initial_lr = 1e-4
epochs = 20
BatchSize = 32


# In[45]:


opt = Adam( learning_rate = initial_lr, decay = initial_lr/epochs)


# In[47]:


model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])


# In[49]:


training = model.fit(img_aug.flow(X_train, y_train, batch_size = BatchSize),
                    steps_per_epoch = len(X_train)//BatchSize,
                    validation_data = (X_test, y_test),
                    validation_steps = len(X_test)//BatchSize,
                    epochs = epochs)


# In[50]:


pred = model.predict(X_test, batch_size = BatchSize)
pred = np.argmax(pred, axis = 1)


# In[52]:


print(classification_report(y_test.argmax(axis = 1), pred, target_names = lb.classes_))


# In[54]:


model.save('Face_Mask_Detector.model',save_format = 'h5')


# In[58]:



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), training.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), training.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), training.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, epochs), training.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="center right")
plt.savefig("Training Loss and Accuracy.png")


# In[ ]:




