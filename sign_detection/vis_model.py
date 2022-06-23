import visualkeras

from collections import defaultdict
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=(30,30,3)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# Second Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=3, activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
# Flattening the layer and adding Dense Layer
model.add(Flatten())
model.add(Dense(units=32, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.summary()

color_map = defaultdict(dict)
color_map[Conv2D]['fill'] = '#476A6F'
color_map[Dropout]['fill'] = '#519E8A'
color_map[MaxPool2D]['fill'] = '#7EB09B'
color_map[Dense]['fill'] = '#C5C9A4'
color_map[Flatten]['fill'] = '#ECBEB4'

visualkeras.layered_view(model, legend=True, color_map=color_map, spacing=30, to_file='output.png').show() # write and show
visualkeras.graph_view(model, to_file='output_cnn.png').show() # write and show
