import keras

# 資料預先處理
(train_img, train_label), (test_img, test_label) = keras.datasets.mnist.load_data()
x_train = train_img.reshape(len(train_img), train_img[0].size) / 255
y_train = keras.utils.to_categorical(train_label)
x_test = test_img.reshape(len(test_img), test_img[0].size) / 255
y_test = keras.utils.to_categorical(test_label)

# model 三步驟：(1) 設計網路 (2) 編譯網路 (3) 擬合網路權重
## step-1
model = keras.models.Sequential()
model.add(keras.layers.Dense(units=120, input_dim=784, activation='relu'))
model.add(keras.layers.Dense(units=120, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax'))
model.summary()

## step-2
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

## step-3
model.fit(x_train, y_train, epochs=20)

# 執行預測
loss, metrics_accuracy = model.evaluate(x=x_test, y=y_test)
print('loss={:.6f}, accuracy={:.2f}%'.format(loss, metrics_accuracy * 100))

## 執行結果
# loss=0.123097, accuracy=97.93%
