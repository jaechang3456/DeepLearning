# DeepLearning1
### 분류의 문제에서, 사용에 사용할 loss 함수 선택 방법 : 
#### 분류의 문제에서 2가지의 분류 즉, 0과 1로 나뉠수 있는 분류문제의 loss함수는 binary_crossentropy를 사용하고, 3개 이상의 분류 문제에서 loss 함수는 categorical_crossentropy 함수를 사용한다. 또한 결과값이 실수가아닌 정수로 출력된다면, 앞에 sparse를 붙여, sparse_categorical_crossentropy를 사용한다.
ex) model.compile(optimizer='adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']

### 러닝커브 함수 코드와 트레이닝 어큐러시, 밸리데이션 어큐러시를 통한 오버피팅 
## 러닝커브 함수를 만드는 함수는 다음과 같다.

def learning_curve(history, epoch):

  plt.figure(figsize=(10,5))
  
  epoch_range = np.arange(1, epoch +1)
  
  plt.subplot(1, 2, 1) 
  plt.plot(epoch_range, history.history['accuracy']) 
  plt.plot(epoch_range, history.history['val_accuracy']) 
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Val'])
  plt.show()

  plt.subplot(1, 2, 2)
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Model Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.legend(['Train', 'Val'])
  plt.show()


  
![다운로드](https://user-images.githubusercontent.com/78472987/109621966-eef40f80-7b7e-11eb-8831-c7c82f7e6c15.png)

  함수를 정의하고, 아래와 같이 호출시 위와 같은 그림이 나온다.
  learning_curve(history, 10)
러닝커브 함수를 사용하면, loss값과 accuracy값이 눈에 한번에 보여 분석하기 쉽다. 그래프를 보면 빨간선이 validation값이고, 파란선이 train값이다.
X축은 epoch인걸 알수있는데, 위 그래프에서는 epoch 4~5 사이에서부터 validation값과, train값의 loss와 accuracy차이가 증가함을 볼 수 있다.
이러한 상황을 오버피팅 되었다고 하는데, 너무 많은 데이터를 학습시키면 train값의 loss는 줄어들고, accuracy는 올라갈수 있어도, 새로 알게되는 값, 즉 우리가 예측해야 할 값을 제대로 예측하지 못하는 상황이 발생한다.
  
##콜백 함수 사용법

def train_mnist():
    
    class myCallback(tf.keras.callbacks.Callback) :
      def on_epoch_end(self, epoch, logs={}) :
          if(logs.get('accuracy') > 0.99 ) :
            print('\n"Reached 99% accuracy so cancelling training!"')
            self.model.stop_training = True
    my_cb = myCallback()
    
    mnist = tf.keras.datasets.mnist

    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1 , 28*28)
    X_test = X_test.reshape(-1, 28*28)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    model = tf.keras.models.Sequential([
        Dense(input_dim = 784, units=512, activation='relu'),
        Dense(units=512, activation='relu'),
        Dense(units=10, activation ='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # model fitting
    history = model.fit(
        X_train, y_train,
        epochs = 8,
        callbacks = [my_cb]
    )
    # model fitting

    return history.epoch, history.history['accuracy'][-1]
