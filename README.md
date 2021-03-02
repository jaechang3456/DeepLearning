# DeepLearning
### 분류의 문제에서, 사용에 사용할 loss 함수 선택 방법 : 
#### 분류의 문제에서 2가지의 분류 즉, 0과 1로 나뉠수 있는 분류문제의 loss함수는 binary_crossentropy를 사용하고, 3개 이상의 분류 문제에서 loss 함수는 categorical_crossentropy 함수를 사용한다. 또한 결과값이 실수가아닌 정수로 출력된다면, 앞에 sparse를 붙여, sparse_categorical_crossentropy를 사용한다.
ex) model.compile(optimizer='', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']

### 러닝커브 함수 코드와 트레이닝 어큐러시, 밸리데이션 어큐러시를 통한 오버피팅 
#### 러닝커브 함수를 만드는 함수는 다음과 같다.
def learning_curve(history, epoch) :

  plt.figure(figsize=(10,5))
  정확도 차트
  epoch_range = np.arange(1, epoch +1)

  plt.subplot(1, 2, 1)

  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Model Accuracy')
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.legend(['Train', 'Val'])
  plt.show()

  loss 차트
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
