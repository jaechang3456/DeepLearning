# DeepLearning
### 분류의 문제에서, 사용에 사용할 loss 함수 선택 방법 : 
#### 분류의 문제에서 2가지의 분류 즉, 0과 1로 나뉠수 있는 분류문제의 loss함수는 binary_crossentropy를 사용하고, 3개 이상의 분류 문제에서 loss 함수는 categorical_crossentropy 함수를 사용한다. 또한 결과값이 실수가아닌 정수로 출력된다면, 앞에 sparse를 붙여, sparse_categorical_crossentropy를 사용한다.
ex) model.compile(optimizer='', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']

### 러닝커브 함수 코드와 트레이닝 어큐러시, 밸리데이션 어큐러시를 통한 오버피팅 
