# DeepLearning
### 분류의 문제에서, 사용에 사용할 유효성 함수 선택 방법 : binary_crossentropy



model.compile(optimizer='adam',
              loss = 'categorical_crossentropy', # 실수일 경우 앞에 sparse를 빼고 사용함
              metrics = ['accuracy'])
