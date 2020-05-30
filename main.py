from subj_sent.models.model import Model
from subj_sent.models.mlp import MLP

model = MLP(1)
#model.load('opa')
X = [13,10,13,10,13,10,13,10,13,10]
y = [0,1,0,1,0,1,0,1,0,1]
history = model.train(X, y, epochs = 5, batch_size = 2, validation_split = 0.1)
print(type(history))
print(model.evaluate(X,y))
print(model.predict(X))
model.save('jooj')