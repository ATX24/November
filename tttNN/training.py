#Now, lets actually train a model using real data
import pandas as pd
import tictactoe 

df = pd.read_csv('Tic tac initial results.csv')
df = df.dropna()


move_1 = df['MOVE1']
move_2 = df['MOVE2']

for i, value in enumerate(list(move_2)):
    if value == '?':
        df = df.drop(i)

move_1 = df['MOVE1']
move_2 = df['MOVE2']

X_train = []
y_train = []


for move in list(move_1):
    if move == 0:
        X_train.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
    if move == 1:
        X_train.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
    if move == 2:
        X_train.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
    if move == 3:
        X_train.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
    if move == 4:
        X_train.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
    if move == 5:
        X_train.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
    if move == 6:
        X_train.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
    if move == 7:
        X_train.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
    if move == 8:
        X_train.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

for move in list(move_2):
    if move == '0':
        y_train.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
    if move == '1':
        y_train.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
    if move == '2':
        y_train.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
    if move == '3':
        y_train.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
    if move == '4':
        y_train.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
    if move == '5':
        y_train.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
    if move == '6':
        y_train.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
    if move == '7':
        y_train.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
    if move == '8':
        y_train.append([0, 0, 0, 0, 0, 0, 0, 0, 1])

    
    

print(len(X_train))
print(len(y_train))


    

neural_net = tictactoe.tictac()

neural_net.rundat(X_train, y_train, 1000)