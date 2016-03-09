from drafterANN import *

epochs = 100
offset = 225 * 3000

print(Match.select().where(Match.seq_num > 1770445525).count())

ann = DotoAnn()

for epoch in range(epochs):
    batch_xs = np.zeros((epoch_size, n_input), np.int)
    batch_ys = np.zeros((epoch_size, n_out), np.int)
    i=0
    for m in Match.select().where(Match.seq_num > 1770445525).offset(epoch_size*epoch + offset).limit(epoch_size):
        for h in m.radiant_heroes.split(","):
            batch_xs[i][int(h)] = 1
        for h in m.dire_heroes.split(","):
            batch_xs[i][int(h) + max_heroes] = 1
        batch_ys[i][not m.radiant_win] = 1
        i+=1
    ann.train(batch_xs, batch_ys)
    
    if (epoch%20==0):
        print("Epoch:", epoch, "Accuracy:", ann.test_accuracy())
        
print("Done")
print("Accuracy:", ann.test_accuracy())
ann.save()
sess.close()