from drafterANN import *

batch_size = 64
offset = 0

epoch_size = Match.select().where(Match.seq_num > 0).count()
iterations = epoch_size // batch_size

print("Epoch Size:", epoch_size)
print("Iterations", iterations)

ann = DotoAnn()

try:
    while True:
        for iteration in range(iterations):
            batch_xs = np.zeros((batch_size, n_input), np.int)
            batch_ys = np.zeros((batch_size, n_out), np.int)
            i=0
            for m in Match.select().where(Match.seq_num > 0).offset(batch_size*iteration + offset).limit(batch_size):
                for h in m.radiant_heroes.split(","):
                    batch_xs[i][int(h)] = 1
                for h in m.dire_heroes.split(","):
                    batch_xs[i][int(h) + max_heroes] = 1
                batch_ys[i][0 if m.radiant_win else 1] = 1
                i+=1
            ann.train(batch_xs, batch_ys)

            if (iteration%200==0):
                print("Iteration:", iteration, iteration / iterations, "Validation Accuracy:", ann.test_accuracy(), "Test Accuracy:", ann.test_accuracy(batch_xs, batch_ys))
                ann.save()

except BaseException as e:
    raise e    
finally:
    print("Done")
    print("Accuracy:", ann.test_accuracy())
    ann.save()
    db.close()
    sess.close()
