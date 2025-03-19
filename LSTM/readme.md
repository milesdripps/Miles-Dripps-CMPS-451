log_trainings.csv contains different training settings. I trained the BiLSTM on the first 95% of the data,
before testing its prediction on the last 5%, or the last 6 months. I changed different parameters, but the
current ones in the code reflect the highest accuracy I could find, which was using a training step of 30 days,
seeing that 20 epochs and 100 neurons kept the time around 20 minutes with the best results. I noticed that training
with all of the parameters made the model worse, and training with just temp and dew was pretty good, since they are
similar in correlation. 
