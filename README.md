# Minigo
A minimalist Go/Weiqi/Baduk engine using supervised deep learning

# How strong is it?
It should be roughly FOX 1 dan

# How to install?
1. Download repository (model file is already there)
2. Install python dependencies
3. Run "python gtp.py" to interact with engine in terminal or use Sabaki GUI

# What neural net is used?
I trained a 10 residual block net with 128 convolutional filters from scratch 
using 2097422 samples for 5 epochs resulting in 41.03% policy
accuracy and 1.63 MSE value on unseen positions. I used Google Colab
free T4 GPU runtime. Training took about 5 hours in total. For comparison
on Intel Core i5 4460 it would take around 92 hours.

# How to train your own net
1. Download games.txt from release page
2. Use "build_dataset.py" to build training data
3. Run "python train.py <device> <batch_size> <learning_rate> <data_file>"
This would run 1 epoch on given data, I was using 20 chunks ~100000 samples
each, stored on google drive, if you have enough RAM and GPU power just make
a single chunk of at least 1M samples (~5GB) and train it at least 5 times.
Feel free to use "python accuracy.py <model> <train_data> <val_data>" to measure
model quality.
