The facial recognition data-set of a startup uas all jumbled up.
Some of their thousands of images have been rotated left, rotated right, and even turned upside-down!
Me, as a computer vision scientist, was called to solve this problem.

I have trained a ResNET50 model to classify all images in the 4 rotation types:
rotated left, rotated right, upside-down and upright

I have used pre-trained weigths loaded with tf.keras.applications options.
My time was running low, so, I have trained it just once (it lasted 15 minutes)!
But the results were promissing, I have got 95% of accuracy on training set.

Backlog items

1.   I had trained a CIFAR10 model for this problem but results were awful (75%)
     so, I discarded them.

2.   My predictions with ResNET model can be found in dropbox in the following link:
     https://www.dropbox.com/sh/vmlmxnvmuc2naet/AABoqAvJ8Eha73Cgogfi5USKa?dl=0
     it's name is "test.preds.csv".

     numpy array with pred images also can be found in dropbox with the name: test_array.npy

3.   Test set images with correct orientations also can be found in the dropbox
     https://www.dropbox.com/sh/vmlmxnvmuc2naet/AABoqAvJ8Eha73Cgogfi5USKa?dl=0
     it's name is "ziptest.rar".

4.   To run this code just type:

     python rotvision.py

     You will need:
     * Image files on "test" folder
     * Weigths and defintions of this ResNET model that can be found in the
       dropbox:
       https://www.dropbox.com/sh/vmlmxnvmuc2naet/AABoqAvJ8Eha73Cgogfi5USKa?dl=0
       Their names are: rotvision_trained_model.h5 and rotvision_trained_model.json
       You have to paste them at "saved_models" folder.

5.   The summary of my apprach is described above.

6.   For next steps I would let my machine crunch more number to train the model better.



