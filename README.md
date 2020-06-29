# Lane Detection assignment
	Deep Learning algorithm implementation to automate the lane detection process from street images.

![Teaser Image](https://github.com/chandra411/Lane-Detection/blob/master/models/out.jpg)

## Installations
	* Platform: Python2 (generating tf_records code is not compactible with python3 except that everthing else will work fine with Python3)
	* (Optional) create python virtual environment and activate it
	* pip -r install requirments.txt
-Download [models](https://drive.google.com/drive/folders/1B3CYhD0oxkOcrXGXh7SRA1_dKIQgTAon?usp=sharing) and copy the items to models directory
## Training 
	* (Optional) Generate data augmentation 
		python augmentation.py --data_path={Give you data located path}
		* In the augmentation code I have taken few static things and used multiprocessing, please use it carefully and change it as per your data.
	* Generate TF_Records 
		python generate_tfrecords.py --image_dir={path to your input images} --label_dir={path to your label images} --tfr_dir=./tf_records
	* Training network
		python train.py --tf_record_pattren=./tf_records/train-?????-of-00002 --resnet_50_checkpoint_path=./models/res50/resnet_v2_50.ckpt --batch_size=16 --train_outs=./train_outs

## Infrence
	* Testing network with pretrained model
	python test.py --test_dir=./test_images --checkpoint=./models/model-185003 --out_dir=./test_out

## Note: Please look into the colab note book file (notebook.ipynb) which I used for training and testing in google colab



