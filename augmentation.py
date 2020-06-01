#python augmentation.py --data_path=./chandu 
import numpy as np
import cv2
import os
import random
import argparse
import copy
from multiprocessing import Pool
from multiprocessing import Process
import multiprocessing
import multiprocessing.pool
ap = argparse.ArgumentParser()
ap.add_argument("--data_path", required=True, help="path to data set")
args = vars(ap.parse_args())
Image_dir = "Images/"
mask_dir = "Labels/"
OUT_DIR = args["data_path"]
HEIGHT = 720
WIDTH = 1280

if not os.path.exists(OUT_DIR+Image_dir):
	os.mkdir(OUT_DIR+Image_dir)
if not os.path.exists(OUT_DIR+mask_dir):
        os.mkdir(OUT_DIR+mask_dir)

def left_translation(img, move):
	if len(img.shape) == 3:
		rows, cols, channels = img.shape
	else:
		rows, cols = img.shape
	for i in range(rows):
		for j in range(cols):
			if j < cols-move:
				img[i,j] = img[i,j+move]
			else:
				img[i,j] = img[i, cols-move-1]						
	return img
	
def right_translation(img, move):
	if len(img.shape) == 3:
		rows, cols, channels = img.shape
	else:
		rows, cols = img.shape	
	for i in range(rows):
		for j in range(cols-1,-1, -1):
			if j >= move:
				img[i,j] = img[i,j-move]
			else:	
				img[i,j] = img[i, move+1]		
	return img

def bottom_translation(img, move):	
	if len(img.shape) == 3:
		rows, cols, channels = img.shape
	else:
		rows, cols = img.shape	
	for i in range(rows-1,-1,-1):
		for j in range(cols):
			if i >= move:
				img[i,j] = img[i-move,j]
			else:	
				img[i,j] = img[move+1, j]
	return img	

def top_translation(img, move):	
	if len(img.shape) == 3:
		rows, cols, channels = img.shape
	else:
		rows, cols = img.shape	
	for i in range(rows):
		for j in range(cols):
			if i < rows-move:
				img[i,j] = img[i+move,j]
			else:
				img[i,j] = img[rows-move-1, j]
	return img

def points_left_translation(points, move):
	for i in range(len(points)):
		points[i][0] -= move	
	return points

def points_right_translation(points, move):
	for i in range(len(points)):
		points[i][0] += move	
	return points

def points_bottom_translation(points, move):
	for i in range(len(points)):
		points[i][1] += move	
	return points

def points_top_translation(points, move):
	for i in range(len(points)):
		points[i][1] -= move	
	return points

def flip_points(s_points, t_points, f_points):
	for i in range(len(s_points)):
		s_points[i][0] = POINTS_WIDTH-s_points[i][0]
	for i in range(len(t_points)):
		t_points[i][0] = POINTS_WIDTH-t_points[i][0]
	return [s_points, t_points, f_points]

def color_aug(img):
	shape = img.shape	
	if shape[2] == 4:
		r,g,b,a = cv2.split(img)
		gbr_img = cv2.merge((g,b,r,a))
		brg_img = cv2.merge((b,r,g,a))
		grb_img = cv2.merge((r,r,r,a))
	elif shape[2] == 3:
		r,g,b = cv2.split(img)
		gbr_img = cv2.merge((g,b,r))
		brg_img = cv2.merge((b,r,g))
		grb_img = cv2.merge((r,r,r))
	return gbr_img, brg_img, grb_img


class NoDaemonProcess(multiprocessing.Process):
# make 'daemon' attribute always return False
	def _get_daemon(self):
		return False
	def _set_daemon(self, value):
		pass
	daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
	Process = NoDaemonProcess


def Augmentation_points_images(prod_n):
	
	if os.path.exists(OUT_DIR+Image_dir+'c3tff_'+prod_n):
		print(prod_n+' already augmented!!!')
		return
	print(prod_n)
	prod_n = prod_n.split('.')[0]
	prod_img = cv2.imread(OUT_DIR+Image_dir+prod_n+'.jpg', cv2.IMREAD_UNCHANGED)
	prod_pgn_img = cv2.imread(OUT_DIR+mask_dir+prod_n+'.jpg',cv2.IMREAD_UNCHANGED)

	#resizing
	print(prod_img.shape,(WIDTH, HEIGHT))
	prod_img = cv2.resize(prod_img, (WIDTH, HEIGHT))
	prod_pgn_img = cv2.resize(prod_pgn_img,(WIDTH,HEIGHT))

	
	#keeping copies of images
	prod_img_orig, prod_pgn_img_orig = prod_img.copy(), prod_pgn_img.copy()

	cv2.imwrite(OUT_DIR+Image_dir+'zzzz_'+prod_n+'.jpg', prod_img)
	cv2.imwrite(OUT_DIR+mask_dir+'zzzz_'+prod_n+'.jpg',prod_pgn_img)
	

	cv2.imwrite(OUT_DIR+Image_dir+'zfff_'+prod_n+'.jpg', np.fliplr(prod_img))
	cv2.imwrite(OUT_DIR+mask_dir+'zfff_'+prod_n+'.jpg', np.fliplr(prod_pgn_img))

	#color_aug
	r_prod, g_prod, b_prod = color_aug(prod_img)	
	cv2.imwrite(OUT_DIR+Image_dir+'c1zzz_'+prod_n+'.jpg', r_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c1fff_'+prod_n+'.jpg', np.fliplr(r_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c2zzz_'+prod_n+'.jpg', g_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c2fff_'+prod_n+'.jpg', np.fliplr(g_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c3zzz_'+prod_n+'.jpg', b_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c3fff_'+prod_n+'.jpg', np.fliplr(b_prod))

	#pgn and points are kept constant incase of color aug
	for i in range(1, 4):
		os.system('cp '+OUT_DIR+mask_dir+'zzzz_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'zzz_'+prod_n+'.jpg')
		os.system('cp '+OUT_DIR+mask_dir+'zfff_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'fff_'+prod_n+'.jpg')

	###left translation
	move = random.randint(20,50)
	prod_img = left_translation(prod_img, move)
	prod_pgn_img = left_translation(prod_pgn_img,move)

	cv2.imwrite(OUT_DIR+Image_dir +'zlll_'+ prod_n+'.jpg', prod_img)
	cv2.imwrite(OUT_DIR+Image_dir  +'zlff_'+ prod_n+'.jpg', np.fliplr(prod_img))
	cv2.imwrite(OUT_DIR+mask_dir +'zlll_'+ prod_n+'.jpg', prod_pgn_img)
	cv2.imwrite(OUT_DIR+mask_dir +'zlff_'+ prod_n+'.jpg', np.fliplr(prod_pgn_img))
	
	r_prod, g_prod, b_prod = color_aug(prod_img)	
	cv2.imwrite(OUT_DIR+Image_dir+'c1lll_'+prod_n+'.jpg', r_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c1lff_'+prod_n+'.jpg', np.fliplr(r_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c2lll_'+prod_n+'.jpg', g_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c2lff_'+prod_n+'.jpg', np.fliplr(g_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c3lll_'+prod_n+'.jpg', b_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c3lff_'+prod_n+'.jpg', np.fliplr(b_prod))

	for i in range(1, 4):
	
		os.system('cp '+OUT_DIR+mask_dir+'zlll_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'lll_'+prod_n+'.jpg')
		os.system('cp '+OUT_DIR+mask_dir+'zlff_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'lff_'+prod_n+'.jpg')

	#right translation
	prod_img, prod_pgn_img = prod_img_orig.copy(), prod_pgn_img_orig.copy()

	move = random.randint(20,50)
	prod_img = right_translation(prod_img, move)
	prod_pgn_img = right_translation(prod_pgn_img,move)
	cv2.imwrite(OUT_DIR+Image_dir  +'zrrr_'+prod_n+'.jpg', prod_img)
	cv2.imwrite(OUT_DIR+Image_dir  +'zrff_'+ prod_n+'.jpg', np.fliplr(prod_img))
	cv2.imwrite(OUT_DIR+mask_dir  +'zrrr_'+ prod_n+'.jpg', prod_pgn_img)
	cv2.imwrite(OUT_DIR+mask_dir+'zrff_'+ prod_n+'.jpg', np.fliplr(prod_pgn_img))

	r_prod, g_prod, b_prod = color_aug(prod_img)	
	cv2.imwrite(OUT_DIR+Image_dir+'c1rrr_'+prod_n+'.jpg', r_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c1rff_'+prod_n+'.jpg', np.fliplr(r_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c2rrr_'+prod_n+'.jpg', g_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c2rff_'+prod_n+'.jpg', np.fliplr(g_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c3rrr_'+prod_n+'.jpg', b_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c3rff_'+prod_n+'.jpg', np.fliplr(b_prod))

	for i in range(1, 4):
		os.system('cp '+OUT_DIR+mask_dir+'zrrr_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'rrr_'+prod_n+'.jpg')
		os.system('cp '+OUT_DIR+mask_dir+'zrff_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'rff_'+prod_n+'.jpg')

	#bottom translation
	prod_img, prod_pgn_img = prod_img_orig.copy(), prod_pgn_img_orig.copy()
	
	move = random.randint(10,30)
	prod_img = bottom_translation(prod_img, move)
	prod_pgn_img = bottom_translation(prod_pgn_img,move)
	cv2.imwrite(OUT_DIR+Image_dir  +'zbbb_'+ prod_n+'.jpg', prod_img)
	cv2.imwrite(OUT_DIR+Image_dir  +'zbff_'+ prod_n+'.jpg', np.fliplr(prod_img))
	
	cv2.imwrite(OUT_DIR+mask_dir +'zbbb_'+ prod_n+'.jpg', prod_pgn_img)
	cv2.imwrite(OUT_DIR+mask_dir +'zbff_'+ prod_n+'.jpg', np.fliplr(prod_pgn_img))

	
	r_prod, g_prod, b_prod = color_aug(prod_img)	
	cv2.imwrite(OUT_DIR+Image_dir+'c1bbb_'+prod_n+'.jpg', r_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c1bff_'+prod_n+'.jpg', np.fliplr(r_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c2bbb_'+prod_n+'.jpg', g_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c2bff_'+prod_n+'.jpg', np.fliplr(g_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c3bbb_'+prod_n+'.jpg', b_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c3bff_'+prod_n+'.jpg', np.fliplr(b_prod))
	for i in range(1, 4):
		os.system('cp '+OUT_DIR+mask_dir+'zbbb_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'bbb_'+prod_n+'.jpg')
		os.system('cp '+OUT_DIR+mask_dir+'zbff_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'bff_'+prod_n+'.jpg')

	#top translation
	prod_img, prod_pgn_img = prod_img_orig.copy(), prod_pgn_img_orig.copy()
	
	move = random.randint(10,30)
	prod_img = top_translation(prod_img, move)
	prod_pgn_img = top_translation(prod_pgn_img, move)
	
	cv2.imwrite(OUT_DIR+Image_dir  +'zttt_'+ prod_n+'.jpg', prod_img)
	cv2.imwrite(OUT_DIR+Image_dir  +'ztff_'+ prod_n+'.jpg', np.fliplr(prod_img))
	
	cv2.imwrite(OUT_DIR+mask_dir  +'zttt_'+ prod_n+'.jpg', prod_pgn_img)
	cv2.imwrite(OUT_DIR+mask_dir +'ztff_'+ prod_n+'.jpg', np.fliplr(prod_pgn_img))

	r_prod, g_prod, b_prod = color_aug(prod_img)	
	cv2.imwrite(OUT_DIR+Image_dir+'c1ttt_'+prod_n+'.jpg', r_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c1tff_'+prod_n+'.jpg', np.fliplr(r_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c2ttt_'+prod_n+'.jpg', g_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c2tff_'+prod_n+'.jpg', np.fliplr(g_prod))
	cv2.imwrite(OUT_DIR+Image_dir+'c3ttt_'+prod_n+'.jpg', b_prod)
	cv2.imwrite(OUT_DIR+Image_dir+'c3tff_'+prod_n+'.jpg', np.fliplr(b_prod))
	for i in range(1, 4):
		os.system('cp '+OUT_DIR+mask_dir+'zttt_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'ttt_'+prod_n+'.jpg')
		os.system('cp '+OUT_DIR+mask_dir+'ztff_'+prod_n+'.jpg '+OUT_DIR+mask_dir+'c'+str(i)+'tff_'+prod_n+'.jpg')

lst = sorted([n for n in os.listdir(os.path.join(OUT_DIR,Image_dir)) if n.endswith('.jpg')])
print(lst[0])

agents = 6
pool = MyPool(agents)
pool.map(Augmentation_points_images, lst)
pool.close()
pool.join()
