import matplotlib.pyplot as plt 
import numpy as np 
import torch


def imshow_with_key(img, key, denormalize = True):
	img = img.permute(1,2,0).cpu().detach().numpy()
	key = key.view(-1,2).cpu().detach().numpy() #(136,) -> (68,2)
	
	if denormalize == True:
		img = img * np.array([0.485, 0.456, 0.406]) + np.array([0.229, 0.224, 0.225])
		
	plt.imshow(img)
	plt.scatter(key[:,0],key[:,1], s = 10, c = 'r')
    
def compare_keypoints(image, key, out_key):
	image = image.squeeze().permute(1,2,0)
	image = image.cpu().detach().numpy()
	image = image * np.array([0.485, 0.456, 0.406]) + np.array([0.229, 0.224, 0.225]) 

	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

	ax1.set_title('original-keypoints')
	ax1.imshow(image)
	key = key.view(-1,2).cpu().detach().numpy()
	ax1.scatter(key[:,0],key[:,1], s = 10, c = 'b')

	ax2.set_title('model-keypoints')
	ax2.imshow(image)
	out_key = out_key.view(-1,2).cpu().detach().numpy()
	ax2.scatter(out_key[:,0],out_key[:,1], s = 10, c = 'g')
