from PIL import Image

datadir = '/mnt/flownet/data/'
for i in range(22872):
	index = i+1
	im = Image.open(datadir+str(index).zfill(5)+"_img1.ppm")
	im.save(datadir+str(index).zfill(5)+"_img1.jpeg")
	im = Image.open(datadir+str(index).zfill(5)+"_img2.ppm")
	im.save(datadir+str(index).zfill(5)+"_img2.jpeg")
