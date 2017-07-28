fw1_test = open('img1_list_test.txt', 'w')
fw2_test = open('img2_list_test.txt', 'w')
fw3_test = open('flo_list_test.txt', 'w')
datadir = '/home/susean/Flownet/dataset/Sintel/training/clean/'
flowdir = '/home/susean/Flownet/dataset/Sintel/training/flow/'
movies = ["alley_1", "ambush_2", "ambush_5", "ambush_7", "bamboo_2", "bandage_2", "cave_4", "market_5", "mountain_1", "shaman_3", "sleeping_2", "temple_3"		, "alley_2", "ambush_4", "ambush_6", "bamboo_1", "bandage_1", "cave_2", "market_2", "market_6", "shaman_2", "sleeping_1", "temple_2"]
count = 0
for movie in movies:
	if movie == "ambush_2":
		for i in xrange(1,21):
			fw1_test.write(datadir+movie+"/frame_"+str(i).zfill(4)+".png\n")
			fw2_test.write(datadir+movie+"/frame_"+str(i+1).zfill(4)+".png\n")
			fw3_test.write(flowdir+movie+"/frame_"+str(i).zfill(4)+".flo\n")
			count += 1		
	elif movie == "ambush_4":
		for i in xrange(1,32):#33
			fw1_test.write(datadir+movie+"/frame_"+str(i).zfill(4)+".png\n")
			fw2_test.write(datadir+movie+"/frame_"+str(i+1).zfill(4)+".png\n")
			fw3_test.write(flowdir+movie+"/frame_"+str(i).zfill(4)+".flo\n")
			count += 1 
	elif movie == "ambush_6":
		for i in xrange(1,20):
			fw1_test.write(datadir+movie+"/frame_"+str(i).zfill(4)+".png\n")
			fw2_test.write(datadir+movie+"/frame_"+str(i+1).zfill(4)+".png\n")
			fw3_test.write(flowdir+movie+"/frame_"+str(i).zfill(4)+".flo\n")
			count += 1 
	elif movie == "market_6":
	    for i in xrange(1,40):
			fw1_test.write(datadir+movie+"/frame_"+str(i).zfill(4)+".png\n")
			fw2_test.write(datadir+movie+"/frame_"+str(i+1).zfill(4)+".png\n")
			fw3_test.write(flowdir+movie+"/frame_"+str(i).zfill(4)+".flo\n")
			count += 1 
	else:
		for i in xrange(1,50):	
			fw1_test.write(datadir+movie+"/frame_"+str(i).zfill(4)+".png\n")
			fw2_test.write(datadir+movie+"/frame_"+str(i+1).zfill(4)+".png\n")
			fw3_test.write(flowdir+movie+"/frame_"+str(i).zfill(4)+".flo\n")
			count += 1 
print("pair frame nums = {0}".format(count))

fw1_test.close()
fw2_test.close()
fw3_test.close()
