fr = open('hkkim.txt', 'r')
fw1_test = open('img1_list_test.txt', 'w')
fw2_test = open('img2_list_test.txt', 'w')
fw3_test = open('flo_list_test.txt', 'w')
fw1_train = open('img1_list_train.txt', 'w')
fw2_train = open('img2_list_train.txt', 'w')
fw3_train = open('flo_list_train.txt', 'w')
datadir = '/mnt/flownet/data/'
index = 0
train = 0
test = 0
for flag in fr:
	index +=1
	if flag[0] == '1':
		fw1_train.write(datadir+str(index).zfill(5)+"_img1.jpeg\n")
		fw2_train.write(datadir+str(index).zfill(5)+"_img2.jpeg\n")
		fw3_train.write(datadir+str(index).zfill(5)+"_flow.flo\n")
		train +=1
	else:
		fw1_test.write(datadir+str(index).zfill(5)+"_img1.jpeg\n")
		fw2_test.write(datadir+str(index).zfill(5)+"_img2.jpeg\n")
		fw3_test.write(datadir+str(index).zfill(5)+"_flow.flo\n")
		test +=1
print("train nums = {0}, test nums = {1}".format(train, test))
fr.close()
fw1_test.close()
fw2_test.close()
fw3_test.close()
fw1_train.close()
fw2_train.close()
fw3_train.close()
