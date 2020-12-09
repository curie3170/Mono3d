import os
#from utils import readlines
      
def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


raw_root = "/root/dataset/kitti_raw1"
print(raw_root)
splits_dir = os.path.join(os.path.dirname(__file__), "image_sets")
train_mapping = readlines(os.path.join(splits_dir, "train_mapping.txt"))
train_rand = readlines(os.path.join(splits_dir, "train_rand.txt"))
train_rand = train_rand[0].split(",")
train_list = readlines(os.path.join(splits_dir, "train.txt"))
test_list = readlines(os.path.join(splits_dir, "val.txt"))

train_files=[]
for idx, rand_idx in enumerate(train_rand):
    if str(idx).zfill(6) in train_list:
        raw = train_mapping[int(rand_idx)-1].split(" ")
        #train_file = str(raw[0]+"/"+raw[1]+" "+raw[2]+" "+"l") #Monodepth2 dataloader format
        train_file = os.path.join(raw_root, str(raw[0]+"/"+raw[1]+"/"+raw[2]+".jpg"))
        post_path = os.path.join(raw_root, str(raw[0]+"/"+raw[1]+"/"), "image_02/data", str(int(raw[2])+1).zfill(10) + ".jpg")
        prev_path = os.path.join(raw_root, str(raw[0]+"/"+raw[1]+"/"), "image_02/data", str(int(raw[2])-1).zfill(10) + ".jpg")
        if (not os.path.exists(post_path)):
            print(str(idx).zfill(6))
            continue
        if (not os.path.exists(prev_path)):
            print(str(idx).zfill(6))
            continue
        train_files.append(train_file)
'''
f = open(os.path.join(splits_dir, "train_files.txt"), 'w')
for idx in len(train_files):
    f.write(train_files[idx]+"\n")
f.close()
'''
val_files=[]
for idx, rand_idx in enumerate(train_rand):
    if str(idx).zfill(6) in test_list:
        raw = train_mapping[int(rand_idx)-1].split(" ")
        val_file = str(raw[0]+"/"+raw[1]+" "+raw[2]+" "+"l")
        post_path = os.path.join(raw_root, str(raw[0]+"/"+raw[1]+"/"), "image_02/data", str(int(raw[2])+1).zfill(10) + ".jpg")
        prev_path = os.path.join(raw_root, str(raw[0]+"/"+raw[1]+"/"), "image_02/data", str(int(raw[2])-1).zfill(10) + ".jpg")
        if (not os.path.exists(post_path)):
            print(str(idx).zfill(6))
            continue
        if (not os.path.exists(prev_path)):
            print(str(idx).zfill(6))
            continue

        val_files.append(val_file)
'''
f = open(os.path.join(splits_dir, "val_files.txt"), 'w')
for val_file in val_files:
    f.write(val_file+"\n")
f.close()
'''

