import cv2
import glob

class dataLoader():

    def __init__(self, image_path, mode):
        """
        Args:
            image_path (str): the path where the image is located
            mode (int): 0=grayscale, 1=color 
        """
        # all file names
        self.img_arr = glob.glob(str(image_path) + str("/*c1.tif"))
        # Calculate len
        self.data_len = len(self.img_arr)
        #save mode
        if (mode == 1 or mode==0):
            self.mode = mode
        else:
            raise ValueError("Mode of dataloader not defined! [0=grayscale, 1=color]")

    def __getitem__(self, index):
        """Get specific data corresponding to the index
        Args:
            index (int): index of the data
        Returns:
            imgs[0]: ndarray of bright color image (height,width,rgb)
            imgs[1]: ndarray of blue flourecent image (height,width,rgb)
            imgs[2]: ndarray of red flourecent image (height,width,rgb)
        """
        #read image paths
        id = self.img_arr[index].split("/")[-1].replace('c1.tif','')
        img_paths = []
        img_paths.append(self.img_arr[index])
        img_paths.append(self.img_arr[index].replace('c1','c2'))
        img_paths.append(self.img_arr[index].replace('c1','c3'))

        #read images
        imgs = []
        for img_path in img_paths:
          imgs.append(cv2.imread(img_path,self.mode))

        return (id, imgs[0],imgs[1],imgs[2])

    def __len__(self):
        """
        Returns:
            length (int): length of the data
        """
        return self.data_len


if __name__ == "__main__":

    database = dataLoader('../data/',3)

    bright, blue, red = database.__getitem__(0)
    print(red.shape)