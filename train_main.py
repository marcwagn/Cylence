from cnn_detection.utils import * 
pass
#clearAnnotations("../database/cnn_data/zeiss_lugol/filament/images/","../database/cnn_data/zeiss_lugol/filament/annotations/")
#clearAnnotations("../database/cnn_data/nikon_aldehyd/background/images/","../database/cnn_data/nikon_aldehyd/background/annotations/")
#clearAnnotations("../database/cnn_data/nikon_aldehyd/infection/images/","../database/cnn_data/nikon_aldehyd/infection/annotations/")

#createSubimages("../database/cnn_data/nikon_aldehyd/raw/images/", "../database/cnn_data/nikon_aldehyd/raw/annotations/", "../database/cnn_data/nikon_aldehyd/", relabel=False)

splitTestTrain("../database/cnn_data/final_mini/images/","../database/cnn_data/final_mini/annotations/","../database/cnn_data/final_mini/", 0.2)

#labelToRGB("../dataset/small/filter/annotations/", "../dataset/small/filter/annotaions_color/")
#rgbToLabel("../database/cnn_data/zeiss_lugol/raw/annotations_color/","../database/cnn_data/zeiss_lugol/raw/annotations/")