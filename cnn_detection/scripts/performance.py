import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import csv
import matplotlib.pyplot as plt

def analysePerformance(pathModel, maxCheckpoints, testDataPath, outFile):
    from image_segmentation_keras.keras_segmentation.predict import model_from_checkpoint_path
    from image_segmentation_keras.keras_segmentation.predict import evaluate

    with open(outFile, 'w', newline='\n') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter='\t',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['meanUoI', 'filament_IoU', 'infection_IoU', 'background_IoU'])

        for checkpoint in range(maxCheckpoints):
            model = model_from_checkpoint_path(pathModel, checkpoint=checkpoint)
            para = evaluate(model,
                            inp_images_dir= testDataPath + 'images_test/',
                            annotations_dir=testDataPath + 'annotations_test/')

            spamwriter.writerow([str(para['mean_IU']),
                                 str(para['class_wise_IU'][0]),
                                 str(para['class_wise_IU'][1]),
                                 str(para['class_wise_IU'][2])])

def plotPerformance(inFile, outFile):
    with open(inFile, newline='\n') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\t', quotechar='|')
        next(spamreader)

        meanUoI = []
        filament_IoU = []
        infection_IoU = []
        background_IoU = []

        for row in spamreader:
            meanUoI.append(float(row[0]))
            filament_IoU.append(float(row[1]))
            infection_IoU.append(float(row[2]))
            background_IoU.append(float(row[3]))

    x = range(1,len(meanUoI)+1,1)

    linewidth = 2
    markersize = 6

    plt.plot(x, filament_IoU, marker='o', markerfacecolor='green', markersize=markersize, color='lightgreen', linewidth=linewidth, label='filament IoU')
    plt.plot(x, infection_IoU, marker='o', markerfacecolor='darkred', markersize=markersize, color='lightcoral', linewidth=linewidth, label='infection IoU')
    plt.plot(x, background_IoU, marker='o', markerfacecolor='blue', markersize=markersize, color='skyblue', linewidth=linewidth, label='background IoU')
    plt.plot(x, meanUoI, marker='o', markerfacecolor='black', markersize=markersize, color='silver', linewidth=linewidth, label='mean IoU')
    plt.xlabel('epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.savefig(outFile)

if __name__ == "__main__":
    analysePerformance('../../../model/vgg_unet_cross_aug_ep200/vgg_unet_cross', 
                        200, 
                        '../../../data/final_pub/',
                        '../../../results/performance/cross_200.csv')
    #plotPerformance('../results/performance/ep100/cross.csv','../results/performance/ep100/cross.png')