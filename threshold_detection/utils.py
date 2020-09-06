import numpy as np
import matplotlib.pyplot as plt

def surfacePlot(img, outFile):
    """
    Args:
        img: grayscale image
        path: output file for img.png
    Returns:
        none
    """
    # create the x and y coordinate arrays
    xx, yy = np.mgrid[0:img.shape[0], 0:img.shape[1]]

    # create the figure
    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    ax = fig.gca(projection='3d')

    #create plot
    surf = ax.plot_surface(xx, yy, img ,rstride=1, cstride=1, cmap='tab20',
                            linewidth=0)

    #add color scale
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #add axis label
    ax.set_xlabel('image height')
    ax.set_ylabel('image width')
    ax.set_zlabel('signal intensity')

    # show it
    plt.savefig(outFile, dpi=300)
