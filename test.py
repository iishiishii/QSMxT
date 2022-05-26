import nibabel as nib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


def histogram(image, normalize):
    """
    Parameters
    ----------
    image : ndarray
        Image for which the histogram is to be computed.
    bins : int or ndarray
        The number of histogram bins. For images with integer dtype, an array
        containing the bin centers can also be provided. For images with
        floating point dtype, this can be an array of bin_edges for use by
        ``np.histogram``.
    source_range : string, optional
        'image' (default) determines the range from the input image.
        'dtype' determines the range from the expected range of the images
        of that data type.
    normalize : bool, optional
        If True, normalize the histogram by the sum of its values.
    """

    image = image.flatten()
    
    hist, bin_edges = np.histogram(image, bins=np.arange(image.min(), image.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    std = np.std(image)
    mean = np.mean(image)
    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers, mean, std

def thresholding(image):
    nii_file = nib.load(image)
    array = nii_file.get_fdata()
    hist, bin, mu, std = histogram(array, True)
    normal_distribution = norm.pdf(bin, mu, std)
    difference = [normal_distribution[i] - hist[i] if hist[i]<normal_distribution[i] else 0 for i in range(len(hist)) ]
    maxpoint = max(range(len(difference)), key=difference.__getitem__)
    threshold = bin[maxpoint]/np.amax(array)*100
    return threshold

if __name__ == "__main__":

    print(thresholding("sub-masopma01-0001bl_ses-1_run-1_echo-1_part-magnitude_MEGRE.nii"))
    # flat_array = array.flatten()
    # print(np.amax(array), flat_array.shape, array.shape)

    # mu = 0
    # variance = 1
    # sigma = math.sqrt(variance)
    # x = np.linspace(mu - np.amax(flat_array)/2, mu + np.amax(flat_array)/2, len(flat_array))
    # plt.plot(x, stats.norm.pdf(x, mu, sigma))

    
    # y,x,_ = plt.hist(array.ravel(), bins=np.arange(flat_array.min(), flat_array.max()), density=True, color='g')


    # print(hist,bin, len(hist), len(bin))
    # plt.plot(bin, hist, color='b')

    # mu, std = stats.norm.fit(flat_array)


    # print(len(x), len(y), len(p), np.amax(flat_array))
    # print(np.amax(flat_array-p))

    # print(difference,maxpoint, bin[maxpoint])
    # plt.plot(bin,p, 'k', linewidth=2)
    # plt.axvline(x=bin[maxpoint])
    # title = "Fit results: mu = %.2f,  std = %.2f" % (mu, std)
    # plt.title(title)
    # plt.show()