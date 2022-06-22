#!/usr/bin/env python
import nibabel as nib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from nipype.interfaces.base import SimpleInterface, BaseInterfaceInputSpec, TraitedSpec, File, traits


def histogram(image, normalize):
    image = image.flatten()
    
    hist, bin_edges = np.histogram(image, bins=np.arange(image.min(), image.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    std = np.std(image)
    mean = np.mean(image)
    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers, mean, std

def thresholding(in_file, op_string=None):
    nii_file = nib.load(in_file)
    array = nii_file.get_fdata()
    hist, bin, mu, std = histogram(array, True)
    normal_distribution = norm.pdf(bin, mu, std)
    difference = [normal_distribution[i] - hist[i] if hist[i]<normal_distribution[i] else 0 for i in range(len(hist)) ]
    maxpoint = max(range(len(difference)), key=difference.__getitem__)
    threshold = bin[maxpoint]/np.amax(array)*100

    op_string =  '-thrp {threshold} -bin -ero -dilM'.format(threshold=threshold)
    return op_string


class ThresholdInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True)


class ThresholdOutputSpec(TraitedSpec):
    op_string = traits.String()


class ThresholdInterface(SimpleInterface):
    input_spec = ThresholdInputSpec
    output_spec = ThresholdOutputSpec

    def _run_interface(self, runtime):
        self._results['op_string'] = thresholding(self.inputs.in_file)
        return runtime


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'in_file',
        type=str
    )

    parser.add_argument(
        'op_string',
        nargs='?',
        default=None,
        const=None,
        type=str
    )

    args = parser.parse_args()
    op_string = thresholding(args.in_file, args.op_string)
    