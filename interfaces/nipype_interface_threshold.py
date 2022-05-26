#!/usr/bin/env python
import nibabel as nib
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from nipype.interfaces.base import SimpleInterface, BaseInterfaceInputSpec, TraitedSpec, File
from nipype.interfaces.fsl import ImageMaths

def histogram(image, normalize):
    image = image.flatten()
    
    hist, bin_edges = np.histogram(image, bins=np.arange(image.min(), image.max()))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.
    std = np.std(image)
    mean = np.mean(image)
    if normalize:
        hist = hist / np.sum(hist)
    return hist, bin_centers, mean, std

def thresholding(in_file, out_file=None):
    nii_file = nib.load(in_file)
    array = nii_file.get_fdata()
    hist, bin, mu, std = histogram(array, True)
    normal_distribution = norm.pdf(bin, mu, std)
    difference = [normal_distribution[i] - hist[i] if hist[i]<normal_distribution[i] else 0 for i in range(len(hist)) ]
    maxpoint = max(range(len(difference)), key=difference.__getitem__)
    threshold = bin[maxpoint]/np.amax(array)*100

    out_file =  ImageMaths(
                suffix='_mask',
                op_string=f'-thrp {threshold} -bin -ero -dilM'
            )
    return out_file

class ThresholdInputSpec(BaseInterfaceInputSpec):
    in_file = File(mandatory=True, exists=True)


class ThresholdOutputSpec(TraitedSpec):
    out_file = File()


class ThresholdInterface(SimpleInterface):
    input_spec = ThresholdInputSpec
    output_spec = ThresholdOutputSpec

    def _run_interface(self, runtime):
        self._results['out_file'] = thresholding(self.inputs.in_file)
        return runtime


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'in_file',
        type=str
    )

    