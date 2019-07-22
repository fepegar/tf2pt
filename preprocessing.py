from pathlib import Path

import nibabel as nib

from niftynet.io.image_reader import ImageReader
from niftynet.utilities.util_common import ParserNamespace
from niftynet.io.image_sets_partitioner import ImageSetsPartitioner
from niftynet.layer.histogram_normalisation import HistogramNormalisationLayer
from niftynet.layer.binary_masking import BinaryMaskingLayer

def preprocess(input_path,
               model_path,
               output_path,
               cutoff,
    ):
    input_path = Path(input_path)
    output_path = Path(output_path)
    input_dir = input_path.parent
    
    DATA_PARAM = {
        'Modality0': ParserNamespace(
            path_to_search=str(input_dir),
            filename_contains=('nii.gz',),
            interp_order=0,
            pixdim=None,
            axcodes='RAS',
            loader=None,
        )
    }

    TASK_PARAM = ParserNamespace(image=('Modality0',))
    data_partitioner = ImageSetsPartitioner()
    file_list = data_partitioner.initialise(DATA_PARAM).get_file_list()
    reader = ImageReader(['image'])
    reader.initialise(DATA_PARAM, TASK_PARAM, file_list)
    
    binary_masking_func = BinaryMaskingLayer(
        type_str='mean_plus',
    )
    
    hist_norm = HistogramNormalisationLayer(
        image_name='image',
        modalities=['Modality0'],
        model_filename=str(model_path),
        binary_masking_func=binary_masking_func,
        cutoff=cutoff,
        name='hist_norm_layer',
    )
    
    image = reader.output_list[0]['image']
    data = image.get_data()
    norm_image_dict, mask_dict = hist_norm({'image': data})
    data = norm_image_dict['image']
    nii = nib.Nifti1Image(data.squeeze(), image.original_affine[0])
    dst = output_path
    nii.to_filename(str(dst))