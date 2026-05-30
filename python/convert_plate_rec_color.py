import sys
from rknn.api import RKNN

DATASET_PATH = '../model/dataset.txt'
DEFAULT_RKNN_PATH = '../model/plate_rec_color.rknn'
DEFAULT_QUANT = True

def parse_arg():
    if len(sys.argv) < 3:
        print("Usage: python3 {} onnx_model_path [platform] [dtype(optional)] [output_rknn_path(optional)] [dataset_path(optional)]".format(sys.argv[0]));
        print("       platform choose from [rk3562, rk3566, rk3568, rk3576, rk3588, rv1109, rv1126, rk1808]")
        print("       dtype choose from    [i8, fp] for [rk3562, rk3566, rk3568, rk3576, rk3588, rv1126b]")
        print("       dtype choose from    [u8, fp] for [rv1109, rv1126, rk1808]")
        exit(1)

    model_path = sys.argv[1]
    platform = sys.argv[2]

    do_quant = DEFAULT_QUANT
    if len(sys.argv) > 3:
        model_type = sys.argv[3]
        if model_type not in ['i8', 'u8', 'fp']:
            print("ERROR: Invalid model type: {}".format(model_type))
            exit(1)
        elif model_type in ['i8', 'u8']:
            do_quant = True
        else:
            do_quant = False

    if len(sys.argv) > 4:
        output_path = sys.argv[4]
    else:
        output_path = DEFAULT_RKNN_PATH

    if len(sys.argv) > 5:
        dataset_path = sys.argv[5]
    else:
        dataset_path = DATASET_PATH

    return model_path, platform, do_quant, output_path, dataset_path

if __name__ == '__main__':
    model_path, platform, do_quant, output_path, dataset_path = parse_arg()

    rknn = RKNN(verbose=False)

    print('--> Config model')
    # The original PyTorch preprocessing is:
    #   img = (img / 255.0 - 0.588) / 0.193
    # RKNN preprocessing works on raw uint8 pixels, so convert the mean/std to
    # the 0-255 domain to keep board-side uint8 input consistent with Python.
    rknn.config(
        mean_values=[[149.94, 149.94, 149.94]],
        std_values=[[49.215, 49.215, 49.215]],
        target_platform=platform,
        quantized_algorithm='normal',
        quantized_method='channel'
    )
    print('done')

    print('--> Loading model')
    ret = rknn.load_onnx(model=model_path, inputs=['input'], input_size_list=[[1, 3, 48, 168]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    print('--> Building model')
    if do_quant:
        print('    dataset: {}'.format(dataset_path))
        ret = rknn.build(do_quantization=True, dataset=dataset_path)
    else:
        ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')

    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')

    rknn.release()
