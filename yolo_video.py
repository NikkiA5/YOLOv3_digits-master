import time
import os
import sys
import argparse
from yolo import YOLO, detect_video
from PIL import Image

import mAP

def evaluate(yolo, data_path):

    whole_process_start = time.time()
    
    try:
        for subdir, _, files in os.walk(data_path):
            print('[INFO] Working on: ' + str(subdir))
            files.sort()
            for _file in files:
                if str(_file).lower().endswith(('.png', '.jpg', '.jpeg', '.jfiff', '.tiff', '.bmp')):
                    print('>>> File: ' + str(_file))
                    # Checks if there's a corresponding annotation file
                    try:
                        # load image 
                        image = Image.open(os.path.join(subdir, _file))

                    except Exception as e:
                        print("Error reading file: " + str(_file))
                        # Break the current iteration
                        continue

                    # inferences
                    r_image, inferences = yolo.detect_image(image)
                    #print(inferences)
                    #r_image.show()

                    # write results to file
                    with open('data/eval_res/' + _file.replace('jpg', 'txt')
                        .replace('jpeg', 'txt')
                        .replace('png', 'txt')
                        .replace('jfiff', 'txt')
                        .replace('tiff', 'txt')
                        .replace('bmp', 'txt'), 'w') as f:
                        # inferences is empty, but we need to create an empty file
                        if not inferences:
                            f.write('')
                        else:
                            for inference in inferences:
                                strToWrite = str(inference['class']) + ' ' + str(inference['score']) + ' ' + str(inference['x1']) + \
                                    ' ' + str(inference['y1']) + ' ' + \
                                    str(inference['x2']) + ' ' + str(inference['y2']) + '\n'
                                f.write(strToWrite)
    except Exception as e:
        print(e)

    yolo.close_session()
    end = time.time() - whole_process_start
    with open('data/eval_res/timing.txt', 'w') as f:
        f.write(str(end))
    
    ## mAP eval
    mAP.calculate('data/annotations/test/TXT/', 'data/eval_res/', 'data/eval_res/results/')

def detect_img(yolo):
    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image)
            r_image.show()
    yolo.close_session()

FLAGS = None

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model', type=str,
        help='path to model weight file, default ' + YOLO.get_defaults("model_path")
    )

    parser.add_argument(
        '--anchors', type=str,
        help='path to anchor definitions, default ' + YOLO.get_defaults("anchors_path")
    )

    parser.add_argument(
        '--classes', type=str,
        help='path to class definitions, default ' + YOLO.get_defaults("classes_path")
    )

    parser.add_argument(
        '--gpu_num', type=int,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults("gpu_num"))
    )

    parser.add_argument(
        '--image', default=False, action="store_true",
        help='Image detection mode, will ignore all positional arguments'
    )

    parser.add_argument(
        '--eval', default=False, action="store_true",
        help='Evaluation mode, running on eval_path argument'
    )

    parser.add_argument(
        '--eval_path', type=str, default="data/examples/",
        help='Path to evaluation files'
    )
    '''
    Command line positional arguments -- for video detection mode
    '''
    parser.add_argument(
        "--input", nargs='?', type=str,required=False,default='./path2your_video',
        help = "Video input path"
    )

    parser.add_argument(
        "--output", nargs='?', type=str, default="",
        help = "[Optional] Video output path"
    )

    FLAGS = parser.parse_args()

    if FLAGS.image:
        """
        Image detection mode, disregard any remaining command line arguments
        """
        if FLAGS.eval:
            evaluate(YOLO(**vars(FLAGS)), FLAGS.eval_path)
        else:
            print("Image detection mode")
            if "input" in FLAGS:
                print(" Ignoring remaining command line arguments: " + FLAGS.input + "," + FLAGS.output)
            detect_img(YOLO(**vars(FLAGS)))
    elif "input" in FLAGS:
        detect_video(YOLO(**vars(FLAGS)), FLAGS.input, FLAGS.output)
    else:
        print("Must specify at least video_input_path.  See usage with --help.")
