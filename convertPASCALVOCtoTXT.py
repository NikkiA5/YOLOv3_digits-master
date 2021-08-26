import sys
import os
import argparse
from bs4 import BeautifulSoup as soup 

parser = argparse.ArgumentParser(description='Convert XML Pascal VOC style annotations to one txt file ready for YOLO training.')
parser.add_argument('--pascal_path', help='Path to Darknet cfg file.', default='data/annotations/')
parser.add_argument('--output_name', help='Path to output Keras model file.', default='annotations.txt')

def convertVOCtoTXT(handler):
    xml_data = soup(handler, 'lxml')

    classCorrespondance = {
        '10': '0',
        '1': '1',
        '2': '2',
        '3': '3',
        '4': '4',
        '5': '5',
        '6': '6',
        '7': '7',
        '8': '8',
        '9': '9',
    }

    line_training = xml_data.find('filename').text 
    line_mAP = ''
    
    for obj in xml_data.find_all('object'):
        line_training += ' ' + \
        obj.find('xmin').text + ',' + \
        obj.find('ymin').text + ',' + \
        obj.find('xmax').text + ',' + \
        obj.find('ymax').text + ',' + \
        classCorrespondance[obj.find('name').text]

        line_mAP += classCorrespondance[obj.find('name').text] + ' ' + \
            obj.find('xmin').text + ' ' + \
            obj.find('ymin').text + ' ' + \
            obj.find('xmax').text + ' ' + \
            obj.find('ymax').text + '\n'
    
    line_training += '\n'
    
    return line_training, line_mAP

def main(args):
    print('[INFO] Converting VOC style to YOLO txt')
    
    # Iterate thourgh the directories
    for subdir, _, files in os.walk(args.pascal_path):
        # Skip subdir (avoid TXT subdir)
        if(subdir != args.pascal_path):
            continue
        print('[INFO] Working on: ' + str(subdir))
        print(os.pardir)
        txt = ''
        if(not files):
            continue
        files.sort()
        for _file in files:
            print('>>> File: ' + str(_file))
            if str(_file).lower().endswith('.xml'):
                try:
                    handler = open(os.path.join(subdir, _file)).read()
                except Exception as e:
                    print('[ERROR] Reading file: ' +
                            str(os.path.join(subdir, _file)))
                    sys.exit(0)
                    
                # Convert files
                line_training, line_mAP= convertVOCtoTXT(handler)
                txt += line_training

                # Write res to a file (for later use in mAP calculation)
                with open(os.path.join(args.pascal_path, 'TXT', _file.replace('xml', 'txt')), 'w') as f:
                    f.write(line_mAP)

        with open(subdir + '_' + args.output_name, 'w') as f:
            f.write(txt)
    print('[INFO] Done converting VOC style to YOLO txt')

if __name__ == '__main__':
    main(parser.parse_args())