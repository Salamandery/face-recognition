from mtcnn.mtcnn import MTCNN
from PIL import Image
from os import listdir
from os.path import isdir
from numpy import array, asarray

detector = MTCNN()

def extract_face(filename, size=(160,160)):
    img = Image.open(filename)
    img = img.convert('RGB')
    array = asarray(img)
    
    results = detector.detect_faces(array)
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height

    face = array[y1:y2, x1:x2]

    image = Image.fromarray(face)
    image = image.resize(size)

    return image

def load_faces(source, target):
    print(source)
    print(target)

def discovery_dir(source, target):
    for subdir in listdir(source):
        path = source + subdir + "\\"
        path_target = target + subdir + "\\"

        if not isdir(path):
            continue

        load_faces(path, path_target)

if __name__ == '__main__':
    discovery_dir("C:\\Users\\suporte\\Downloads\\projetos\\facce_id\\dataset\\fotos",
                  "C:\\Users\\suporte\\Downloads\\projetos\\facce_id\\dataset\\faces")