import os
import csv
import subprocess
import imghdr

import cv2
import pandas as pd

from sklearn.cluster import MiniBatchKMeans
from skimage.metrics import structural_similarity as ssim


# resize Image
def image_resize(image_to_resize: object, set_width: int):
    width = int(image_to_resize.shape[1] * set_width / image_to_resize.shape[1])
    height = int(image_to_resize.shape[0] * set_width / image_to_resize.shape[1])
    dimensions = (width, height)
    return cv2.resize(image_to_resize, dimensions, interpolation=cv2.INTER_AREA)


# quantize image to reduce colour data for easy raster to vector conversion
def quantize_image(image: object, number_of_colors: int):
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    clt = MiniBatchKMeans(n_clusters=int(number_of_colors))
    labels = clt.fit_predict(image)
    quantized_image = clt.cluster_centers_.astype("uint8")[labels]
    quantized_image = quantized_image.reshape((h, w, 3))
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_LAB2BGR)
    # return resized image
    return quantized_image


def save_image_to_path(path: str, image: object) -> object:
    cv2.imwrite(path, image)


def convert_to_png(in_path: str, out_path: str):
    subprocess.call(["svg2png.bat", in_path, out_path])


def check_similarity(path_1: str, path_2: str, image_size):
    img1 = cv2.imread(path_1)
    img2 = image_resize(cv2.imread(path_2), image_size)

    w1, h1 = img2.shape[:-1]
    img1 = cv2.resize(img1, (h1, w1))

    s = ssim(img1, img2, multichannel=True)
    return str(s * 100)


def variable_full_process(category):
    image_size = 720
    end_file_type = ".svg"

    # Initialize directories for category
    try:
        os.mkdir("images/png/" + category)
        os.mkdir("images/tempimages/" + category)
        os.mkdir("images/converted/" + category)
        os.mkdir("csv/" + category)
    except OSError as e:
        print(e)

    # Load necessary paths for the reading and writing
    data_path = "images/data/" + category
    temp_path = "images/tempimages/" + category
    out_svg_base_path = "images/converted/" + category
    png_base_path = "images/png/" + category

    csv_path = 'csv/' + category

    for img in os.listdir(data_path):

        image = cv2.imread(os.path.join(data_path, img))
        image = image_resize(image, image_size)
        image = quantize_image(image, 16)

        image_name = img.split(".")[0]
        image_extension = img.split(".")[1]

        try:
            os.mkdir(os.path.join(temp_path, image_name))
            os.mkdir(os.path.join(out_svg_base_path, image_name))
            os.mkdir(os.path.join(png_base_path, image_name))
        except OSError as e:
            print(e)

        out_svg_specific_path = os.path.join(out_svg_base_path, image_name)

        sample_path = os.path.join(temp_path, image_name, "sample." + image_extension)
        save_image_to_path(sample_path, image)

        index = 0

        with open(csv_path + "/" +image_name + '.csv', 'w', newline='') as f:
            thewriter = csv.writer(f)

            thewriter.writerow(["index", "ltres", "qtres", "pathomit", "file_path", "similarity"])

            for ltres in range(0, 9, 2):
                for qtres in range(0, 9, 2):
                    for pathomit in range(0, 101):
                        if pathomit == 1 or pathomit == 10 or pathomit == 100:
                            print("progress:" + str(int(index / 75 * 100)) + "%")

                            index = index + 1

                            subprocess.call([
                                'java', '-jar', 'libraries/imageTrace.jar', sample_path,
                                'outfilename', out_svg_specific_path + "/" + str(index) + end_file_type,
                                'pathomit', str(1 / pathomit),
                                'ltres', str(ltres),
                                'qtres', str(qtres),
                                'colorsampling', str(0),
                                'colorquantcycles', str(16)
                            ])

                            png_out_path = "images/png/" + category + "/" + image_name + "/" + str(index) + ".png"

                            convert_to_png("../../../" + out_svg_specific_path + "/" + str(index) + end_file_type,
                                           "../../../" + png_out_path)

                            similarity_val = check_similarity(sample_path, png_out_path, image_size)

                            thewriter.writerow(
                                [str(index), str(ltres), str(qtres), str(pathomit), str(index) + end_file_type,
                                 str(similarity_val)])

    print("completed!")


def read_csv(category: str):
    base_file_path = 'csv'

    with open('final_csv/' + category + '.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)

        thewriter.writerow(["index", "ltres", "qtres", "pathomit", "file_path", "similarity"])

        for csv_path in os.listdir(os.path.join(base_file_path, category)):
            df = pd.read_csv(os.path.join(base_file_path, category, csv_path), engine='python')
            df = df.sort_values('similarity', ascending=False)
            df.to_csv(os.path.join(base_file_path, category, "converted_" + csv_path), index=False)

        for csv_path in os.listdir(os.path.join(base_file_path, category)):
            if (csv_path.startswith("converted_")):
                input_file = csv.DictReader(open(os.path.join(base_file_path, category, csv_path)))

                # print highest 10 accracy values
                index = 0

                for row in input_file:
                    if index < 1:
                        thewriter.writerow(
                            [str(row['index']), str(row['ltres']), str(row['qtres']), str(row['pathomit']),
                             str(row['index']) + row['file_path'],
                             str(row['similarity'])])

                    index += 1


def cleanup(base_file_path: object, category: object) -> object:
    for csv_path in os.listdir(os.path.join(base_file_path, category)):
        if csv_path.startswith("converted_"):
            os.unlink(os.path.join(base_file_path, category, csv_path))


def finalize_params():
    category = 'landclass'

    variable_full_process(category)
    read_csv(category)
    cleanup('csv', category)


finalize_params()


# Usage of Identified Param files ######################################################################################
def get_best_param_range():
    path = 'final_csv/landclass.csv'

    ltres_low = find_lowest(path, "ltres")
    ltres_high = find_highest(path, "ltres")

    qtres_low = find_lowest(path, "qtres")
    qtres_high = find_highest(path, "qtres")

    pathomit_low = find_lowest(path, "pathomit")
    pathomit_high = find_highest(path, "pathomit")

    return [('ltres_low', ltres_low), ('ltres_high', ltres_high), ('qtres_low', qtres_low), ('qtres_high', qtres_high),
            ('pathomit_low', pathomit_low), ('pathomit_high', pathomit_high)]


def find_lowest(path, prop_val):
    input_file = csv.DictReader(open(path))
    lowest = 0
    index = 0

    for row in input_file:
        if index == 0:
            lowest = row[prop_val]
        else:
            if row[prop_val] < lowest:
                lowest = row[prop_val]

        index += 1

    return lowest


def find_highest(path, prop_val):
    input_file = csv.DictReader(open(path))
    highest = 0
    index = 0

    for row in input_file:
        if index == 0:
            highest = row[prop_val]
        else:
            if row[prop_val] > highest:
                highest = row[prop_val]

        index += 1

    return highest


def find_landclass_optimal_params(image_path):
    end_file_type = ".svg"
    param_range = get_best_param_range()
    image_size = 250

    image = cv2.imread(image_path)
    image = image_resize(image, image_size)
    image = quantize_image(image, 16)
    image_extension = imghdr.what(image_path)
    temp_image_path = 'temp/sample.' + image_extension
    save_image_to_path(temp_image_path , image)

    index = 0

    with open('temp.csv', 'w', newline='') as f:
        thewriter = csv.writer(f)

        thewriter.writerow(["index", "ltres", "qtres", "pathomit", "file_path", "similarity"])

        for ltres in range(int(param_range[0][1]), int(param_range[1][1])+1, 1):
            for qtres in range(int(param_range[2][1]), int(param_range[3][1])+1, 1):
                for pathomit in range(int(param_range[4][1]), int(param_range[5][1])):
                    if pathomit == 1 or pathomit == 10 or pathomit == 100:
                        index = index + 1

                        svg_out_path = "temp/svg/" + str(index) + end_file_type

                        subprocess.call([
                            'java', '-jar', 'imageTrace.jar', temp_image_path,
                            'outfilename', svg_out_path,
                            'pathomit', str(1 / pathomit),
                            'ltres', str(ltres),
                            'qtres', str(qtres),
                            'colorsampling', str(0),
                            'colorquantcycles', str(16)
                        ])

                        png_out_path = "temp/png/" + str(index) + ".png"

                        convert_to_png("../../" + svg_out_path, "../../" + png_out_path)

                        similarity_val = check_similarity(temp_image_path, png_out_path, image_size)

                        thewriter.writerow(
                            [str(index), str(ltres), str(qtres), str(pathomit), str(index) + end_file_type,
                             str(similarity_val)])