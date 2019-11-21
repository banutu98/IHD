from utilities.defines import HemorrhageTypes, CSV_FILENAME
import time


def create_output_csv(output_dict):
    content = "ID,Label\n"
    for image_id in output_dict:
        for num, hemorrhageType in enumerate(HemorrhageTypes, start=0):
            content += create_output_line(image_id, hemorrhageType.value, output_dict[image_id][num])
    with open(CSV_FILENAME, "w") as f:
        f.write(content)


def create_output_line(image_id, hemorrhage_type, probability):
    return image_id + "_" + hemorrhage_type + "," + str(probability) + "\n"


# code for testing -> time for creating csv approx 10 seconds
if __name__ == '__main__':
    start = time.time()
    output_dict = dict()
    for index in range(78545):
        output_dict["image" + str(index)] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    create_output_csv(output_dict)
    print(time.time() - start)
