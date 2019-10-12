from utilities.HemorrhageTypes import HemorrhageTypes

CSV_FILENAME = "Submission.csv"


# output_dict is dictionary where the image ids are the keys and the values are tuples of probabilities
# the order of the probabilities in the tuples should be the following:
# epidural , intraparenchymal, intraventricular, subarachnoid, subdural and any.


def create_output_csv(output_dict):
    content = "ID,Label\n"
    for image_id in output_dict:
        content += create_output_line(image_id, HemorrhageTypes.EP.value, output_dict[image_id][0])
        content += create_output_line(image_id, HemorrhageTypes.IN_PA.value, output_dict[image_id][1])
        content += create_output_line(image_id, HemorrhageTypes.IN_VE.value, output_dict[image_id][2])
        content += create_output_line(image_id, HemorrhageTypes.SUB_AR.value, output_dict[image_id][3])
        content += create_output_line(image_id, HemorrhageTypes.SUB_DU.value, output_dict[image_id][4])
        content += create_output_line(image_id, HemorrhageTypes.ANY.value, output_dict[image_id][5])
    with open(CSV_FILENAME, "w") as f:
        f.write(content)


def create_output_line(image_id, hemorrhage_type, probability):
    return "ID_" + image_id + "_" + hemorrhage_type + "," + str(probability) + "\n"


# code for testing
if __name__ == '__main__':
    output_dict = dict()
    for index in range(100):
        output_dict["image" + str(index)] = (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    create_output_csv(output_dict)
