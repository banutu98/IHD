import datetime
import time
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, redirect, flash, url_for
# import sys
# sys.path.append("..")
from ihd_web import *
import aspectlib

UPLOAD_FOLDER = 'uploaded_files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'

# dictionary with name and model keys
cache = {}


@app.route("/")
def homepage():
    return render_template('index.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/ihd")
def ihd():
    return render_template('ihd.html')


@app.route("/result")
def result():
    return render_template('result.html')


@app.route("/upload_file", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        uploaded_files = request.files.getlist("files")
        if len(uploaded_files) == 1:
            if len(uploaded_files[0].filename) == 0:
                flash('No files selected!')
                return redirect(request.url)
        current_files = list()
        for file in uploaded_files:
            if os.path.splitext(file.filename)[1] == '.dcm':
                save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(save_path)
                current_files.append(save_path)
        if len(current_files):
            # add logger aspect
            with aspectlib.weave(predict, model_log):
                any_prob, subtype, subtype_prob = predict(current_files)
            any_prob, subtype_prob = round(any_prob * 100, 2), round(subtype_prob * 100, 2)
            for num, hemorrhageType in enumerate(HemorrhageTypes, start=0):
                if num == subtype:
                    subtype = hemorrhageType.value
                    break
            images = prepare_gallery(current_files)
            clean(current_files)
            print(any_prob, subtype, subtype_prob)
            return render_template('result.html', any_prob=any_prob, subtype=subtype,
                                   subtype_prob=subtype_prob, images=images)
        flash('No dcm files selected!')
        return redirect(request.url)
    return render_template('ihd.html')


def predict(files):
    if len(files) == 1:
        # add model caching aspect
        with aspectlib.weave(get_model, model_cache):
            predictions = predict_single_file(files[0])
        if predictions[0] < 0.1:
            return 0, 0, 0
        else:
            subtype = np.argmax(predictions[1:])
            return predictions[0], subtype, predictions[subtype]
    else:
        # add model caching aspect
        with aspectlib.weave(get_model, model_cache):
            sequence_predictions = predict_file_sequence(files)
        has_hemorrhage_prob = 0
        for seq in sequence_predictions:
            if seq[0] >= 0.1 and seq[0] > has_hemorrhage_prob:
                has_hemorrhage_prob = seq[0]
        if has_hemorrhage_prob < 0.1:
            return 0, 0, 0
        else:
            max_prob = 0
            sequence_nr = 0
            sequence_idx = 0
            for i in range(len(sequence_predictions)):
                seq = sequence_predictions[i][1:]
                max_index = np.argmax(seq)
                if seq[max_index] > max_prob:
                    max_prob = seq[max_index]
                    sequence_idx = max_index
                    sequence_nr = i
            return sequence_predictions[sequence_nr][0], sequence_idx, sequence_predictions[sequence_nr][sequence_idx]


def prepare_gallery(current_files):
    images = list()
    line = list()
    image_files = [Preprocessor.preprocess(file) for file in current_files]
    for i in range(len(image_files)):
        plt.imshow(image_files[i], cmap=plt.cm.get_cmap('bone'))
        img_name = f'brain_image{i}.png'
        image_path = os.path.join(app.static_folder, 'images', img_name)
        plt.savefig(image_path)
        if len(line) < 5:
            line.append(img_name)
        else:
            images.append(line)
            line = [img_name]
    if 0 < len(line) <= 5:
        images.append(line)
    return images


def clean(files):
    for file in files:
        try:
            os.unlink(file)
        except FileNotFoundError:
            continue


@aspectlib.Aspect
def model_cache(name):
    if not cache or cache["name"] != name:
        models = yield aspectlib.Proceed(name)
        cache["name"] = name
        cache["value"] = models
        yield aspectlib.Return(models)
    else:
        yield aspectlib.Return(cache["value"])


@aspectlib.Aspect
def model_log(files):
    start_time = time.time()
    results = yield aspectlib.Proceed(files)
    end_time = time.time() - start_time
    type_prediction = "single" if len(files) == 1 else "sequential"
    date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = "[%s] Request for %s prediction with results: %s Execution Time: %f\n"
    with open("log.txt", "a") as logfile:
        logfile.write(message % (date, type_prediction, str(results), end_time))


if __name__ == '__main__':
    app.run(use_reloader=False, debug=False, threaded=False)
