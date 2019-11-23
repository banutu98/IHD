from flask import Flask, render_template, request, redirect, flash, url_for
from ihd_web import *

UPLOAD_FOLDER = 'uploaded_files'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'


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
            predict(current_files)
            clean(current_files)
            return redirect(url_for('result'))
        flash('No dcm files selected!')
        return redirect(request.url)
    return render_template('ihd.html')


def predict(files):
    if len(files) == 1:
        predictions = predict_single_file(files[0])
        print(predictions)
    else:
        sequence_predictions = predict_file_sequence(files)
        print(sequence_predictions)
    return files


def clean(files):
    for file in files:
        os.unlink(file)


if __name__ == '__main__':
    app.run(use_reloader=False, debug=False, threaded=False)
