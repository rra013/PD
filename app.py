from flask import Flask, render_template, request, redirect, flash, url_for, render_template, request, jsonify
import os
from backend import execute

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Necessary for using flashed messages

progress = {'status': 'Ready to begin.'}

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def display_form():
    message = flash("message")  # Retrieve the message if it exists
    return render_template('index.html', message=message)

@app.route('/process', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        flash('No file part')
    else:
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
        else:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            run_python_script(filename, 
                              request.form['param1'], 
                              request.form['param2'], 
                              request.form['param3'],
                              request.form['param4'])
            flash('File uploaded and processed successfully!')

    return redirect('/')

@app.route('/status')
def status():
    return jsonify(progress)

def run_python_script(video_path, param1, param2, param3, param4):
    # Replace with your Python code that processes the video
    damagedFrames = execute(video_path, float(param1), int(param2), int(param3), float(param4), progress)
    progress['status'] = progress['status'] + '\nList of filenames where damage has exceeded threshold: '
    for name in damagedFrames:
        progress['status'] = progress['status'] + '\n' + name
    return "Processing started", 202

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs("Damaged Asphault", exist_ok=True)
    app.run(debug=True)