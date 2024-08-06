from flask import Flask, request, send_file
import shutil
from pathlib import Path
import subprocess
import os

app = Flask(__name__)


@app.route('/run_script', methods=['POST'])
def run_script():
    input_name = request.form.get('input')
    gts_name = request.form.get('gts')
    propagate = request.form.get('propagate')
    checkpoint = request.form.get('checkpoint')

    script_parameters = [
        'python',
        'infer_video_tiny_debug.py',
        '--img_path',
        input_name,
        '--gts_path',
        gts_name,
        '--propagate',
        propagate,
        '--checkpoint',
        checkpoint,
    ]
    process = subprocess.Popen(script_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    #TODO: remove custom model?
    
    if process.returncode == 0:
        return f'Success: {stdout.decode("utf-8")}'
    else:
        return f'Error: {stderr.decode("utf-8")}'

@app.route('/download_file', methods=['GET'])
def download_file():
    output_name = request.form.get('output')
    return send_file(output_name, as_attachment=True)

@app.route('/upload', methods=['POST'])
def upload_file():    
    file = request.files['file']

    if file:
        file.save(file.filename)
        return 'File uploaded successfully'


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
