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
    checkpoint = 'checkpoints/2.1/%s'%(request.form.get('checkpoint'),)
    cfg = request.form.get('config')

    script_parameters = [
        'python',
        'infer_SAM21_slicer.py',
        '--cfg', 
        cfg,
        '--img_path',
        input_name,
        '--gts_path',
        gts_name,
        '--propagate',
        propagate,
        '--checkpoint',
        checkpoint,
        '--pred_save_dir',
        'data/video/segs_tiny',
    ]
    process = subprocess.Popen(script_parameters, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print('=================================\n', stderr, '\n=================================')

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

@app.route('/upload_model', methods=['POST'])
def upload_model():    
    file = request.files['file']
    model_name = os.path.basename(file.filename).split('.')[0]
    checkpoint_dir = "./checkpoints/2.1/%s"%model_name

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    file.save(os.path.join(checkpoint_dir, os.path.basename(file.filename)))
    return 'Model uploaded successfully'

@app.route('/upload_config', methods=['POST'])
def upload_config():    
    file = request.files['file']
    config_dir = "./sam2"

    Path(config_dir).mkdir(parents=True, exist_ok=True)

    file.save(os.path.join(config_dir, 'custom_' + os.path.basename(file.filename)))
    return 'Config file uploaded successfully'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
