from flask import Flask, render_template, request, redirect, url_for
import os
from utils import predict_mask
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            mask_path = predict_mask(filepath)
            return render_template('result.html', image_url=filepath, mask_url=mask_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
