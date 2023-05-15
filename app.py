from flask import Flask, request, render_template

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file.save(file.filename)
            return 'Archivo cargado correctamente.'
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
