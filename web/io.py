from flask import Flask, render_template
#from flask_webpack import Webpack

#webpack = Webpack()

app = Flask("hyperchamber.io")
#webpack.init_app(app)


@app.route('/')
def index():
    return render_template('index.html')



if __name__ == '__main__':
    app.run()
