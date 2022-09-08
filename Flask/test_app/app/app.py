from flask import Flask,render_template,request
from estimate import estimater
app = Flask(__name__)


@app.route("/")
def index():
    name = request.args.get("name")
    return render_template("index.html",name=name)

@app.route("/index",methods=["post"])
def post():
    name = request.form["name"]
    serect = request.form.get('radio')
    if serect=="東京":
        name=estimater(name,serect)
    elif serect=="全国":
        name=estimater(name,serect)
    return render_template("index.html", name=name,serect=serect)

if __name__ == "__main__":
    app.run(debug=True)