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
    name=estimater(name)
    return render_template("result.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)