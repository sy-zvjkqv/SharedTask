from flask import Flask,render_template,request
import folium
from estimate import estimater
from map import view,first,second,third
app = Flask(__name__)


@app.route("/")
def index():
    name = request.args.get("name")
    return render_template("index.html")

@app.route("/index",methods=["post"])
def post():
    name = request.form["name"]
    select = request.form.get('radio')
    if select=="東京":
        name=estimater(name,select)
    elif select=="全国":
        name=estimater(name,select)
    folium_map=view(name,select)
    folium_map.save('templates/map.html')
    return render_template("index.html", name=name)

if __name__ == "__main__":
    app.run(debug=True)