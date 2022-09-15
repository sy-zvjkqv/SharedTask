from flask import Flask,render_template,request
import folium
from estimate import estimater
from map import view,first,second,third,polygon
app = Flask(__name__)


@app.route("/")
def index():
    name = request.args.get("name")
    return render_template("index.html")

@app.route("/index",methods=["post"])
def post():
    text = request.form["name"]
    select = request.form.get('radio')
    if select=="東京":
        code=estimater(text,select)
    elif select=="全国":
        code=estimater(text,select)
    start_coords,sw,se,ne,nw=polygon(code,select)
    #folium_map=view(name,select)
    #folium_map.save('templates/map.html')
    return render_template("map.html",text=text,code=code, start_coords= start_coords,sw=sw,se=se,ne=ne,nw=nw)

if __name__ == "__main__":
    app.run(debug=True)
