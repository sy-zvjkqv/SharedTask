from flask import Flask,render_template,request, url_for,Blueprint
import folium
from estimate import estimater
from map import view,first,second,third,polygon
app = Flask(__name__)
#app.config["APPLICATION_ROOT"]="texttolocation/"
#my_blueprint=Blueprint('my_blueprint',__name__,template_folder="templates",url_prefix="/texttolocation/")
#app.register_blueprint(my_blueprint)

#my_blueprint.route('/')

PREFIX="/texttolocation"
@app.route(PREFIX+"/")
def index():
    #name = request.args.get("name")
    return render_template("index.html")
    #return "aa".format(url_for("index"))

@app.route("/map",methods=["POST"])
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
    #return render_template("map.html",text=text,code=code, start_coords= start_coords,sw=sw,se=se,ne=ne,nw=nw)
    return {'code':code}

@app.route(PREFIX+"/test" ,methods=["GET"])
def test():
    return {"hello": "world"}

@app.route(PREFIX+"/calc_geocode",methods=["POST"])
def calc_geocode():
    tweet_text=request.json["tweet_text"]
    region=request.json["region"]
    if region=="東京":
        geo_code=estimater(tweet_text,region)
    elif region=="全国":
        geo_code=estimater(tweet_text,region)
    return {'geo_code':geo_code}


if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True,port=5125)
