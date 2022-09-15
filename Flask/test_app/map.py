import folium
def mesh2sq(meshcode,dim):
    meshcode = str(meshcode)
    if dim==1:
        p = meshcode[0:2]
        u = meshcode[2:4]
        return p,u
    elif dim==2:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        return p,q,u,v,
    elif dim==3:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        r = meshcode[6]
        w = meshcode[7]
        return p,q,r,u,v,w
    else:
        p = meshcode[0:2]
        u = meshcode[2:4]
        q = meshcode[4]
        v = meshcode[5]
        r = meshcode[6]
        w = meshcode[7]
        m = meshcode[8]
        n = meshcode[9]
        return p,q,r,u,v,w,m,n
def first(meshcode):
    p,u=mesh2sq(meshcode,1)
    p=float(p)
    u=float(u)
    lat = p/1.5 * 3600
    lng = (u+100) * 3600
    south = lat
    west = lng
    north = lat + 40 * 60
    east = lng + 1 * 3600
    return south/3600, west/3600, north/3600, east/3600
def second(meshcode):
    p,q,u,v=mesh2sq(meshcode,2)
    p=float(p)
    u=float(u)
    q=float(q)
    v=float(v)
    lat = p/1.5*3600 + q*5*60
    lng = (u+100)*3600 + v*7.5*60
    south = lat
    west = lng
    north = lat + 5 * 60
    east = lng + 7.5 * 60
    return south/3600,west/3600,north/3600,east/3600
def third(meshcode):
    p,q,r,u,v,w=mesh2sq(meshcode,3)
    p=float(p)
    u=float(u)
    q=float(q)
    v=float(v)
    r=float(r)
    w=float(w)
    lat = p/1.5*3600 + q*5*60 + r*30
    lng = (u+100)*3600 + v*7.5*60 + w*45
    south = lat
    west = lng
    north = lat + 30
    east = lng + 45
    return south/3600,west/3600, north/3600, east/3600
"""
def harf(meshcode):
    p,q,r,u,v,w,m,n=mesh2sq(meshcode)
    lat = p/1.5*3600 + q*5*60 + r*30
    lng = (u+100)*3600 + v*7.5*60 + w*45
    south = lat + (m > 2 ? 1 : 0) * 15
    north = lat + (m > 2 ? 2 : 1) * 15
    west = lng + (m%2 == 0 ? 1 : 0) * 22.5
    east = lng + (m%2 == 0 ? 2 : 1) * 22.5
    return {"south":south/3600, "west":west/3600, "north":north/3600, "east":east/3600};
  }

  exports.quater = function (meshcode){
    var south, west, north, east;
    var lat, lng;
    var code = sliceMeshcode(meshcode);
    lat = code.p/1.5*3600 + code.q*5*60 + code.r*30;
    lng = (code.u+100)*3600 + code.v*7.5*60 + code.w*45;
    south = lat + ((code.m > 2 ? (code.n > 2 ? ((code.n + code.m) > 5 ? 3 : 2) : 2) : (code.n > 2 ? 1 : 0))) * 7.5;
    north = lat + ((code.m > 2 ? (code.n > 2 ? ((code.n + code.m) > 5 ? 3 : 2) : 2) : (code.n > 2 ? 1 : 0)) + 1) * 7.5;
    west = lng + ((code.m%2 == 0 ? (code.n%2 == 0 ? ((code.n%2 + code.m%2) > 1 ? 3 : 2) : 2) : (code.n%2 == 0 ? 1 : 0))) * 11.25;
    east = lng + ((code.m%2 == 0 ? (code.n%2 == 0 ? ((code.n%2 + code.m%2) > 1 ? 3 : 2) : 2) : (code.n%2 == 0 ? 1 : 0)) + 1) * 11.25;
    return {"south":south/3600, "west":west/3600, "north":north/3600, "east":east/3600};
  }

})(typeof exports === 'undefined' ? this.meshcode2latlng = {} : exports);"""
def view(meshcode,select):
    if select=="東京":
        south,west,north,east=second(meshcode)
    elif select=="全国":
        south,west,north,east=first(meshcode)
    start_coords =  ((south+north)/2,(west+east)/2)
    folium_map = folium.Map(location=start_coords, zoom_start=10)
    sw=(south,west)
    se=(south,east)
    ne=(north,east)
    nw=(north,west)
    folium.Polygon(
    locations=[sw, se, ne, nw], # 多角形の頂点
    color="red", # 線の色
    weight=10, # 線の太さ
    fill=True, # 塗りつぶす
    fill_opacity=0.5 # 透明度（1=不透明）
    ).add_to(folium_map)
    return folium_map

def polygon(meshcode,select="全国"):
    if select=="東京":
        south,west,north,east=second(meshcode)
    elif select=="全国":
        south,west,north,east=first(meshcode)
    start_coords =  [(south+north)/2,(west+east)/2]
    sw=[south,west]
    se=[south,east]
    ne=[north,east]
    nw=[north,west]
    return start_coords,sw,se,ne,nw
