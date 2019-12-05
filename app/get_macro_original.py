import sys
sys.path.append('/var/www/EQMapper/python')

import MySQLdb 
import EQMapper as E
from GMT import RGB
from numpy import array,min, max
import time, calendar
import os


db = MySQLdb.connect(user='seisread',passwd='readseisdb2',host='seisdb2.oma.be',db='seismo')
c = db.cursor()


try:
    id_earth=int(sys.argv[1])
    c.execute('SELECT id_earth, latitude, longitude, name, ML, date, time from earthquakes where id_earth=%i'%id_earth)
    force = True
except:
    c.execute('SELECT id_earth, latitude, longitude, name, ML, date, time from earthquakes where web_status=1')
    force = False
results = c.fetchall()

for result in results:
    id_earth, latitude, longitude, name, ML, Date, Time = result
    print "Getting data for %i" % id_earth
#     c.execute('SELECT min(latitude), max(latitude), min(longitude), max(longitude), count(id_earth) as Count, MAX(web_analyse.lastmod) from web_analyse inner join communes on web_analyse.id_com = communes.id where id_earth=%i and fiability>=10 and (latitude is not Null and longitude is not Null) group by communes.code_p having count(id_earth)>=3'%id_earth)
    c.execute('SELECT min(latitude), max(latitude), min(longitude), max(longitude), count(id_earth) as Count, MAX(web_analyse.lastmod) from web_analyse inner join communes on web_analyse.id_com = communes.id where id_earth=%i and fiability>=10 and (latitude is not Null and longitude is not Null) group by communes.code_p'%id_earth)
    macros = c.fetchall()
    if len(macros) !=0:
        macros = array(macros)
        print(macros)

        # ERROR HM : if macros[:5] contain None -> ERROR (see Thomas warning line 48)
        try:
            minlat, maxlat, minlon, maxlon, count, maxlastmod = min(macros[:,0]),max(macros[:,1]),min(macros[:,2]),max(macros[:,3]),sum(macros[:,4]),max(macros[:,5])
        except:
            minlat, maxlat, minlon, maxlon, count = min(macros[:,0]),max(macros[:,1]),min(macros[:,2]),max(macros[:,3]),sum(macros[:,4])
            maxlastmod = 9e9

#         print Date, Time
        Time = str(Time).split(':')
        Time = "%02i%02i" % (int(Time[0]),int(Time[1]))
        datetime = "%s_%s" % (Date, Time)
        print "id_earth = %i"%id_earth, " | %s "%name
        print "  datetime = %s" % datetime
        print "  lastmod = %s" % maxlastmod
        print "  count = %i" % count
        print "  Bounds : ", minlat, maxlat, minlon, maxlon

        try:
            name = "%s : ML%3.1f"%(name, float(ML))
        except:
            pass
        # HACK TO MATCH THE MODIFICATION DONE BY ALEXANDRE = LASTMOD NOT DEFINED IN THE FORMS, CRASHING! SHOULD BE REVERTED ASAP !!!!
        try:
            maxmod = calendar.timegm(time.strptime(str(maxlastmod),"%Y-%m-%d %H:%M:%S"))
        except:
            maxmod = 9e9
        map = False
        if os.path.exists('/home/seismo/lib/macro_maps/ROB/large/%i.png'%id_earth) and not force:
            creation_time = os.stat('/home/seismo/lib/macro_maps/ROB/large/%i.png'%id_earth)[-1]
#             print creation_time, maxmod
            if maxmod > creation_time:
                print "New data available for mapping !"
                map = True
            else:
                print "No new data available !"
                map = False
        else:
            map = True
        
        if map == True:
            print "New data has been found since last maps were created ! Creating maps !"
            #Large_scale :
            E.EQMapper('/home/seismo/lib/macro_maps/ROB/large/%i'%id_earth,E.GMT_region(minlon-1,maxlon+1,minlat-.5,maxlat+.5),"Mercator",1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',want_event=True,event_ID=id_earth,event_label=name,want_cities=True,city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',macro_web_minreplies=1, macro_web_minfiability=10.0,land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
            E.EQMapper('/home/seismo/lib/macro_maps/BNS/large/%s'%datetime,E.GMT_region(minlon-1,maxlon+1,minlat-.5,maxlat+.5),"Mercator",1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',want_event=True,event_ID=id_earth,event_label=name,want_cities=True,city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',macro_web_minreplies=1, macro_web_minfiability=10.0, land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
            
            #Small_scale :
            E.EQMapper('/home/seismo/lib/macro_maps/ROB/small/%i'%id_earth,E.GMT_region(longitude-1,longitude+1,latitude-.5,latitude+.5),"Mercator",1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',want_event=True,event_ID=id_earth,event_label=name,want_cities=True,city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',macro_web_minreplies=1, macro_web_minfiability=10.0, land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
            
            E.EQMapper('/home/seismo/lib/macro_maps/BNS/small/%s'%datetime,E.GMT_region(longitude-1,longitude+1,latitude-.5,latitude+.5),"Mercator",1000,want_macro=True,macro_ID=id_earth,macro_symbol_style='diamond',want_event=True,event_ID=id_earth,event_label=name,want_cities=True,city_labels=True,city_label_size=6,copyright_label='Collaborative project of ROB and BNS',macro_web_symbol_size_fixed=False,macro_cpt='KVN_intensity_ROB',macro_web_minreplies=1, macro_web_minfiability=10.0, land_color=RGB(255,255,255), ocean_color=RGB(255,255,255) )
    
    else:
        print "Not enough data to draw a map (<3 replies)"
    print "--------------------------------------"

