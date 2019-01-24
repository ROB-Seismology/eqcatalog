# C      
# C	GEOCODING INDIVIDUAL ADDRESSES FROM MACROSEISMIC INQUIRIES
# C
# C      
# C      Koen VAN NOTEN 	ROYAL OBSERVATORY OF BELGIUM
# C
# C		 V0.2. JANUARY 2014
# C

import csv
import json
import time
import os
import math
import geocoder

start = time.strftime("%c")

# Input parameters macroseismic investigation
year = 2018
mag = 3.1
ID = 6625 # Kinrooi

# providers = 'google'
# providers = 'Komoot'
providers = 'ArcGIS'
# providers = 'osm'


#Specifying folder parameters for macroseismic inquiry file
folder = r'C:\OMA\Data Analyse\Macroseismic maps_Inquiries\20180526 Kinrooi 3.1'

# in_filespec = r'C:\OMA\data analyse\macroseismic inquiries\20180526 Kinrooi 3.1\macroseismic_inq_for_6625_modified_floorcorrected_no dupl_test.csv'
in_filespec = 'macroseismic_inq_for_6625_modified_floorcorrected_no dupl.csv'
print (in_filespec)

#2018 Kinrooi EQ
epi_lat = 51.175
epi_lon = 5.687
hypo_depth = 16.6

out_filespec = os.path.splitext(in_filespec)[0] + "_geocoded-" + providers + ".csv"
error_filespec = os.path.splitext(in_filespec)[0] + "_error-" + providers + ".csv"
print (out_filespec)

#Earthquake locations
#epi_lon = 6.620
#epi_lat = 50.979
#hypo_depth = 1.0

#epi_lon = 4.3716
#epi_lat = 50.8429	# Brussel 2011-02-18
#hypo_depth = 0.0
#epi_lon = 1.2700	# Folkstone 2015-05-22
#epi_lat = 51.3600
#hypo_depth = 15.0
#epi_lon = 6.7900	# explosion in Germany 2014-01-03
#epi_lat = 50.6560
#epi_lon = 3.1458 # veldegem
#epi_lat = 51.0742
#hypo_depth = 1.0
#epi_lat = 51.659 #Goch
#epi_lon = 6.156
#hypo_depth = 18.5
#epi_lat = 50.6383 #Crt-St-E 2014
#epi_lon = 4.5643
#hypo_depth = 4.8
#epi_lat = 50.886 # Alsdorf 2002
#epi_lon = 6.207
#hypo_depth = 16.4
#epi_lat = 51.2305 #Dusseldorf 2016
#epi_lon = 6.8508
#hypo_depth = 6.2

'''
# For Brabant Walloon - seismic swarm:
reloc = r'C:\OMA\Data Analyse\Events\hypoDD.reloc.txt'
keys = [] 	#event ID's in relocations
lons = []	#event lons in relocations
lats = []	#event lats in relocations
depth = []	#depth lats in relocations

with open(reloc) as file:
        for line in file:
                columns = line.split()
                keys.append(int(columns[1]))	#name the ID column
                lons.append(float(columns[3]))	#name the lon column
                lats.append(float(columns[2]))	#name the lat column
                depth.append(float(columns[4]))    #name the depth column
index = keys.index(ID)      # index the ID line that has to be searched
epi_lon = lons[index]       # find corresponding lons, lats
epi_lat = lats[index]
hypo_depth = depth[index]
'''

### Calculate epicentral distance from each geocoded address (code from mapping/geo)
def distance(origin, destination):	
    lon1, lat1 = origin[:2]
    lon2, lat2 = destination[:2]
    radius = 6371 # km

    dlat = math.radians(lat2-lat1)
    dlon = math.radians(lon2-lon1)
    a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = radius * c
    return d

# Geocoding the macroseismic addresses
with open(in_filespec) as file:
    responses = csv.reader(file)
    geocod_file = open(out_filespec, "w", newline='')
    error_file = open(error_filespec, "w", newline='')
    writer = csv.writer(geocod_file)
    writer2 = csv.writer(error_file)
    header = True
    total_n = 0
    geocoded_n = 0
    for response in responses:
        #Defining the headers in the macroseismic csv file:
        if header:
            geolat_column = response.index('Geo. Latitude')
            geolon_column = response.index('Geo. Longitude')
            country_column = response.index('country')
            response[country_column] = "Country"		#Renaming the country column name, as this column name is used 2 times.
            street_column = response.index('street')
            city_column = response.index('city')
            zip_column = response.index('zip')
            MI_column = response.index('MI')
            com_name_column = response.index('com_name')
            type = geolon_column+1
            response.insert(type, "address type")
            response[geolat_column] = "Geo_Latitude"
            response[geolon_column] = "Geo_Longitude"
            writer2.writerow(response) #writing the header of the error file of ungeocoded addresses
            depi = MI_column+1
            dhypo = MI_column+2
            response.insert(depi, "d_epi")
            response.insert(dhypo, "d_hypo")
            header = False
            writer.writerow(response)#writing the header of the geocoded file

        #Applying the code to each address
        else:
            total_n += 1 # total_n = total_n + 1
            street = response[street_column]
            #print (street)
#            if street != "": #activate to geocode all addresses and keep the empty response
            if street == "": #activate to only keep the geocoded addresses
                response.insert(type, "no address")
                writer2.writerow(response)
                print("nr: %s" % total_n)
                print ("    NO ADDRESS")
            else:
                city = response[city_column]
                com_name = response[com_name_column]
                country = response[country_column]
                zip = response[zip_column]
                input_address = street + " " + zip + " " + com_name + " " + country
#                address = street + " " + zip + " " + country
                address = street + " " + com_name + " " + country
                geolat = response[geolat_column]
                geolon = response[geolon_column]
                if 'google' in providers:	#GOOGLE PROVIDER
                    time.sleep(5)
                    address = address.decode('Latin-1', errors='replace').encode('ascii', errors='xmlcharrefreplace') #unclick this line when tomtom geocoding
                    address = urllib.quote(address)
                    geocode_url = "http://maps.googleapis.com/maps/api/geocode/json?address=%s&sensor=false" % address
                    req = urllib3.urlopen(geocode_url)
                    res = json.loads(req.read())
                    print (res)
                    status = res["status"]
                    if status == 'ZERO_RESULTS':
                                        print ("	nr: %s"%total_n)
                                        print ("WRONG ADDRESS-ZERO results" + address)
    #									webbrowser.open('https://www.google.be/maps/preview#!q=%s' % address)
                                        response.insert(type,"ZERO results")
                                        writer2.writerow(response)
                    else:
                                location_type = res["results"][0]["geometry"]["location_type"]
                                print (location_type)
                                if location_type in ('RANGE_INTERPOLATED', 'ROOFTOP','GEOMETRIC_CENTER'):
                                        geocoded_n += 1
                                        location = res["results"][0]["geometry"]["location"]
                                        lon, lat = location["lng"], location["lat"]
                                        format_address = res["results"][0]["formatted_address"]
                                        google_address = format_address.encode('ascii', errors='xmlcharrefreplace').decode('Latin-1', errors='replace')
                                        print ("	nr: %s"%geocoded_n + " of %s"%total_n)
                                        print (input_address)
                                        print (google_address + " = %s"%location_type)
    #
                                        # output file
                                        response[geolat_column] = lat
                                        response[geolon_column] = lon
                                        response.insert(type,location_type)
                                        d_epi = distance((epi_lon,epi_lat),(lon,lat))
                                        d_hypo = math.sqrt(math.pow(d_epi,2) + math.pow(hypo_depth,2))
                                        response.insert(depi, d_epi) # Create new column with epicentral distance
                                        response.insert(dhypo, d_hypo)
                                        writer.writerow(response)
                                else:
                                        response.insert(type,location_type)
                                        location = res["results"][0]["geometry"]["location"]
                                        lon, lat = location["lng"], location["lat"]
                                        response[geolat_column] = lat
                                        response[geolon_column] = lon
                                        writer2.writerow(response)
                else:	#OSM, ArcGIS PROVIDERs or GOOGLE in geocoder python
                    for provider in [providers]:
                        g = geocoder.get(address, provider=provider)
                        lon, lat = g.lng, g.lat
                        location_type = g.quality
#						print g.quality

                        if g.address:
                            ## Define qualities for different providers
                            if location_type in ('SubAdmin','poi'):
                            # if location_type in ('no_idea_do_everything'):
#							if location_type in ('PointAddress', 'StreetName', 'StreetAddress', 'addresspoint', 'house', 'street','pointAddress', 'interpolated',''):
#							esri specifications: 'PointAddress', 'StreetName', 'StreetAddress'
#							tomtom specifications: 'addresspoint', 'house', 'street'
#							Nokia specifications: 'pointAddress', 'interpolated',''
                            # if location_type in ('country'):
#                           Google specifications: 'street_address','route','establishment'

#								webbrowser.open('https://www.google.be/maps/preview#!q=%s' % address)
                                response[geolat_column] = lat
                                response[geolon_column] = lon
                                response.insert(type,location_type)
                                writer2.writerow(response)
                            else:
                                geocoded_n += 1
                                print ("nr: %s"%geocoded_n + " of %s"%total_n)
                                print ("     " + input_address)
                                print ("     corrected address (%s"%location_type + "):")
                                print (b"   " + g.address.encode('ascii', errors='replace'))
                                response.insert(type,location_type)
                                response[geolat_column] = lat
                                response[geolon_column] = lon
                                d_epi = distance((epi_lon,epi_lat),(lon,lat))
                                d_hypo = math.sqrt(math.pow(d_epi,2) + math.pow(hypo_depth,2))
                                response.insert(depi, d_epi) # Create new column with epicentral distance
                                response.insert(dhypo, d_hypo)
                                writer.writerow(response)
                        else:
                            response.insert(type,"no results")
                            response[geolat_column] = lat
                            response[geolon_column] = lon
                            writer2.writerow(response)


print (geocoded_n, 'of', total_n, 'addresses geocoded,', total_n - geocoded_n, "errors")
end = time.strftime("%c")
print (start)
print (end)