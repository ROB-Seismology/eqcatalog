import os
import datetime
from eqcatalog.seismodb import query_ROB_LocalEQCatalog

out_folder = "D:\\GIS-data\\KSB-ORB"
#out_folder = r"E:\Home\_kris\Meetings\2018 - Opendeurdagen"
#out_folder = r"C:\Program Files (x86)\SeismicEruption\OpenDoorDays"

filespec = os.path.join(out_folder, "ROB.HY4")

region = (0,8,49,52)
start_date = datetime.date(1985, 1, 1)
#end_date = datetime.date(2007, 10, 1)
end_date = datetime.datetime.now()
min_mag = 0.0
max_mag = 7.0
catalog = query_ROB_LocalEQCatalog(region, start_date=start_date,
                            end_date=end_date, Mmin=min_mag, Mmax=max_mag)
catalog.export_HY4(filespec)
