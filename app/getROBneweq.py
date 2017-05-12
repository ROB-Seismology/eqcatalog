#! /usr/bin/env python

import os, sys, platform
import datetime

## Add python path to allow running from cron on linux machine
if platform.uname()[0] == "Linux":
	pythondir = os.path.join(os.environ["HOME"], "python", "seismo")
	sys.path.insert(0, pythondir)
	logfile = os.path.join(os.environ["HOME"], "tmp", "ROBneweq.log")
else:
	logfile = r"C:\Temp\ROBneweq.log"

from thirdparty.recipes.sendmail import sendmail
#import users.kris.Seismo.db.seismodb as seismodb
import eqcatalog.seismodb as seismodb


## Users
USERS = [dict(email="kris.vanneste@oma.be", tel="+32473499229")]


## Mail settings
mailserverURL = "smtp.oma.be"
sender = "seismo@oma.be"
#recipients = ["kris.vanneste@proximus.be", "kris.vanneste@oma.be"]
recipients = ["smsmail@oma.be"]
subject = ', '.join([user['tel'] for user in USERS])


def construct_msg(eq):
	date, time = eq.datetime.date(), eq.datetime.time()
	#url = "http://seismologie.oma.be/active.php?LANG=NL&CNT=BE&LEVEL=211&id=%d" % last_eq.ID
	hash = last_eq.get_rob_hash()
	url = "http://seismologie.oma.be/en/seismology/earthquakes-in-belgium/%s" % hash

	msg = "ROB\n"
	msg += "ID %d\n" % eq.ID
	msg += "ML %.1f\n" % eq.ML
	msg += "%s\n" % eq.name
	msg += "%d-%02d-%02d %02d:%02d:%02d\n" % (date.year, date.month, date.day, time.hour, time.minute, int(round(time.second)))
	msg += "Lon: %.3f, Lat: %.3f\n" % (eq.lon, eq.lat)
	msg += "Depth: %s\n" % eq.depth
	try:
		msg += "\n%s" % url
	except:
		pass

	return msg


## Determine actual time
now = datetime.datetime.utcnow()

## Read log file containing events sent previously
try:
	logf = open(logfile)
except:
	prev_last_earthID = 0
else:
	prev_last_earthID = int(logf.readline())
	logf.close()


## Obtain most recent id_earth from database
last_earthID = seismodb.get_last_earthID()
#print last_earthID

if last_earthID != prev_last_earthID:
	try:
		last_eq = seismodb.query_ROB_LocalEQCatalogByID(id_earth=last_earthID)[0]
	except:
		pass
	else:
		if last_eq.name != None and last_eq.ML >= 1.0:
			mail_body = construct_msg(last_eq)
			sendmail(mailserverURL, sender, recipients, subject=subject, text=mail_body)
			sendmail(mailserverURL, sender, [user['email'] for user in USERS], subject="ROB New Earthquake", text=mail_body)
			logf = open(logfile, "w")
			logf.write("%d\n" % last_earthID)
			logf.close()