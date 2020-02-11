# -*- coding: iso-Latin-1 -*-
"""
ROB Warning Center

Automated dispatching of warning messages to different clients,
who may be subscribed to different warning criteria.

Meant to be run as a cron job on a linux machine.
Example crontab entry:
*/5 * * * * /usr/bin/python -W ignore /home/kris/python/seismo/eqcatalog/app/warning_center.py --run > /dev/null
"""

from __future__ import absolute_import, division, print_function, unicode_literals


import os, platform
import datetime, pytz
import numpy as np

## Add python path to allow running from cron on linux machine
home_folder = os.path.expanduser('~')
if platform.uname()[0] == "Linux":
	import sys
	pythondirs = [os.path.join(home_folder, "python", folder)
					for folder in ("seismo", "thirdparty")]
	for pythondir in pythondirs:
		sys.path.insert(0, pythondir)

import db.simpledb as simpledb
import eqcatalog
from thirdparty.recipes.sendmail import sendmail



## Note: we use python datetime instead of numpy datetime64 in eqcatalog
## because the latter is not natively supported by sqlite

## POLLING_INTERVAL is the interval between subsequent runs of the script
## Set at 5 minutes (see crontab entry), but currently not used
POLLING_INTERVAL = datetime.timedelta(0, 5 * 60)
## MAX_EVENT_AGE is the time interval before the current time
## to look for new events that should be reported
## Set at 2 days
MAX_EVENT_AGE = datetime.timedelta(2)


class WarningCenter():
	"""
	Class managing the warning center

	:param db:
		instance of :class:`simpledb.SQLDB`
	"""
	def __init__(self, db):
		self.db = db

	## Define database tables
	ColDef = {}
	ColDef['warcen_criteria'] = [
		dict(name='id', type='INTEGER', notnull=1, pk=1),
		dict(name='event_type', type='TEXT', dflt_value='ke'),
		dict(name='Mmin', type='REAL', notnull=1),
		dict(name='lon0', type='FLOAT', notnull=1),
		dict(name='lat0', type='FLOAT', notnull=1),
		dict(name='lon1', type='FLOAT', dflt_value=None),
		dict(name='lat1', type='FLOAT', dflt_value=None),
		dict(name='dist', type='FLOAT', dflt_value=0),
		dict(name='description', type='TEXT', notnull=1)]

	ColDef['warcen_clients'] = [
		dict(name='id', type='INTEGER', notnull=1, pk=1),
		dict(name='name', type='TEXT', notnull=1),
		dict(name='email', type='TEXT'),
		dict(name='phone', type='TEXT'),
		dict(name='delay', type='INTEGER', dflt_value=5),
		dict(name='lang', type='TEXT', dflt_value='EN')]

	ColDef['warcen_criteria_clients'] = [
		dict(name='id_crit', type='INTEGER', notnull=1),
		dict(name='id_client', type='INTEGER', notnull=1)]

	ColDef['warcen_events'] = [
		dict(name='id_crit', type='INTEGER', notnull=1),
		dict(name='id_earth', type='INTEGER', notnull=1),
		dict(name='origin_time', type='TIMESTAMP', notnull=1),
		dict(name='flag_time', type='TIMESTAMP', notnull=1)]

	ColDef['warcen_messages'] = [
		dict(name='id_client', type='INTEGER', notnull=1),
		dict(name='id_earth', type='INTEGER', notnull=1),
		dict(name='time_sent', type='TIMESTAMP', notnull=1)]

	tables = ColDef.keys()

	def create_db(self):
		"""
		Create database tables
		"""
		db_tables = self.db.list_tables()
		for table_name in self.tables:
			if not table_name in db_tables:
				self.db.create_table(table_name, self.ColDef[table_name])

	def populate_db(self, criteria, clients, client_criteria):
		"""
		Populate database tables

		:param criteria:
			list with instances of :class:`WarningCriterion`
		:param clients:
			list with instances of :class:`WarningClient`
		:param client_criteria:
			dict mapping client IDs to lists with criterion IDs
		"""
		## Insert or update criteria
		db_criteria = self.get_warning_criteria()
		db_criteria_ids = [crit.id for crit in db_criteria]
		for criterion in criteria:
			if criterion.id in db_criteria_ids:
				if not criterion in db_criteria:
					criterion.update_db_record()
			elif not criterion in db_criteria:
				criterion.insert_in_db()

		## Insert or update clients
		db_clients = self.get_warning_clients()
		db_client_ids = [client.id for client in db_clients]
		for client in clients:
			if client.id in db_client_ids:
				if not client in db_clients:
					client.update_db_record()
			elif not client in db_clients:
				client.insert_in_db()

			## Subscribe clients to criteria
			subscribed_criteria = client.get_subscribed_criteria()
			subscribed_criteria_ids = [crit.id for crit in subscribed_criteria]
			client_subscription_ids = client_criteria[client.id]
			for id_crit in client_subscription_ids:
				if not id_crit in subscribed_criteria_ids:
					client.subscribe_to_criterion(id_crit)

			## Unsubscribe from criteria that are no longer in list
			for id_crit in subscribed_criteria_ids:
				if not id_crit in client_subscription_ids:
					client.unsubscribe(id_crit)

		## Remove obsolete clients and criteria
		active_client_ids = [client.id for client in clients]
		for client in db_clients:
			if not client.id in active_client_ids:
				self.remove_client(client)

		active_criteria_ids = [crit.id for crit in criteria]
		for crit in db_criteria:
			if not crit.id in active_criteria_ids:
				self.remove_criterion(crit)

	def clear_db(self):
		"""
		Clear database
		"""
		db_tables = self.db.list_tables()
		for table_name in db_tables:
			self.db.drop_table(self.table_name)
		self.create_db()

	def close_db(self):
		"""
		Close database
		"""
		self.db.close()

	def remove_client(self, client):
		"""
		Remove client from database
		"""
		for criterion in client.get_subscribed_criteria():
			client.unsubscribe(criterion.id)
		table_clause = 'warcen_clients'
		where_clause = 'id = %d' % client.id
		self.db.delete_records(table_clause, where_clause)

	def remove_criterion(self, criterion):
		"""
		Remove criterion from database
		"""
		for client in criterion.get_clients():
			client.unsubscribe(criterion.id)
		table_clause = 'warcen_criteria'
		where_clause = 'id = %d' % criterion.id
		self.db.delete_records(table_clause, where_clause)

	def read_config(self, config_file):
		"""
		Import criteria and clients from configuration file
		"""
		from runpy import run_path
		config = run_path(config_file)

		criteria = [WarningCriterion(self, **kwargs) for kwargs in config['CRITERIA']]
		clients = [WarningClient(self, **kwargs) for kwargs in config['CLIENTS']]
		client_criteria = config['CLIENT_CRITERIA']

		self.populate_db(criteria, clients, client_criteria)

	def get_warning_criteria(self):
		"""
		Get list of warning criteria

		:return:
			list with instances of :class:`WarningCriterion`
		"""
		table_clause = 'warcen_criteria'
		criteria = []
		for rec in self.db.query(table_clause):
			wc = WarningCriterion(self, **rec)
			criteria.append(wc)
		return criteria

	def get_warning_clients(self, id=[]):
		"""
		Get list of clients

		:param id:
			list, client IDs

		:return:
			list with instances of :class:`WarningClient`
		"""
		table_clause = 'warcen_clients'
		where_clause = ''
		if id:
			where_clause = 'id IN (%s)' % ','.join(map(str, id))
		clients = []
		for rec in self.db.query(table_clause, where_clause=where_clause):
			client = WarningClient(self, **rec)
			clients.append(client)
		return clients

	def get_criterion_id(self, criterion_description):
		"""
		Obtain ID of criterion corresponding to given description

		:param criterion_description:
			str, description of criterion

		:return:
			int, criterion ID
		"""
		table_clause = 'warcen_criteria'
		column_clause = 'id'
		where_clause = 'description = "%s"' % criterion_description
		db_recs = self.db.query(table_clause, column_clause, where_clause=where_clause)
		db_recs = list(db_recs)
		if len(db_recs):
			return db_recs[0]['id']

	def poll_catalog(self, verbose=False, dry_run=False):
		"""
		Main function that polls the earthquake catalog
		and decides if warnings are sent to different clients
		based on the criteria they are subscribed to.

		This should be launched as a cron job every POLLING_INTERVAL

		:param verbose:
			bool, whether or not to print additional information
		:param dry_run:
			bool, whether or not to actually send (True) or just print (False)
			the warning messages
			(default: True)
		"""
		## Query database for events less than MAX_EVENT_AGE old
		start_date = datetime.datetime.utcnow() - MAX_EVENT_AGE
		catalog = eqcatalog.rob.query_local_eq_catalog(start_date=start_date,
											event_type="all", verbose=verbose)
		if verbose:
			catalog.print_list()
		warning_criteria = self.get_warning_criteria()

		## Check if there are events in catalog that need to be flagged
		if len(catalog):
			#num_dyfi = eqcatalog.rob.get_num_online_macro_enquiries(catalog.get_ids())
			for criterion in warning_criteria:
				subcatalog = criterion.filter_catalog(catalog)
				#max_delay = criterion.get_max_delay()
				#max_flag_age = datetime.timedelta(0, max_delay * 60)
				max_flag_age = datetime.timedelta(0)
				flagged_events = criterion.get_flagged_events(max_flag_age,
															verbose=verbose)
				flagged_event_ids = [fe.id_earth for fe in flagged_events]
				for eq in subcatalog:
					if not eq.ID in flagged_event_ids:
						origin_time = eqcatalog.time_functions.to_py_datetime(
																	eq.datetime)
						criterion.flag_event(eq.ID, origin_time)
						if verbose:
							msg = "Event flagged for criterion %s: #%d"
							msg %= (criterion.description, eq.ID)
							print(msg)

		## Loop over all criteria and flagged events,
		## and determine if warning needs to be sent
		sent_messages = {}
		for criterion in warning_criteria:
			subcatalog = criterion.filter_catalog(catalog)
			for client in criterion.get_clients():
				max_flag_age = datetime.timedelta(0, client.delay * 60)
				flagged_events = criterion.get_flagged_events(max_flag_age,
															verbose=verbose)
				## If event is flagged...
				for event in flagged_events:
					## ... and no message has been sent...
					if not event.is_message_sent(client.id):
						## ... and earthquake is still in database...
						eq = subcatalog.get_event_by_id(event.id_earth)
						if eq:
							## ... and event_type has not changed since flagging
							## (should not be necessary)
							#if criterion.event_type in ('all', eq.event_type):
							client.send_warning(eq, dry_run=dry_run)
							if not dry_run:
								if not eq.ID in sent_messages:
									sent_messages[eq.ID] = [client.id]
								else:
									sent_messages[eq.ID].append(client.id)
							if verbose:
								msg = "Message sent for event #%d to client %s"
								msg %= (eq.ID, client.name)
								print(msg)

		## Send mail to manager (client with id=0) if messages have been sent
		if sent_messages:
			subject = "WarningCenter message log"
			msg_lines = ["The following warning messages have been sent:"]
			for eq_id in sorted(sent_messages.keys()):
				eq = subcatalog.get_event_by_id(eq_id)
				msg_lines.append("- Event #%s (ML=%.1f, %s)" % (eq_id, eq.ML, eq.datetime))
				clients = self.get_warning_clients(id=sorted(sent_messages[eq_id]))
				for client in clients:
					msg_lines.append("\t- Client %s" % client.name)
			msg = '\n'.join(msg_lines)
			[list_manager] = self.get_warning_clients(id=[0])
			self.send_email(list_manager, msg, subject=subject, dry_run=False)

	def send_manual_warning(self, id_earth, client_ids, dry_run=True):
		"""
		Send manual warning for given earthquake to given clients

		:param id_earth:
			int, ID of event in ROB catalog
		:param client_ids:
			list of integers, client IDs
		:param dry_run:
			bool, whether or not to actually send (True) or just print (False)
			the warning message
			(default: True)
		"""
		for client in self.get_warning_clients():
			if client.id in client_ids:
				client.send_manual_warning(id_earth, dry_run=dry_run)

	@classmethod
	def compose_message(cls, eq, lang, msg_type, revision=False):
		"""
		Compose warning message

		:param eq:
			instance of :class:`eqcatalog.LocalEarthquake`
		:param lang:
			str, language, one of 'EN', 'NL' or 'FR'
		:param msg_type:
			str, message type, 'sms' or 'email'
		:param revision:
			bool, whether or not message is a revision
			(default: False)

		:return:
			str, message
		"""
		import thirdparty.recipes.tinyurl as tinyurl

		lang = lang.upper()

		if revision:
			msg = "%s\n\n" % MSG_FIELDS['title_revision'][lang]
		else:
			msg = "%s\n\n" % MSG_FIELDS['title'][lang]

		msg += "ML: %.1f\n" % eq.ML

		if msg_type == 'email':
			event_type_name = eqcatalog.get_earthquake_type(eq.event_type, lang)
			msg += "%s: %s\n" % (MSG_FIELDS['event_type'][lang], event_type_name)
		else:
			msg += "%s: %s\n" % (MSG_FIELDS['event_type'][lang], eq.event_type)

		dt = eqcatalog.time_functions.to_py_datetime(eq.datetime)
		msg += ("%s: %d-%02d-%02d %02d:%02d:%02d\n"
				% (MSG_FIELDS['time'][lang], dt.year, dt.month, dt.day,
				dt.hour, dt.minute, int(round(dt.second))))
		local_dt = utc_to_local_dt(dt)
		msg += ("%s: %d-%02d-%02d %02d:%02d:%02d\n"
				% (MSG_FIELDS['local_time'][lang], local_dt.year, local_dt.month,
				local_dt.day, local_dt.hour, local_dt.minute, int(round(local_dt.second))))

		msg += "%s: %s\n" % (MSG_FIELDS['region'][lang], eq.name)
		EW = 'E' if eq.lon >= 0 else 'W'
		NS = 'N' if eq.lat >= 0 else 'S'
		msg += "Lon: %.3f%c, Lat: %.3f%c\n" % (abs(eq.lon), EW, abs(eq.lat), NS)
		msg += "%s: %.0f km\n" % (MSG_FIELDS['depth'][lang], eq.depth)

		num_macro = eq.get_num_online_macro_enquiries()
		if num_macro:
			msg += "%s: %d\n" % (MSG_FIELDS['num_dyfi'][lang], num_macro)

		msg += "ID: %d\n" % eq.ID

		hash = eq.get_rob_hash()
		base_url = "http://seismologie.oma.be"
		url = '%s/%s/%s' % (base_url, MSG_FIELDS['url'][lang], hash)
		if msg_type == 'sms':
			try:
				short_url = tinyurl.shorten_url(url)
			except:
				msg += "\n%s\n%s\n" % (MSG_FIELDS['website_long'][lang], url)
			else:
				msg += "%s: %s" % (MSG_FIELDS['website_short'][lang], short_url)
		else:
			msg += "\n%s\n%s\n" % (MSG_FIELDS['website_long'][lang], url)
			msg += MSG_FIELDS['signature'][lang]

		return msg

	@classmethod
	def send_email(cls, client, msg, subject="ROB New Earthquake", dry_run=True):
		"""
		Send email to client

		:param client:
			instance of :class:`WarningClient`
		:param msg:
			str, text message
		:param subject:
			str, email subject
			(default: "ROB New Earthquake")
		:param dry_run:
			bool, whether or not to actually send (True) or just print
			(False) the message
			(default: True)
		"""
		mailserver_url = "smtp.oma.be"
		sender = "seismo@oma.be"
		#sender = "donotreply@oma.be"
		if client.email:
			recipients = client.email.split(',')
			if dry_run:
				print(recipients)
				print(subject)
				print(msg)
			else:
				sendmail(mailserver_url, sender, recipients, subject=subject, text=msg)

	@classmethod
	def send_sms(cls, client, msg, dry_run=True):
		"""
		Send SMS to client

		:param client:
			instance of :class:`WarningClient`
		:param msg:
			str, text message
		:param dry_run:
			bool, whether or not to actually send (True) or just print
			(False) the message
			(default: True)
		"""
		mailserver_url = "smtp.oma.be"
		sender = "seismo@oma.be"
		recipients = ["smsmail@oma.be"]
		if client.phone:
			subject = client.phone
			if dry_run:
				print(subject)
				print(msg)
			else:
				sendmail(mailserver_url, sender, recipients, subject=subject, text=msg)

	def get_pending_events(self):
		"""
		Determine pending events (= flagged events for which no message
		has been sent yet)

		:return:
			dict, mapping instances of :class:`FlaggedEvent` to client IDs
		"""
		events_clients = {}
		for criterion in self.get_warning_criteria():
			client_events = criterion.get_pending_events()
			for client_id, pending_events in client_events.items():
				for event in pending_events:
					if not event in events_clients:
						events_clients[event] = [client_id]
					else:
						events_clients[event].append(client_id)
		return events_clients

	def print_info(self):
		"""
		Print some useful information
		"""
		criteria = self.get_warning_criteria()
		clients = self.get_warning_clients()

		print("Warning criteria subscriptions:")
		for criterion in criteria:
			print(criterion)
			for client in criterion.get_clients():
				print("  %s" % client)
		print("")

		print("Warning clients:")
		for client in clients:
			print("  %s: %d messages so far"
					% (client, client.get_num_sent_messages()))
		print("")

		print("Pending events:")
		events_clients = self.get_pending_events()
		for event, client_ids in events_clients.items():
			print("  Event %s pending for:" % event.id_earth)
			for client_id in client_ids:
				hours, minutes, seconds = event.get_remaining_delay(client_id)
				[client] = [cl for cl in clients if cl.id == client_id]
				print("    %s: %02dh%02dm%02ds" % (client, hours, minutes, seconds))


# TODO: could also include whether or not event has been felt
class WarningCriterion():
	"""
	Class representing a criterion to issue a warning

	:param warcen:
		instance of :class:`WarningCenter`
	:param id:
		int, criterion identifier
	:param event_type:
		str, event type, e.g. 'ke', 'ke,ki', 'all'
	:param Mmin:
		float, minimum magnitude
	:param lon0:
		float, longitude 1 (left edge of box or center of circle)
	:param lat0:
		float, latitude 1 (bottom edge of box or center of circle)
	:param lon1:
		float, longitude 2 (right edge of box, None if circle)
		(default: None)
	:param lat1:
		float, latitude 2 (top edge of gox, None if circle)
		(default: None)
	:param dist:
		float, radius in km of circle
		(default: 0)
	:param description:
		str, unqiue string describing warning criterion
	"""
	table_name = 'warcen_criteria'

	def __init__(self, warcen, id, event_type, Mmin, lon0, lat0, lon1=None, lat1=None,
				dist=0, description=''):
		assert not (None in (lon1, lat1) and dist == 0)
		self.warcen = warcen
		self.id = id
		self.event_type = event_type
		self.Mmin = Mmin
		self.lon0 = lon0
		self.lat0 = lat0
		self.lon1 = lon1
		self.lat1 = lat1
		self.dist = dist
		self.description = description

	def __repr__(self):
		return '<WarningCriterion %d: %s>' % (self.id, self.description)

	def __eq__(self, other):
		return (self.id == other.id
			and self.event_type == other.event_type
			and np.allclose(self.Mmin, other.Mmin)
			and np.allclose(self.lon0, other.lon0)
			and np.allclose(self.lat0, other.lat0)
			and (self.lon1 is other.lon1 is None or np.allclose(self.lon1, other.lon1))
			and (self.lon1 is other.lon1 is None or np.allclose(self.lat1, other.lat1))
			and np.allclose(self.dist, other.dist)
			and self.description == other.description)

	def is_region(self):
		"""
		:return:
			bool, whether or not criterion corresponds to a geographic
			box
		"""
		return not None in (self.lon1, self.lat1)

	def get_region(self):
		"""
		:return:
			(minlon, maxlon, minlat, maxlat) tuple defining geographic box
		"""
		if self.is_region():
			return (self.lon0, self.lon1, self.lat0, self.lat1)

	def filter_catalog(self, catalog):
		"""
		Filter earthquake catalog based on warning criterion

		:param catalog:
			instance of :class:`eqcatalog.EQCatalog`, original catalog

		:return:
			instance of :class:`eqcatalog.EQCatalog`, filtered catalog
		"""
		## Filter minimum magnitude
		subcatalog = catalog.subselect(Mmin=self.Mmin, Mtype='ML')
		## Filter event type
		if self.event_type != 'all':
			event_types = self.event_type.split(',')
			subcatalog = subcatalog.subselect(attr_val=('event_type', event_types))
		## Filter geographic area
		if self.is_region():
			return subcatalog.subselect(region=self.get_region())
		else:
			pt = (self.lon0, self.lat0)
			return subcatalog.subselect_distance(pt, self.dist)

	def to_dict(self):
		"""
		Convert criterion to dictionary

		:return:
			dict
		"""
		colnames = [coldef['name'] for coldef in WarningCenter.ColDef[self.table_name]]
		return {colname: getattr(self, colname) for colname in colnames}

	def insert_in_db(self):
		"""
		Insert criterion in database
		"""
		rec = self.to_dict()
		self.warcen.db.add_records(self.table_name, [rec])

	def update_db_record(self):
		"""
		Update database record
		"""
		rec = self.to_dict()
		where_clause = 'id = %d' % self.id
		self.warcen.db.update_row(self.table_name, rec, where_clause)

	def get_clients(self):
		"""
		Obtain list of clients for this criterion

		:return:
			list with instances of :class:`WarningClient`
		"""
		table_clause = 'warcen_clients'
		column_clause = 'warcen_clients.*'
		join_clause = [('JOIN', 'warcen_criteria_clients',
					'warcen_clients.id = warcen_criteria_clients.id_client')]
		where_clause = 'warcen_criteria_clients.id_crit = %d' % self.id
		clients = []
		for rec in self.warcen.db.query(table_clause, column_clause,
							join_clause=join_clause, where_clause=where_clause):
			client = WarningClient(self.warcen, **rec)
			clients.append(client)
		return clients

	def get_flagged_events(self, max_flag_age, verbose=False):
		"""
		Obtain flagged events for this criterion.

		:param max_flag_age:
			instance of :class:`datetime.timedelta`, maximum time since
			event was first flagged. Events that were flagged less than
			this time interval ago will not be included in the list.
			This allows to filter flagged events based on client delay.
		:param verbose:
			bool, whether or not to print SQL query
			(default: False)

		:return:
			list with instance of :class:`FlaggedEvent`
		"""
		## Add 30 s margin to allow different execution times
		## between subsequent polls
		margin = datetime.timedelta(0, 30)
		current_time = datetime.datetime.utcnow()
		flag_time = current_time - max_flag_age + margin
		origin_time = current_time - MAX_EVENT_AGE - POLLING_INTERVAL - margin
		table_clause = 'warcen_events'
		where_clause = ('id_crit = %d AND flag_time <= "%s"'
						' AND origin_time >= "%s"')
		where_clause %= (self.id, flag_time, origin_time)
		events = []
		for rec in self.warcen.db.query(table_clause, where_clause=where_clause,
									verbose=verbose):
			event = FlaggedEvent(self.warcen, **rec)
			events.append(event)
		return events

	def get_max_delay(self):
		"""
		Determine maximum delay of all clients for this criterion

		:return:
			int, maximum delay (in minutes)
		"""
		clients = self.get_clients()
		max_delay = max([client.delay for client in clients])
		return max_delay

	def flag_event(self, id_earth, origin_time):
		"""
		Flag event meeting this criterion

		:param id_earth:
			int, ID of event in ROB catalog
		:param origin_time:
			instance of :class:`datetime.datetime`, origin time of event
		"""
		table_name = 'warcen_events'
		flag_time = datetime.datetime.utcnow()
		rec = dict(id_crit=self.id, id_earth=id_earth, origin_time=origin_time,
					flag_time=flag_time)
		self.warcen.db.add_records(table_name, [rec])

	def get_pending_events(self):
		"""
		Determine pending events (= flagged events for which no message
		has been sent yet) for this criterion

		:return:
			dict, mapping client IDs to lists with instances of
			:class:`FlaggedEvent`
		"""
		pending_events = {}
		clients = self.get_clients()
		max_flag_age = datetime.timedelta(0)
		for event in self.get_flagged_events(max_flag_age):
			for client in clients:
				if not event.is_message_sent(client.id):
					if not client.id in pending_events:
						pending_events[client.id] = [event]
					else:
						pending_events[client.id].append(event)
		return pending_events


class WarningClient():
	"""
	Class representing a client receiving warnings

	:param warcen:
		instance of :class:`WarningCenter`
	:param id:
		int, client identifier
	:param name:
		str, client name
	:param email:
		str, client email address (multiple addresses are allowed if
		separated by ',')
	:param phone:
		str, client phone number (multiple phone numbers are allowed
		if separated by ',')
	:param delay:
		int, delay in minutes to apply between the time an event is
		flagged according to a given criterion, and the time a message
		is sent. This allows some time to improve the first assessment
		of an event, e.g. if a quarry blast is entered in the catalog
		as 'ke', there is still time to change its status before a
		message is sent erroneously
	:param lang:
		str, language, one of 'EN', 'NL', 'FR'
	"""
	table_name = 'warcen_clients'

	def __init__(self, warcen, id, name, email, phone, delay, lang):
		self.warcen = warcen
		self.id = id
		self.name = name
		self.email = email
		self.phone = phone
		self.delay = delay
		self.lang = lang

	def __repr__(self):
		return '<Client %d: %s>' % (self.id, self.name)

	def __eq__(self, other):
		return (self.id == other.id
				and self.name == other.name
				and self.email == other.email
				and self.phone == other.phone
				and self.delay == other.delay
				and self.lang == other.lang)

	def to_dict(self):
		"""
		Convert client to dictionary

		:return:
			dict
		"""
		colnames = [coldef['name'] for coldef in WarningCenter.ColDef[self.table_name]]
		return {colname: getattr(self, colname) for colname in colnames}

	def insert_in_db(self):
		"""
		Insert client in database
		"""
		rec = self.to_dict()
		self.warcen.db.add_records(self.table_name, [rec])

	def update_db_record(self):
		"""
		Update database record
		"""
		rec = self.to_dict()
		where_clause = 'id = %d' % self.id
		self.warcen.db.update_row(self.table_name, rec, where_clause)

	def get_subscribed_criteria(self):
		"""
		Obtain list of criteria the client is subscribed to
		"""
		table_clause = 'warcen_criteria'
		column_clause = 'warcen_criteria.*'
		join_clause = [('JOIN', 'warcen_criteria_clients',
					'warcen_criteria_clients.id_crit = warcen_criteria.id')]
		where_clause = 'warcen_criteria_clients.id_client = %d' % self.id
		criteria = []
		for rec in self.warcen.db.query(table_clause, column_clause,
							join_clause=join_clause, where_clause=where_clause):
			criterion = WarningCriterion(self.warcen, **rec)
			criteria.append(criterion)
		return criteria

	def subscribe_to_criterion(self, id_crit):
		"""
		Subscribe to a warning criterion

		:param id_crit:
			int, criterion ID
		"""
		if isinstance(id_crit, str):
			id_crit = self.warcen.get_criterion_id(id_crit)
		table_name = 'warcen_criteria_clients'
		rec = {'id_crit': id_crit, 'id_client': self.id}
		self.warcen.db.add_records(table_name, [rec])

	def unsubscribe(self, id_crit):
		"""
		Unsubscribe from a warning criterion

		:param id_crit,
			int, criterion ID
		"""
		if isinstance(id_crit, str):
			id_crit = self.warcen.get_criterion_id(id_crit)
		table_name = 'warcen_criteria_clients'
		where_clause = 'id_crit = %d AND id_client = %d'
		where_clause %= (id_crit, self.id)
		self.warcen.db.delete_records(table_name, where_clause)

	def mark_event_as_sent(self, id_earth):
		"""
		Mark that warning message has been sent for given event

		:param id_earth:
			int, event ID in ROB catalog
		"""
		table_name = 'warcen_messages'
		time_sent = datetime.datetime.utcnow()
		if not self.is_event_sent(id_earth):
			rec = dict(id_client=self.id, id_earth=id_earth, time_sent=time_sent)
			self.warcen.db.add_records(table_name, [rec])
		else:
			col_dict = dict(time_sent=time_sent)
			where_clause = 'id_client = %d AND id_earth = %d'
			where_clause %= (self.id, id_earth)
			self.warcen.db.update_row(table_name, col_dict, where_clause)

	def is_event_sent(self, id_earth):
		"""
		Check if warning message has been sent for given event

		:param id_earth:
			int, event ID in ROB catalog

		:return:
			bool
		"""
		flagged_event = FlaggedEvent(self.warcen, 0, id_earth, 0, 0)
		return flagged_event.is_message_sent(self.id)

	def get_num_sent_messages(self, start_date=None, end_date=None):
		"""
		Determine number of messages sent to this client

		:param start_date:
			instance of :class:`datetime.datetime`, start date
			(default: None)
		:param end_date:
			instance of :class:`datetime.datetime`, end date

		:return:
			int, number of messages
		"""
		if start_date is None:
			start_date = datetime.datetime(2019, 1, 1)
		if end_date is None:
			end_date = datetime.datetime.now()
		table_clause = 'warcen_messages'
		column_clause = 'Count(*) as num_sent_messages'
		where_clause = 'id_client = %d AND time_sent BETWEEN "%s" and "%s"'
		where_clause %= (self.id, start_date, end_date)
		db_recs = self.warcen.db.query(table_clause, column_clause, where_clause=where_clause)
		num_sent_messages = list(db_recs)[0]['num_sent_messages']
		return num_sent_messages

	def send_warning(self, eq, revision=False, dry_run=True):
		"""
		Send warning to client for given event

		:param eq:
			instance of :class:`eqcatalog.LocalEarthquake`
		:param revision:
			bool, whether or not message is a revision
			(default: False)
		:param dry_run:
			bool, whether or not to actually send (True) or just print (False)
			the warning message
			(default: True)
		"""
		if self.email:
			self.send_email(eq, revision=revision, dry_run=dry_run)
		if self.phone:
			self.send_sms(eq, revision=revision, dry_run=dry_run)

		if not dry_run:
			self.mark_event_as_sent(eq.ID)

	def send_email(self, eq, revision=False, dry_run=True):
		"""
		Send email warning to client for given event

		:param eq:
		:param revision:
		:param dry_run:
			see :meth:`send_warning`
		"""
		msg = self.warcen.compose_message(eq, self.lang, 'email', revision=revision)
		if revision:
			subject = MSG_FIELDS['subject_revision'][self.lang.upper()]
		else:
			subject = MSG_FIELDS['subject'][self.lang.upper()]
		self.warcen.send_email(self, msg, subject, dry_run=dry_run)

	def send_sms(self, eq, revision=False, dry_run=True):
		"""
		Send SMS warning to client for given event

		:param eq:
		:param revision:
		:param dry_run:
			see :meth:`send_warning`
		"""
		from eqcatalog.macro.dyfi import strip_accents
		msg = self.warcen.compose_message(eq, self.lang, 'sms', revision=revision)
		msg = strip_accents(msg)
		self.warcen.send_sms(self, msg, dry_run=dry_run)

	def send_manual_warning(self, id_earth, dry_run=True):
		"""
		Send manual warning to client for given event ID

		:param id_earth:
			int, ID of event in ROB catalog
		:param dry_run:
			bool, whether or not to actually send (True) or just print (False)
			the warning message
			(default: True)
		"""
		[eq] = eqcatalog.rob.query_local_eq_catalog_by_id(id_earth)
		if eq:
			if self.is_event_sent(id_earth):
				revision = True
			else:
				revision = False
			if self.email:
				self.send_email(eq, revision=revision, dry_run=dry_run)
			if self.phone:
				self.send_sms(eq, revision=revision, dry_run=dry_run)

			if not dry_run:
				self.mark_event_as_sent(eq.ID)


class FlaggedEvent():
	"""
	Class representing event that has been flagged for generating a warning

	:param warcen:
		instance of :class:`WarningCenter`
	:param id_crit:
		int, criterion ID
	:param id_earth:
		int, ID of earthquake in ROB catalog
	:param origin_time:
		instance of :class:`datetime.datetime`, origin time of event
	:param flag_time:
		instance of :class:`datetime.datetime`, time when event
		was flagged
	"""
	table_name = 'warcen_events'

	def __init__(self, warcen, id_crit, id_earth, origin_time, flag_time):
		self.warcen = warcen
		self.id_crit = id_crit
		self.id_earth = id_earth
		self.origin_time = origin_time
		self.flag_time = flag_time

	def is_message_sent(self, id_client):
		"""
		Determine if message was sent to given client for this event

		:param id_client:
			int, client ID

		:return:
			bool
		"""
		table_clause = 'warcen_messages'
		where_clause = 'id_client = %d AND id_earth = %d'
		where_clause %= (id_client, self.id_earth)
		db_recs = list(self.warcen.db.query(table_clause, where_clause=where_clause))
		if len(db_recs):
			return True
		else:
			return False

	def get_remaining_delay(self, id_client):
		"""
		Determine delay before message will be sent to given client
		for this event

		:param id_client:
			int, client ID

		:return:
			(hours, minutes, seconds) tuple
		"""
		[client] = self.warcen.get_warning_clients([id_client])
		client_delay = datetime.timedelta(0, client.delay * 60)
		current_delay = datetime.datetime.utcnow() - self.flag_time
		remaining_delay = client_delay - current_delay
		hours, seconds = divmod(remaining_delay.seconds, 3600.)
		minutes, seconds = divmod(seconds, 60.)
		hours += remaining_delay.days * 24
		return (hours, minutes, seconds)


def utc_to_local_dt(utc_dt):
	"""
	Convert naive datetime to localized datetime

	:param utc_dt:
		instance of :class:`datetime.datetime`, UTC datetime

	:return:
		instance of :class:`datetime.datetime`, datetime in Belgian time
	"""
	tz = pytz.timezone('Europe/Brussels')
	return utc_dt.replace(tzinfo=pytz.utc).astimezone(tz)


# TODO: add German
MSG_FIELDS = {
	'title': {'EN': 'New seismic event',
				'NL': 'Nieuwe seismische gebeurtenis',
				'FR': 'Nouvel évènement'},
	'title_revision': {'EN': 'Revision of seismic event',
						'NL': 'Gereviseerde seismische gebeurtenis',
						'FR': 'Evènement révisé'},
	'subject': {'FR': "Nouvel évènement enregistré par le réseau sismique belge",
			'NL': "Nieuwe gebeurtenis geregistreerd door Belgisch seismisch netwerk",
			'EN': "New event recorded by the Belgian seismic network"},
	'subject_revision': {'EN': "Revision of event recorded by Belgian seismic network",
			'NL': "Revisie van gebeurtenis geregistreerd door Belgisch seismisch netwerk",
			'FR': "Révision d'événement enregistrée par le réseau sismique belge"},
	'url': {'EN': "en/seismology/earthquakes-in-belgium",
				'NL': "nl/seismologie/aardbevingen-in-belgie",
				'FR': "fr/seismologie/tremblements-de-terre-en-belgique"},
	'event_type': {'EN': 'Type', 'NL': 'Type', 'FR': 'Type'},
	'time': {'EN': 'Time UTC', 'NL': 'Tijd UTC', 'FR': 'Temps UTC'},
	'local_time': {'EN': 'Local time', 'NL': 'Lokale tijd', 'FR': 'Temps local'},
	'region': {'EN': 'Region', 'NL': 'Regio', 'FR': 'Région'},
	'depth': {'EN': 'Depth', 'NL': 'Diepte', 'FR': 'Profondeur'},
	'website_short': {'EN': 'Website', 'NL': 'Website', 'FR': 'Site web'},
	'website_long': {'EN': 'This event was verified by a seismologist.\n'
						'The reported parameters are subject to change, however.\n'
						'Please consult our website for the most up-to-date information:',
					'NL': 'Deze gebeurtenis werd geverifieerd door een seismoloog.\n'
						'De meegedeelde parameters kunnen echter nog gewijzigd worden.\n'
						'Raadpleeg onze website voor de meest actuele informatie:',
					'FR': 'Cet événement a été vérifié par un sismologue.\n'
						'Cependant, les paramètres communiqués sont sujets à changement.\n'
						'Veuillez consulter notre site web pour obtenir les informations les plus récentes:'},
	'signature': {'EN': '\nOD Seismology & Gravimetry\n'
						'Royal Observatory of Belgium\n',
					'NL': '\nOD Seismologie & Gravimetrie\n'
						'Koninklijke Sterrenwacht van België\n',
					'FR': '\nDO Séismologie & Gravimétrie\n'
						'Observatoire royal de Belgique\n'},
	'num_dyfi': {'EN': 'Number of felt reports',
				'NL': 'Aantal meldingen',
				'FR': 'Nombre de rapports'}}



if __name__ == "__main__":
	import argparse

	parser = argparse.ArgumentParser(description="ROB Warning Center")
	parser.add_argument("--init", help="Create WarningCenter database",
						action="store_true")
	parser.add_argument("--info", help="Print registered criteria and clients",
						action="store_true")
	parser.add_argument("--run", help="Poll earthquake catalog and send messages if necessary",
						action="store_true")
	parser.add_argument("--id_earth", help="Send manual warning for given earthquake ID",
						nargs=1, type=int)
	parser.add_argument("--clients", help="Client IDs to send manual warning to",
						nargs='+', type=int)
	parser.add_argument("--dry-run", help="Print message rather than actually sending it",
						action="store_true")
	parser.add_argument("--path", help="Folder containing warningcenter database (default: ~/warningcenter)")
	parser.add_argument("--verbose", help="Print debugging information",
						action="store_true")
	args = parser.parse_args()

	## SQlite database containing information related to warning center
	if args.path:
		WARCEN_DB_PATH = args.path
	else:
		WARCEN_DB_PATH = os.path.join(home_folder, 'warningcenter')
	WARCEN_DB_FILE = os.path.join(WARCEN_DB_PATH, 'rob_warning_center.sqlite')

	## Configuration file
	WARCEN_CONFIG_FILE = os.path.join(WARCEN_DB_PATH, 'rob_warning_center_config.py')

	if os.path.exists(WARCEN_DB_PATH):
		WARCEN_DB = simpledb.SQLiteDB(WARCEN_DB_FILE)
	else:
		print("Error: folder %s does not exist!" % WARCEN_DB_PATH)
		WARCEN_DB = None

	if WARCEN_DB:
		warcen = WarningCenter(WARCEN_DB)
	else:
		warcen = None

	if args.init:
		warcen.create_db()
	if warcen and not args.dry_run:
		warcen.read_config(WARCEN_CONFIG_FILE)
		#warcen.populate_db(CRITERIA, CLIENTS, CLIENT_CRITERIA)
	if args.info:
		warcen.print_info()
	elif args.run:
		warcen.poll_catalog(verbose=args.verbose, dry_run=args.dry_run)
	elif args.id_earth:
		if not args.clients:
			print("Use --clients option to specify client IDs to send warning to")
		else:
			warcen.send_manual_warning(args.id_earth[0], args.clients, dry_run=args.dry_run)
	elif args.dry_run:
		warcen.poll_catalog(verbose=args.verbose, dry_run=True)
	if warcen:
		warcen.close_db()
