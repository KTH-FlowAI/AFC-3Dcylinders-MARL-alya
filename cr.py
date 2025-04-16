#!/bin/env python
#
# SATELLITE TOOLS
#
# Chrono module for performance profiling.
#
# Arnau Miro, Elena Terzic
from __future__ import print_function, division

import numpy as np, time as time_module


CHANNEL_DICT = {}


class channel(object):
	'''
	This is a channel for the cr counter
	'''
	def __init__(self, name, tmax, tmin, tsum, nop, tini):
		self._name = name # Name of the channel
		self._tmax = tmax # Maximum time of the channel
		self._tmin = tmin # Minimum time of the channel
		self._tsum = tsum # Total time of the channel
		self._nop  = nop  # Number of operations
		self._tini = tini # Initial instant (if == 0 channel is not being take into account)

	def __str__(self):
		return 'name %-25s n %4d tmin %e tmax %e tavg %e tsum %e' % (self.name,self.nop,self.tmin,self.tmax,self.tavg,self.tsum)

	def __add__(self, other):
		new = copy.deepcopy(self)
		new._tmax  = max(new._tmax,other._tmax)
		new._tmin  = min(new._tmin,other._tmin)
		new._tsum += other._tsum
		new._nop  += other._nop 
		return new

	def __iadd__(self, other):
		self._tmax  = max(self._tmax,other._tmax)
		self._tmin  = min(self._tmin,other._tmin)
		self._tsum += other._tsum
		self._nop  += other._nop 
		return self

	def reset(self):
		'''
		Reset the channel
		'''
		self._tmax = 0.0
		self._tmin = 0.0
		self._tsum = 0.0
		self._nop  = 0.0
		self._tini = 0.0

	def restart(self):
		self._tini = 0.0

	def start(self,tini):
		self._tini = tini

	def increase_nop(self):
		self._nop += 1

	def increase_time(self,time):
		self._tsum += time

	def set_max(self,time):
		if time > self._tmax or self._nop == 1: self._tmax = time

	def set_min(self,time):
		if time < self._tmin or self._nop == 1: self._tmin = time

	def elapsed(self,time):
		return time - self._tini

	def is_running(self):
		return not self._tini == 0

	@classmethod
	def new(cls,name):
		'''
		Create a new channel
		'''
		return cls(name,0,0,0,0,0)

	@property
	def name(self):
		return self._name
	@property
	def nop(self):
		return self._nop
	@property
	def tmin(self):
		return self._tmin
	@property
	def tmax(self):
		return self._tmax
	@property
	def tavg(self):
		return self._tsum/(1.* self._nop) if self._nop > 0  else 0.
	@property
	def tsum(self):
		return self._tsum
	@property
	def report(self):
		return np.array([self.nop,self.tmin,self.tmax,self.tavg,self.tsum])

def _newch(ch_name):
	'''
	Add a new channel to the list
	'''
	CHANNEL_DICT[ch_name] = channel.new(ch_name)
	return CHANNEL_DICT[ch_name]

def _findch(ch_name):
	'''
	Look for the channel
	'''
	return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else None

def _addsuff(ch_name,suff=-1):
	return ch_name if suff <= 0 else '%s%02d' % (ch_name,suff)

def _findch_crash(ch_name):
	'''
	Look for the channel and crash if it does not exist
	'''
	if not ch_name in CHANNEL_DICT.keys():
		raise ValueError('Channel %s does not exist!' % ch_name)
	return CHANNEL_DICT[ch_name]

def _findch_create(ch_name):
	'''
	Find the channel and if not found create it
	'''
	return CHANNEL_DICT[ch_name] if ch_name in CHANNEL_DICT.keys() else _newch(ch_name)

def _gettime():
	'''
	Returns the number of second since an arbitrary instant but fixed.
	Returned value will always be > 0.
	'''
	return time_module.time()

def _info_serial():
	tsum_array = np.array([CHANNEL_DICT[key].tsum for key in CHANNEL_DICT.keys()])
	name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

	ind = np.argsort(tsum_array) # sorted indices

	print('\ncr_info:')
	for ii in ind[::-1]:
		print(CHANNEL_DICT[name_array[ii]])
	print('')

def _report_serial(fname):
	tsum_array = np.array([CHANNEL_DICT[key].tsum for key in CHANNEL_DICT.keys()])
	name_array = np.array([CHANNEL_DICT[key].name for key in CHANNEL_DICT.keys()])

	ind = np.argsort(tsum_array) # sorted indices

	file = open(fname,'w')
	# Header
	file.write('# name, n, tmin, tmax, tavg, tsum\n')
	
	for ii in ind[::-1]:
		r = CHANNEL_DICT[name_array[ii]].report
		file.write('%-25s, %4d, %e, %e, %e, %e\n'%(name_array[ii],r[0],r[1],r[2],r[3],r[4]))


def cr_reset():
	'''
	Delete all channels and start again
	'''
	CHANNEL_DICT = {}

def cr_info(rank=-1):
	'''
	Print information - order by major sum
	'''
	_info_serial()

def cr_report(filename):
	'''
	Print a report of the execution times in a file
	'''
	_report_serial(filename)

def cr_start(ch_name,suff):
	'''
	Start the chrono of a channel
	'''
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_create(name_tmp)
	if channel.is_running():
		raise ValueError('Channel %s was already set!'%channel.name)
	channel.start( _gettime() )

def cr_stop(ch_name,suff):
	'''
	Stop the chrono of a channel
	'''
	end      = _gettime()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	time     = channel.elapsed(end)

	channel.increase_nop()
	channel.set_max(time)
	channel.set_min(time)
	channel.increase_time(time)

	channel.restart()

def cr_time(ch_name,suff):
	'''
	Get the time of a channel that is running; channel keeps running
	'''
	end = _gettime()
	name_tmp = _addsuff(ch_name,suff)
	channel  = _findch_crash(name_tmp)
	return channel.elapsed(end)
