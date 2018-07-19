#!/usr/bin/env python

from pycbc.waveform import *
from pycbc.pnutils import *
from numpy import *

waveforms = []


for q in arange(2, 3, 1):
  M = 30.
  eta = q / (1. + q)**2
  s1x=0.
  s1y=0.
  s2x=0.
  s2y=0.
  s1z=-0.6
  #s2z= 0.9

  for s2z in arange(.5, .98, .02):
      m1, m2 = mtotal_eta_to_mass1_mass2(M, eta)

      hp, hc = get_td_waveform(approximant='SEOBNRv3', mass1=m1, mass2=m2, spin1x=s1x, spin1y=s1y, spin1z=s1z, spin2x=s2x, spin2y=s2y, spin2z=s2z, f_lower=25., delta_t=1./8192.)
      waveforms.append( [m1, m2, hp, hc] )
      print 'generated waveform for q = %f' % q


      for m1, m2, hp, hc in waveforms:
        idx = hp.sample_times.data >= 0.
        pdata = hp.data[idx][0]
        cdata = hc.data[idx][0]
        savetxt('q_%.2f_s1z_%.2f_s2z_%.2f_f25.dat' % (m1/m2, s1z, s2z) ,\
          zip(hp.data / sqrt(pdata**2 + cdata**2)), fmt='%.18e')
        print 'saved waveform for q = %f' % (m1/m2)



