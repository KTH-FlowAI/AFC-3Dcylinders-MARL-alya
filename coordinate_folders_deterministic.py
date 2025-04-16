#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 14:33:54 2022

@author: francisco
"""

import subprocess

def adjust_folders(num_servers):
    for ii in range(num_servers):
        
        subprocess.call("cd alya_files && rm -r environment{0}/EP* && cp -r deterministic/* ./environment{0} " \
                        "> baseline/logs/log_restore_last_episode.log 2>&1".format(ii+1), shell = True)