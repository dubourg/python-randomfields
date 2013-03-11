#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import ctypes


def get_available_memory():
    """Returns the available memory on the system in bytes."""

    if os.name == "nt":
        kernel32 = ctypes.windll.kernel32
        c_ulong = ctypes.c_ulong

        class MEMORYSTATUS(ctypes.Structure):
            _fields_ = [
                ("dwLength", c_ulong),
                ("dwMemoryLoad", c_ulong),
                ("dwTotalPhys", c_ulong),
                ("dwAvailPhys", c_ulong),
                ("dwTotalPageFile", c_ulong),
                ("dwAvailPageFile", c_ulong),
                ("dwTotalVirtual", c_ulong),
                ("dwAvailVirtual", c_ulong)
            ]
        memoryStatus = MEMORYSTATUS()
        memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUS)
        kernel32.GlobalMemoryStatus(ctypes.byref(memoryStatus))
        memory_in_bytes = memoryStatus.dwTotalPhys / 1024 ** 2
    elif os.name == 'posix':
        answer = os.popen("free -b")
        if answer.readlines()[0].split()[2] != 'free':
            raise Exception("Can't find out how much memory is available!")
        memory_in_bytes = os.popen("free -b").readlines()[1].split()[3]
    else:
        raise Exception("Can't find out how much memory is available!")

    return int(memory_in_bytes)
