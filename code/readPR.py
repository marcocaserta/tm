import os
from os import path, listdir
import html2text 
from subprocess import call
import fnmatch

pressrelease_folders = ["NEWS_v4/"]
pressrelease_folders_txt = ["NEWS_TXT_v4/"]
prefix = path.expanduser("~/gdrive/research/nlp/data/")


i = -1
for folder in pressrelease_folders:
    i += 1
    print("Copying from ", folder ," to ", pressrelease_folders_txt[i])
    fullpath = path.join(prefix, folder)
    fullpath_txt = path.join(prefix, pressrelease_folders_txt[i])
    totFilesInFolder = len(fnmatch.filter(os.listdir(fullpath),
    '*.txt'))
    countFiles = 0
    for f in listdir(path.join(prefix, folder)):
        countFiles += 1
        fullname = fullpath + f
        text = open(fullname).readlines()


        newname = fullpath_txt + f 
        fullcommand = "inscript.py " + fullname + " -o " + newname
        os.system(fullcommand)
        print("[{0:8d}/{1:8d} ] = {2:s}".format(countFiles, totFilesInFolder,
        fullcommand))

        if countFiles > 999:
            break


