import SimplePasteBin as helper
import time
import sys
from datetime import timedelta
import os.path

APIKEY = "@PASTEBIN_API_KEY@"
USERNAME = "@PASTEBIN_USERNAME@"
PASSWORD = "@PASTEBIN_PASSWORD@"
PRIVATEKEY = ""

pb = helper.SimplePasteBin(
    username=USERNAME,
    password=PASSWORD,
    api_key=APIKEY,
    is_verbose=True)

pname = time.strftime("%Y%m%d-%H%M%S")

def GetUpTime():
    with open('/proc/uptime', 'r') as f:
        uptime_seconds = float(f.readline().split()[0])
        uptime_string = str(timedelta(seconds=uptime_seconds))
    return uptime_string

def GetAgentsBanner():
    str = ''.join([
        "##############################################\n",
        "  Python PasteBin Agent V1.1\n",
        "  Instance Up-time: ", GetUpTime(), '\n',
        "  Date: ", pname, '\n',
        "##############################################\n\n",
    ])
    return str

def ReadLogFile(fname):
    f = open(fname, "r")
    content = f.read()
    f.close()
    final_content = ''.join([GetAgentsBanner(), content])
    return final_content

def TryUpload(fname, pastename="AutoBuild-", mode="HW-"):
    ret_vals = 0
    try:
        ret_vals += pb.login()
        logcontent = ReadLogFile(fname)
        joinedpname = ''.join([pastename, mode, pname])
        ret_vals +=  pb.create_paste(joinedpname, logcontent, 'N', 'private')
    except Exception as e:
        print(e)
        ret_vals += 1
    return ret_vals

def MainFunc():
    print("Running python3 pastebin script...")
    rslt = 0

    fname = "autobuild_hw_log.txt"
    if os.path.isfile(fname):
        print("Found HW log file.")
        rslt += TryUpload(fname, mode="HW-")

    fname = "autobuild_hwemu_log.txt"
    if os.path.isfile(fname):
        print("Found HW-EMU log file.")
        rslt += TryUpload(fname, mode="HWEMU-")

    fname = "autobuild_swemu_log.txt"
    if os.path.isfile(fname):
        print("Found SW-EMU log file.")
        rslt += TryUpload(fname, mode="SWEMU-")

    if rslt == 0:
        print("The log file has been uploaded to PasteBin.com as a private paste.")
        sys.exit(0)
    else:
        print("PasteBin script has failed.")
        sys.exit(5)

MainFunc()
