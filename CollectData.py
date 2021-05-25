from TrafficController import TrafficController
from YouTubePlayer import YouTubePlayer
from PufferPlayer import PufferPlayer
from MITMProxy.MITMProxy import MITMProxy
from Barrier import Barrier
from selenium.common.exceptions import ElementClickInterceptedException
from multiprocessing import Process

import random
random.seed(15213)
import pandas as pd
import glob, sys, os, time

def runTrace(data_dir, trace_dir, trace_name, provider_name, link, datapoints_interval, num_runs, dev_interface):
    # Make result directories
    if not os.path.isdir(data_dir + provider_name):
        os.mkdir(data_dir + provider_name)

    if not os.path.isdir(os.path.join(data_dir, provider_name, trace_name)):
        os.mkdir(os.path.join(data_dir, provider_name, trace_name))
    else:
        num_files = len(os.listdir(os.path.join(data_dir, provider_name, trace_name)))
        print("{} already exists with {} files.".format(os.path.join(data_dir, provider_name, trace_name), num_files))
        return

    output_path = os.path.join(data_dir, provider_name, trace_name)

    # Setup player object according to provider_name
    if "youtube" in link:
        player = {
            "player": YouTubePlayer,
            "link": link,
        }
    else:
        player = {
            "player": PufferPlayer,
            "link": link,
        }

    threadOrchestrator(player, provider_name, trace_dir, trace_name, datapoints_interval, num_runs, output_path, dev_interface)

def threadOrchestrator(player, abr, trace_dir, trace_file, datapointsInterval, numRuns, output_path, dev_interface):
    for run in range(numRuns):
        print("=======RUN {}=======".format(run))
        killWebdrivers()

        tc = TrafficController(trace_dir + trace_file, dev_interface)
        
        proxy = MITMProxy(9001)
        proxy.startProxy("{}/proxy".format(output_path))

        barrier = Barrier(2)

        videoProcess = Process(
            target=threadInstance,
            args=(
                player["player"],
                1,
                tc,
                run,
                barrier,
                player["link"],
                abr,
                trace_file,
                datapointsInterval,
                output_path,
                proxy
            )
        )

        videoProcess.start()

        tc_process = Process(target=incrementTC, args=(tc, barrier))
        tc_process.start()

        tc_process.join()
        videoProcess.join()

        time.sleep(1)
        videoProcess.terminate()
        killWebdrivers()
        
        proxy.stopProxy()

        time.sleep(10)

def threadInstance(player, playerNum, tc, run, barrier, link, abr, traceFile, datapointsInterval, output_path, proxy):
    output_file = os.path.join(output_path, "player_data-{}-{}".format(abr, run))
    browser = player(link, output_file, playerNum, headless=False, proxy=proxy)

    while True:
        try:
            browser.initVideo()
            break
        except ElementClickInterceptedException:
            print("Retrying initVideo")
            browser.f.close()
            browser.driver.close()
            browser.driver.quit()
            browser = player(link, output_file, playerNum, headless=False, proxy=proxy)
            time.sleep(2)

    # Barrier to wait for all browsers to be ready
    barrier.wait()

    browser.start()
    numDatapointsCollected = 0
    start = time.time()
    t = start
    nextBw = tc.bandwidths[0]
    while True:
        res, _ = browser.collectData(nextBw)
        numDatapointsCollected += 1

        # Exit condition
        if time.time() - t > 1:
            nextBw = tc.getNextBW(time.time() - start + tc.times[0])
            if nextBw is None: # and (time.time() - start) >= MAX_TRACE_LEN
                break
            print("[{}] - {}({}) - {}".format(playerNum, tc.bIdx, nextBw, res))
            t = time.time()

        time.sleep(datapointsInterval)

    browser.f.close()
    browser.driver.close()
    browser.driver.quit()

    browser.stopDisplay()

    return True

def get_trace_list(trace_file):
    with open(trace_file) as f:
        traces = [t.strip() for t in f.readlines()]
    return traces

def killWebdrivers():
    os.system('killall %s' % ("chromedriver"))

def incrementTC(tc, barrier):
    barrier.wait()
    start = time.time()
    t = start
    while True:
        if time.time() - t > 1:
            nextBw = tc.getNextBW(time.time() - start + tc.times[0])
            print(tc.bIdx, time.time() - start + tc.times[0], nextBw)
            if nextBw is None:
                break
            tc.throttleTC(nextBw)
            t = time.time()

        time.sleep(0.5)

def countNumLines(traceDir):
    c = 0
    for _, traceFile in enumerate(glob.glob("{}/*".format(traceDir))):
        with open(traceFile) as tf:
            lines = tf.readlines()
            print(len(lines), traceFile)
            c += len(lines)
    print(c)

if __name__ == "__main__":
    # traces = random.sample(os.listdir("Traces/"), 100)
    traces = get_trace_list("pensieve_traces.txt")
    link = "http://3.236.180.116:8080/player/"
    abr = "mpc"
    datapoints_interval = 1
    data_dir = "Data/ReverseEngineering/"
    trace_dir = "Traces/"
    num_runs = 1
    dev_interface = "wlo1"

    input("Running {}, press ENTER to continue...".format(abr))

    for idx, traceName in enumerate(traces):
        print("~~~~~~~~~`[{}] - {}~~~~~~~~~~~~~~~".format(idx, traceName))
        runTrace(data_dir, trace_dir, traceName, abr, link, datapoints_interval, num_runs, dev_interface)
        break