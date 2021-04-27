import os
import sys
import time
import random
random.seed(15213)

from multiprocessing import Process
from selenium.common.exceptions import ElementClickInterceptedException
from TrafficController import TrafficController
from Barrier import Barrier
from PufferPlayer import PufferPlayer
from YouTubePlayer import YouTubePlayer

NUM_BROWSERS = 1
MAX_TRACE_LEN = 320 #seconds

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

def threadOrchestrator(players, abr, trace_dir, trace_file, datapointsInterval, numRuns, output_path, dev_interface):
    for run in range(numRuns):
        print("=======RUN {}-{}=======".format(NUM_BROWSERS, run))
        killWebdrivers()

        tc = TrafficController(trace_dir + trace_file, dev_interface)

        processes = []

        barrier = Barrier(NUM_BROWSERS+1)

        for player_num in range(1, NUM_BROWSERS+1):
            processes.append(Process(
                target=threadInstance,
                args=(
                    players[player_num-1]["player"],
                    player_num,
                    tc,
                    run,
                    NUM_BROWSERS,
                    barrier,
                    players[player_num-1]["link"],
                    abr,
                    trace_file,
                    datapointsInterval,
                    output_path
                )
            ))

        for p in processes:
            p.start()

        tc_process = Process(target=incrementTC, args=(tc, barrier))
        tc_process.start()

        tc_process.join()
        for p in processes:
            p.join()

        time.sleep(1)
        for p in processes:
            p.terminate()
        killWebdrivers()

        time.sleep(10)

def threadInstance(player, playerNum, tc, run, NUM_BROWSERS, barrier, link, abr, traceFile, datapointsInterval, output_path):
    output_file = os.path.join(output_path, "{}-{}-{}_{}".format(abr, NUM_BROWSERS, run, playerNum))
    browser = player(link, output_file, playerNum, True)

    while True:
        try:
            browser.initVideo()
            break
        except ElementClickInterceptedException:
            print("Retrying initVideo")
            browser.f.close()
            browser.driver.close()
            browser.driver.quit()
            browser = player(link, output_file, playerNum, True)
            time.sleep(2)

    # Barrier to wait for all browsers to be ready
    barrier.wait()

    browser.start()
    numDatapointsCollected = 0
    start = time.time()
    t = start
    while True:
        res = browser.collectData()
        numDatapointsCollected += 1

        # Exit condition
        if time.time() - t > 1:
            nextBw = tc.getNextBW(time.time() - start + tc.times[0])
            if nextBw is None: # and (time.time() - start) >= MAX_TRACE_LEN
                break
            print("[{}] - {} - {}".format(playerNum, tc.bIdx, res))
            t = time.time()

        time.sleep(datapointsInterval)

    browser.f.close()
    browser.driver.close()
    browser.driver.quit()

    if abr != "youtube":
        browser.display.stop()

    return True

def mainHomogeneous(data_dir, trace_dir, trace_name, abr, links, datapoints_interval, num_runs, dev_interface):
    # Make result directories
    if not os.path.isdir(data_dir + abr):
        os.mkdir(data_dir + abr)

    if not os.path.isdir(os.path.join(data_dir, abr, trace_name)):
        os.mkdir(os.path.join(data_dir, abr, trace_name))
    else:
        num_files = len(os.listdir(os.path.join(data_dir, abr, trace_name)))
        print("{} already exists with {} files.".format(os.path.join(data_dir, abr, trace_name), num_files))
        return

    output_path = os.path.join(data_dir, abr, trace_name)

    # Setup player objects
    if abr == "youtube":
        player = YouTubePlayer
    else:
        player = PufferPlayer

    players = []
    for _ in range(NUM_BROWSERS):
        players.append({"player": player, "link": links[0]})

    threadOrchestrator(players, abr, trace_dir, trace_name, datapoints_interval, num_runs, output_path, dev_interface)


def mainHeterogeneous(data_dir, trace_dir, trace_name, composition, links, datapoints_interval, num_runs, dev_interface):
    # Make result directories
    if not os.path.isdir(data_dir + composition):
        os.mkdir(data_dir + composition)

    if not os.path.isdir(os.path.join(data_dir, composition, trace_name)):
        os.mkdir(os.path.join(data_dir, composition, trace_name))
    else:
        num_files = len(os.listdir(os.path.join(data_dir, composition, trace_name)))
        print("{} already exists with {} files.".format(os.path.join(data_dir, composition, trace_name), num_files))
        return

    output_path = os.path.join(data_dir, composition, trace_name)

    # Setup player objects according to composition
    # Hardcoding to MPC and pensieve now
    players = []
    for i in range(NUM_BROWSERS):
        players.append({
            "player": PufferPlayer,
            "link": links[i],
        })

    threadOrchestrator(players, composition, trace_dir, trace_name, datapoints_interval, num_runs, output_path, dev_interface)

def get_trace_list(trace_file):
    with open(trace_file) as f:
        traces = [t.strip() for t in f.readlines()]
    return traces

if __name__ == "__main__":
    # traces = random.sample(os.listdir("Traces/"), 100)
    traces = get_trace_list("pensieve_traces.txt")
    # link = "http://3.239.110.219:8080/player/"
    # link = "https://www.youtube.com/watch?v=QZUeW8cQQbo"
    #links = [
    #        "http://3.238.155.139:8080/player/",
    #        "http://3.239.5.170:8080/player/",
    #        "http://35.175.174.228:8080/player/",
    #        "http://18.205.66.65:8080/player/",
    #]
    #abr = "mpc-1-robust_mpc-1-bola-1-bba-1"
    links = ["http://34.231.242.33:8080/player/"]
    abr = "bola-only"
    datapoints_interval = 1
    data_dir = "Data/"
    trace_dir = "Traces/"
    num_runs = 1
    dev_interface = "ens5"

    input("Running {}, press ENTER to continue...".format(abr))

    for idx, trace_name in enumerate(traces):
        print("~~~~~~~~~`[{}] - {}~~~~~~~~~~~~~~~".format(idx, trace_name))
        mainHeterogeneous(data_dir, trace_dir, trace_name, abr, links, datapoints_interval, num_runs, dev_interface)
