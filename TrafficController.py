import time, os

class TrafficController:
    def __init__(self, traceFile, dev_interface):
        with open(traceFile) as trace:
            lines = trace.readlines()
        self.bandwidths = [float(l.split(" ")[1]) for l in lines]
        self.times = [float(l.split(" ")[0]) for l in lines]

        print(self.times[:10])

        self.bIdx = 0
        self.t = time.time()

        # TC Configs
        self.password = ""
        self.tc_path = "/sbin/tc"
        self.dev_interface = dev_interface
        self.base_throttle_mbit = 16

        self.cleanTC()
        self.initTC()


    def getNextBW(self, currentTime):
        if currentTime > self.times[self.bIdx]:
            self.t = currentTime
            self.bIdx += 1

        if self.bIdx == len(self.bandwidths):
            return None

        return self.bandwidths[self.bIdx]

    def getCurrentBW(self):
        return self.bandwidths[self.bIdx]

    def runSudoCommands(self, cmds):
        for cmd in cmds:
            os.system('echo %s | sudo -S %s' % (self.password, cmd))

    def throttleTC(self, bandwidth_mbit):
        throttle_cmd = 'sudo {tc_path} qdisc change dev ifb0 root tbf rate {bandwidth_mbit}mbit latency 50ms burst 1540'
        cmd = throttle_cmd.format(tc_path=self.tc_path, bandwidth_mbit=bandwidth_mbit)
        
        self.runSudoCommands([cmd])

    def cleanTC(self):
        cmds= ['sudo %s qdisc del dev %s ingress' % (self.tc_path, str(self.dev_interface)),
                'sudo %s qdisc del dev ifb0 root' % self.tc_path]
        self.runSudoCommands(cmds)

    def initTC(self):
        init_throttle = [
            'sudo modprobe ifb numifbs=1',
            # --------- Add relay
            'sudo tc qdisc add dev {dev_interface} handle ffff: ingress'.format(dev_interface=self.dev_interface),
            # -----------  enable the ifb interfaces:
            'sudo ifconfig ifb0 up',
            # -------- And redirect ingress traffic from the physical interfaces to corresponding ifb interface. For wlp4s0 -> ifb0:
            'sudo {tc_path} filter add dev {dev_interface} parent ffff: protocol all u32 match u32 0 0 action mirred egress redirect dev ifb0'.format(
                tc_path=self.tc_path,
                dev_interface=self.dev_interface),
            # -------------- Limit Speed
            'sudo {tc_path} qdisc add dev ifb0 root tbf rate {base_speed_mbit}mbit latency 50ms burst 1540'.format(
                tc_path=self.tc_path, base_speed_mbit=self.base_throttle_mbit)
        ]
        self.runSudoCommands(init_throttle)

if __name__ == "__main__":
    tc = TrafficController("Traces/report.2010-09-21_0742CEST.log_Norway_0", "ens5")
    t = time.time()

    start = time.time()
    while True:
        # Change BW every second
        if time.time() - t > 1:
            nextBw = tc.getNextBW(time.time() - start)
            print(tc.bIdx, nextBw)
            if nextBw is None:
                break
            t = time.time()

        time.sleep(0.2)