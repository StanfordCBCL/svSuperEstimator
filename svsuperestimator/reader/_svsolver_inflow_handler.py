class SvSolverInflowHandler:
    def __init__(self, filename):
        with open(filename) as ff:
            self.data = ff.read()

    def get_boundary_condition(self):
        time, flow = [], []
        for line in self.data.splitlines():
            t, q = line.split()
            time.append(float(t))
            flow.append(-float(q))

        return {"t": time, "Q": flow}
