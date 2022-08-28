class SvSolverRcrHandler:
    def __init__(self, filename):
        with open(filename) as ff:
            self.data = ff.read()

    def get_boundary_conditions(self):
        bc_data = []
        i = 0
        ele_data = {}
        for j, line in enumerate(self.data.splitlines()):
            if j == 0:
                continue
            if i == 0:
                num_data = int(line)
            elif i == 1:
                ele_data["Rp"] = float(line)
            elif i == 2:
                ele_data["C"] = float(line)
            elif i == 3:
                ele_data["Rd"] = float(line)
            elif i == 4:
                t, pd = line.split()
                ele_data["t"] = [float(t)]
                ele_data["Pd"] = [float(pd)]
            elif 4 < i < 4 + num_data:
                t, pd = line.split()
                ele_data["t"].append(float(t))
                ele_data["Pd"].append(float(pd))
                if i == 3 + num_data:
                    bc_data.append(ele_data)
                    ele_data = {}
                    i = 0
                    continue
            i += 1
        return bc_data
