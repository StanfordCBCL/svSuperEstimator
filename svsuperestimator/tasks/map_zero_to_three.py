from __future__ import annotations

import os
from multiprocessing import Pool
from datetime import datetime

import numpy as np
import pandas as pd
from rich import box
from rich.table import Table
from scipy import optimize
from svzerodsolver import runnercpp
import orjson

from .. import reader, visualizer
from ..reader import CenterlineHandler
from ..reader import utils as readutils
from . import plotutils, taskutils
from .task import Task

from collections import defaultdict
from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk
import re
from scipy.interpolate import interp1d


class MapZeroToThree(Task):
    """Map 0D result to 3D."""

    TASKNAME = "MapZeroToThree"
    DEFAULTS = {
        "num_procs": 1,
        **Task.DEFAULTS,
    }

    def core_run(self):
        """Core routine of the task."""

        self._map_0d_on_centerline()

        pass

    def post_run(self):
        """Postprocessing routine of the task."""

        pass

    def generate_report(self):
        """Generate the task report."""

        pass

    def _map_0d_on_centerline(self):

        cl_handler = self.project["centerline"]
        zerod_handler = self.project["0d_simulation_input"]
        zerod_handler.update_simparams(last_cycle_only=True)

        result0d = runnercpp.run_from_config(zerod_handler.data)

        # assemble output dict
        rec_dd = lambda: defaultdict(rec_dd)
        arrays = rec_dd()

        def collect_arrays(output):
            res = {}
            for i in range(output.GetNumberOfArrays()):
                name = output.GetArrayName(i)
                data = output.GetArray(i)
                res[name] = vtk_to_numpy(data)
            return res

        def convert_csv_to_branch_result(df, zerod_handler):
            # loop branches and segments
            names = list(sorted(set(df["name"])))
            out = {"flow": {}, "pressure": {}, "distance": {}}

            for name in names:
                # extract ids
                br, seg = [int(s) for s in re.findall(r"\d+", name)]

                # add 0d results
                for field in ["flow", "pressure"]:
                    if seg == 0:
                        out[field][br] = [
                            list(df[df.name == name][field + "_in"])
                        ]
                    out[field][br] += [
                        list(df[df.name == name][field + "_out"])
                    ]
                out["time"] = list(df[df.name == name]["time"])

                # add path distance
                for vessel in zerod_handler.data["vessels"]:
                    if vessel["vessel_name"] == name:
                        if seg == 0:
                            out["distance"][br] = [0]
                        l_new = (
                            out["distance"][br][-1] + vessel["vessel_length"]
                        )
                        out["distance"][br] += [l_new]

            # convert to numpy
            for field in ["flow", "pressure", "distance"]:
                for br in out[field].keys():
                    out[field][br] = np.array(out[field][br])
            out["time"] = np.array(out["time"])

            # save to file
            return out

        # extract point arrays from geometries
        arrays_cent = collect_arrays(cl_handler.data.GetPointData())

        # add centerline arrays
        for name, data in arrays_cent.items():
            arrays[name] = data

        # centerline points
        points = vtk_to_numpy(cl_handler.data.GetPoints().GetData())

        # all branch ids in centerline
        ids_cent = np.unique(arrays_cent["BranchId"]).tolist()
        ids_cent.remove(-1)

        results = convert_csv_to_branch_result(result0d, zerod_handler)

        # loop all result fields
        for f in ["flow", "pressure"]:
            if f not in results:
                continue

            # check if ROM branch has same ids as centerline
            ids_rom = list(results[f].keys())
            ids_rom.sort()
            assert (
                ids_cent == ids_rom
            ), "Centerline and ROM results have different branch ids"

            # initialize output arrays
            array_f = np.zeros(
                (arrays_cent["Path"].shape[0], len(results["time"]))
            )
            n_outlet = np.zeros(arrays_cent["Path"].shape[0])

            # loop all branches
            for br in results[f].keys():
                # results of this branch
                res_br = results[f][br]

                # get centerline path
                path_cent = arrays_cent["Path"][arrays_cent["BranchId"] == br]

                # get node locations from 0D results
                path_1d_res = results["distance"][br]
                f_res = res_br

                assert np.isclose(
                    path_1d_res[0], 0.0
                ), "ROM branch path does not start at 0"
                assert np.isclose(
                    path_cent[0], 0.0
                ), "Centerline branch path does not start at 0"
                msg = "ROM results and centerline have different branch path lengths"
                assert np.isclose(path_1d_res[-1], path_cent[-1]), msg

                # interpolate ROM onto centerline
                # limit to interval [0,1] to avoid extrapolation error interp1d due to slightly incompatible lenghts
                f_cent = interp1d(path_1d_res / path_1d_res[-1], f_res.T)(
                    path_cent / path_cent[-1]
                ).T

                # store results of this path
                array_f[arrays_cent["BranchId"] == br] = f_cent

                # add upstream part of branch within junction
                if br == 0:
                    continue

                # first point of branch
                ip = np.where(arrays_cent["BranchId"] == br)[0][0]

                # centerline that passes through branch (first occurence)
                cid = np.where(arrays_cent["CenterlineId"][ip])[0][0]

                # id of upstream junction
                jc = arrays_cent["BifurcationId"][ip - 1]

                # centerline within junction
                is_jc = arrays_cent["BifurcationId"] == jc
                jc_cent = np.where(
                    np.logical_and(is_jc, arrays_cent["CenterlineId"][:, cid])
                )[0]

                # length of centerline within junction
                jc_path = np.append(
                    0,
                    np.cumsum(
                        np.linalg.norm(
                            np.diff(points[jc_cent], axis=0), axis=1
                        )
                    ),
                )
                jc_path /= jc_path[-1]

                # results at upstream branch
                res_br_u = results[f][arrays_cent["BranchId"][jc_cent[0] - 1]]

                # results at beginning and end of centerline within junction
                f0 = res_br_u[-1]
                f1 = res_br[0]

                # map 1d results to centerline using paths
                array_f[jc_cent] += interp1d([0, 1], np.vstack((f0, f1)).T)(
                    jc_path
                ).T

                # count number of outlets of this junction
                n_outlet[jc_cent] += 1

            # normalize results within junctions by number of junction outlets
            is_jc = n_outlet > 0
            array_f[is_jc] = (array_f[is_jc].T / n_outlet[is_jc]).T

            # assemble time steps
            for i, t in enumerate(results["time"]):
                arrays[f + "_" + str(t)] = array_f[:, i]

        # add arrays to centerline and write to file
        for f, a in arrays.items():
            out_array = numpy_to_vtk(a)
            out_array.SetName(f)
            cl_handler.data.GetPointData().AddArray(out_array)

        cl_handler.to_file(
            os.path.join(self.output_folder, "centerline_result.vtp")
        )
