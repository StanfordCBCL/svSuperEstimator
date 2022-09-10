from __future__ import annotations

import os
from multiprocessing import Pool

import numpy as np
from svzerodsolver import runnercpp
import vtk

from .. import reader
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
        self._map_centerline_on_3d()

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
                            list(df[df.name == name][field + "_in"])[0:1]
                        ]
                    out[field][br] += [
                        list(df[df.name == name][field + "_out"])[0:1]
                    ]
                out["time"] = list(df[df.name == name]["time"])[0:1]

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

        # # add centerline arrays
        # for name, data in arrays_cent.items():
        #     arrays[name] = data

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
            arrays[f] = array_f[:, 0]

        # add arrays to centerline and write to file
        for f, a in arrays.items():
            out_array = numpy_to_vtk(a)
            out_array.SetName(f)
            cl_handler.data.GetPointData().AddArray(out_array)

        cl_handler.to_file(
            os.path.join(self.output_folder, "initial_centerline.vtp")
        )

    def _map_centerline_on_3d(self):

        cl_handler = reader.CenterlineHandler.from_file(
            os.path.join(self.output_folder, "initial_centerline.vtp")
        )
        vol_handler: reader.MeshHandler = self.project["3d_simulation_volume"]
        surf_handler = self.project["3d_simulation_surface"]

        def get_centerline_3d_map(cl_handler, vol_handler):
            """
            Create a map from centerine to volume mesh through region growing
            """
            # get points
            points_vol = vtk_to_numpy(vol_handler.data.GetPoints().GetData())
            points_1d = vtk_to_numpy(cl_handler.data.GetPoints().GetData())

            # get volume points closest to centerline
            cp_vol = ClosestPoints(vol_handler.data)
            seed_points = np.unique(cp_vol.search(points_1d))

            # map centerline points to selected volume points
            cp_1d = ClosestPoints(cl_handler.data)
            seed_ids = np.array(cp_1d.search(points_vol[seed_points]))

            # call region growing algorithm
            ids, rad = region_grow(
                vol_handler.data, seed_points, seed_ids, n_max=999
            )

            # check 1d to 3d map
            assert (
                np.max(ids) <= cl_handler.data.GetNumberOfPoints() - 1
            ), "1d-3d map non-conforming"

            return ids, rad

        class ClosestPoints:
            """
            Find closest points within a geometry
            """

            def __init__(self, inp):
                dataset = vtk.vtkPolyData()
                dataset.SetPoints(inp.GetPoints())

                locator = vtk.vtkPointLocator()
                locator.Initialize()
                locator.SetDataSet(dataset)
                locator.BuildLocator()

                self.locator = locator

            def search(self, points, radius=None):
                """
                Get ids of points in geometry closest to input points
                Args:
                    points: list of points to be searched
                    radius: optional, search radius
                Returns:
                    Id list
                """
                ids = []
                for p in points:
                    if radius is not None:
                        result = vtk.vtkIdList()
                        self.locator.FindPointsWithinRadius(radius, p, result)
                        ids += [
                            result.GetId(k)
                            for k in range(result.GetNumberOfIds())
                        ]
                    else:
                        ids += [self.locator.FindClosestPoint(p)]
                return ids

        def region_grow(geo, seed_points, seed_ids, n_max=99):
            # initialize output arrays
            array_ids = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
            array_rad = np.zeros(geo.GetNumberOfPoints())
            array_dist = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
            array_ids[seed_points] = seed_ids

            # initialize ids
            cids_all = set()
            pids_all = set(seed_points.tolist())
            pids_new = set(seed_points.tolist())

            # get points
            pts = vtk_to_numpy(geo.GetPoints().GetData())

            # loop until region stops growing or reaches maximum number of iterations
            i = 0
            while len(pids_new) > 0 and i < n_max:
                # update
                pids_old = pids_new

                # print progress
                print_str = "Iteration " + str(i)
                print_str += "\tNew points " + str(len(pids_old)) + "     "
                print_str += "\tTotal points " + str(len(pids_all))
                print(print_str)

                # grow region one step
                pids_new = grow(geo, array_ids, pids_old, pids_all, cids_all)

                # convert to array
                pids_old_arr = list(pids_old)

                # create point locator with old wave front
                points = vtk.vtkPoints()
                points.Initialize()
                for i_old in pids_old:
                    points.InsertNextPoint(geo.GetPoint(i_old))

                dataset = vtk.vtkPolyData()
                dataset.SetPoints(points)

                locator = vtk.vtkPointLocator()
                locator.Initialize()
                locator.SetDataSet(dataset)
                locator.BuildLocator()

                # find closest point in new wave front
                for i_new in pids_new:
                    i_old = pids_old_arr[
                        locator.FindClosestPoint(geo.GetPoint(i_new))
                    ]
                    array_ids[i_new] = array_ids[i_old]
                    array_rad[i_new] = array_rad[i_old] + np.linalg.norm(
                        pts[i_new] - pts[i_old]
                    )
                    array_dist[i_new] = i

                # count grow iterations
                i += 1

            return array_ids, array_rad

        def grow(geo, array, pids_in, pids_all, cids_all):
            # ids of propagating wave-front
            pids_out = set()

            # loop all points in wave-front
            for pi_old in pids_in:
                cids = vtk.vtkIdList()
                geo.GetPointCells(pi_old, cids)

                # get all connected cells in wave-front
                for j in range(cids.GetNumberOfIds()):
                    # get cell id
                    ci = cids.GetId(j)

                    # skip cells that are already in region
                    if ci in cids_all:
                        continue
                    else:
                        cids_all.add(ci)

                    pids = vtk.vtkIdList()
                    geo.GetCellPoints(ci, pids)

                    # loop all points in cell
                    for k in range(pids.GetNumberOfIds()):
                        # get point id
                        pi_new = pids.GetId(k)

                        # add point only if it's new and doesn't fullfill stopping criterion
                        if array[pi_new] == -1 and pi_new not in pids_in:
                            pids_out.add(pi_new)
                            pids_all.add(pi_new)

            return pids_out

        # get 1d -> 3d map
        map_ids, map_rad = get_centerline_3d_map(cl_handler, vol_handler)

        vol_handler.set_point_data_array(
            "pressure", cl_handler.get_point_data_array("pressure")[map_ids]
        )

        # inverse map
        map_ids_inv = {}
        for i in np.unique(map_ids):
            map_ids_inv[i] = np.where(map_ids == i)

        # create radial coordinate [0, 1]
        rad = np.zeros(vol_handler.data.GetNumberOfPoints())
        for i, ids in map_ids_inv.items():
            rad_max = np.max(map_rad[ids])
            if rad_max == 0:
                rad_max = np.max(map_rad)
            rad[ids] = map_rad[ids] / rad_max

        # set points at wall to hard 1
        wall_ids = (
            surf_handler.get_point_data_array("GlobalNodeID").astype(int) - 1
        )
        rad[wall_ids] = 1

        # mean velocity
        u_mean = cl_handler.get_point_data_array(
            "flow"
        ) / cl_handler.get_point_data_array("CenterlineSectionArea")

        # parabolic velocity
        u_quad = 2 * u_mean[map_ids] * (1 - rad**2)
        for i, ids in map_ids_inv.items():
            u_mean_is = np.mean(u_quad[map_ids_inv[i]])
            u_quad[ids] *= u_mean[i] / u_mean_is

        # parabolic velocity vector field
        velocity = (
            np.outer(u_quad, np.ones(3))
            * cl_handler.get_point_data_array("CenterlineSectionNormal")[
                map_ids
            ]
        )

        # add to volume mesh
        vol_handler.set_point_data_array("velocity", velocity)

        # write to file
        vol_handler.to_file(os.path.join(self.output_folder, "initial.vtu"))
