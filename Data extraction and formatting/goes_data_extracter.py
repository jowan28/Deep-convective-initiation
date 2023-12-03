import cv2 as cv
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4
from scipy.ndimage import convolve
from tobac_flow.flow import Flow
from tobac_flow.detection import detect_growth_markers
from datetime import datetime, timedelta
from tobac_flow import abi
import glob, os
from global_land_mask import globe
import sys


def flatten(l):
    return [item for sublist in l for item in sublist]


def manual_open(
    files,
    ds_slice={"x": slice(700, 2100), "y": slice(700, 1200)},
    variables=[
        "CMI_C04",
        "CMI_C05",
        "CMI_C06",
        "CMI_C07",
        "CMI_C08",
        "CMI_C09",
        "CMI_C10",
        "CMI_C11",
        "CMI_C12",
        "CMI_C13",
        "CMI_C14",
        "CMI_C15",
        "CMI_C16",
        "goes_imager_projection",
    ],
):
    for k in range(len(files)):
        file_name = files[k]
        nc4_ds = netCDF4.Dataset(file_name)
        store = xr.backends.NetCDF4DataStore(nc4_ds)
        if k == 0:
            DS = xr.open_dataset(store)
            DS = DS[variables].isel(ds_slice)
        else:
            DS2 = xr.open_dataset(store)
            DS2 = DS2[variables].isel(ds_slice)
            DS = xr.combine_nested(
                [DS, DS2], concat_dim=["t"], combine_attrs="override"
            )
            store.close()
    return DS


def main_function(year, day0, hour0, hourf):

    # get list of filenames
    file_root = "/gws/nopw/j04/eo_shared_data_vol1/satellite/goes/goes16/ABI-L2-MCMIPC/"
    os.chdir(file_root + str(year))
    formated_days = glob.glob("*")
    file_name_struc = []
    # get files for selected days
    os.chdir(file_root + str(year) + "/" + formated_days[day0 - 1])
    formated_hours = glob.glob("*")
    for j in formated_hours[hour0 - 1 : hourf - 1]:
        os.chdir(file_root + str(year) + "/" + formated_days[day0 - 1] + "/" + j)
        files = [
            file_root + str(year) + "/" + formated_days[day0 - 1] + "/" + j + "/" + file
            for file in glob.glob("*")
        ]
        file_name_struc.append(files)

    # format the filenames
    file_name_struc = flatten(file_name_struc)

    print("The number of files:")
    print(len(file_name_struc))

    # open the data
    goes_ds = manual_open(file_name_struc)
    lat, lon = abi.get_abi_lat_lon(goes_ds)

    # Extract fields and load into memory
    wvd = goes_ds.CMI_C08 - goes_ds.CMI_C10
    try:
        wvd = wvd.compute()
    except AttributeError:
        pass

    bt = goes_ds.CMI_C13
    try:
        bt = bt.compute()
    except AttributeError:
        pass

    swd = goes_ds.CMI_C13 - goes_ds.CMI_C15
    try:
        swd = swd.compute()
    except AttributeError:
        pass

    # Calculate flow
    flow_kwargs = {
        "pyr_scale": 0.5,
        "levels": 6,
        "winsize": 16,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.1,
        "flags": cv.OPTFLOW_FARNEBACK_GAUSSIAN,
    }

    flow = Flow(bt, flow_kwargs=flow_kwargs, smoothing_passes=3)

    wvd_growth, growth_markers = detect_growth_markers(flow, wvd)

    # convolution parameters
    # this is the radius from the mask in each dimension
    dt = 4
    dx = 5
    dy = 5
    conv_filter = np.ones((2 * dt + 1, 2 * dy + 1, 2 * dx + 1))
    not_dcc = convolve(growth_markers, conv_filter, mode="constant", cval=0.0)
    # radius for data saving
    search_radius = 4
    # distance in the time dimension between saves (this is greater than the saved time)
    space_between = 5
    # Find DCCs
    number_detected = np.max(growth_markers.values)
    print("number detected")
    print(number_detected)
    number_in_first_frame = np.max(growth_markers.isel(t=0).values)
    print("number in first frame")
    print(number_in_first_frame)
    # number_initiated = number_detected - number_in_first_frame
    new_DCC = list(range(number_in_first_frame + 1, number_detected + 1))
    print("This is new_DCC:")
    print(new_DCC)
    print("before loop")
    # Loop through DCCs to save to file
    for i in new_DCC:
        first_appearence = np.min(np.where(growth_markers == i))
        first_coords = [
            int(np.round(np.mean(np.where(growth_markers == i), 1)[1:])[0]),
            int(np.round(np.mean(np.where(growth_markers == i), 1)[1:])[1]),
        ]
        init_lat = lat[first_coords[0], first_coords[1]]
        init_lon = lon[first_coords[0], first_coords[1]]

        mask_column = not_dcc[
            :,
            first_coords[0] - search_radius : first_coords[0] + search_radius + 1,
            first_coords[1] - search_radius : first_coords[1] + search_radius + 1,
        ]
        column_sample = np.sum(np.sum(mask_column, 1), 1) == 0
        indexs = np.where(column_sample)[0]
        # this is not optimal
        num_samples = int(np.floor(len(indexs) / space_between))
        sample_locs = indexs[slice(0, space_between * num_samples + 1, space_between)]
        # save the negative examples
        for j in sample_locs:
            # These are the parameters that affect what is saved
            search_range = 4
            d = 4
            clipped = goes_ds.isel(
                {
                    "t": slice(j - search_range, j),
                    "y": slice(first_coords[0] - d, first_coords[0] + d + 1),
                    "x": slice(first_coords[1] - d, first_coords[1] + d + 1),
                }
            )
            if (
                (clipped.x.size == d * 2 + 1)
                and (clipped.y.size == d * 2 + 1)
                and (clipped.t.size == search_range)
            ):
                if globe.is_land(init_lat, init_lon):
                    # this is on land
                    file_name = (
                        "/gws/nopw/j04/aopp/jowanf/land/negdata/DCC_"
                        + str(year)
                        + "_"
                        + str(pd.DatetimeIndex(clipped.t).dayofyear[0])
                        + "_"
                        + str(hour0)
                        + "_"
                        + str(hourf)
                        + "_"
                        + str(i)
                        + "_"
                        + str(j)
                    )
                    clipped.to_netcdf(file_name)
                else:
                    # this is over the sea
                    file_name = (
                        "/gws/nopw/j04/aopp/jowanf/sea/negdata/DCC_"
                        + str(year)
                        + "_"
                        + str(pd.DatetimeIndex(clipped.t).dayofyear[0])
                        + "_"
                        + str(hour0)
                        + "_"
                        + str(hourf)
                        + "_"
                        + str(i)
                        + "_"
                        + str(j)
                    )
                    clipped.to_netcdf(file_name)

        search_range = 4
        search_range = np.min([first_appearence, search_range])
        # Number of pixels around the average detection centre
        d = 4
        clipped = goes_ds.isel(
            {
                "t": slice(first_appearence - search_range, first_appearence),
                "y": slice(first_coords[0] - d, first_coords[0] + d + 1),
                "x": slice(first_coords[1] - d, first_coords[1] + d + 1),
            }
        )
        # save the positive examples
        if (
            (clipped.x.size == d * 2 + 1)
            and (clipped.y.size == d * 2 + 1)
            and (clipped.t.size == search_range)
        ):
            print("okay to save!")
            if globe.is_land(init_lat, init_lon):
                file_name = (
                    "/gws/nopw/j04/aopp/jowanf/land/data/DCC_"
                    + str(year)
                    + "_"
                    + str(pd.DatetimeIndex(clipped.t).dayofyear[0])
                    + "_"
                    + str(hour0)
                    + "_"
                    + str(hourf)
                    + "_"
                    + str(i)
                )
                clipped.to_netcdf(file_name)
            else:
                file_name = (
                    "/gws/nopw/j04/aopp/jowanf/sea/data/DCC_"
                    + str(year)
                    + "_"
                    + str(pd.DatetimeIndex(clipped.t).dayofyear[0])
                    + "_"
                    + str(hour0)
                    + "_"
                    + str(hourf)
                    + "_"
                    + str(i)
                )
                clipped.to_netcdf(file_name)


if __name__ == "__main__":
    print("file running")
    day0 = int(sys.argv[1])
    year = int(sys.argv[2])
    extra_days = 6
    hours = 3
    for k in range(day0, day0 + extra_days):
        for i in range(0, int(np.ceil(24 / hours))):
            hour0 = range(1, 24, hours)[i]
            hourf = np.minimum(24, range(1, 24 + hours, hours)[i + 1])
            main_function(year, k, hour0, hourf)
