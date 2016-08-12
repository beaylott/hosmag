#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  hosmag.py

"""
A ho(mer) sma(rt) grid Python utility module, for use with HOMER
in pre-processing and post-processing.
"""

import pandas as pd
import numpy as np
import numpy.random as np_rand
import matplotlib.pyplot as plt

from calendar import monthrange,month_name

# A year with the correct weekday/weekend pattern (start on Sunday/not a leap year)
# as HOMER uses is 2006...
homer_standard_year_index=pd.date_range(start=pd.datetime(2006,1,1),periods=365)
month_names=[month_name[i] for i in range(1,13)]

admd_clnr_mean=1.464
admd_clnr_std=0.487

def generate_array_of_loads(base_load,daily_variability,time_step_to_time_step_variability,population_size):
    """
    Generate an annual load ready for HOMER including random variability
    using a given monthly/week load array. Return an array containing
    the annual load.
    """

    perturbation_values=[generate_perturbation_values(
        daily_variability=daily_variability,
        time_step_to_time_step_variability=time_step_to_time_step_variability
    ) for _ in range(population_size)]

    # NOTE: Base load is multiplied by 2 to get value in kW
    map_perturb=lambda x: np.multiply(base_load*2,x)

    return map(map_perturb,perturbation_values)

def calculate_admd(
    base_load,
    daily_variability,
    time_step_to_time_step_variability,
    number_of_customers_in_sample=200,
):
    """
    Calculate the after demand maximum diversity as per standard definitions
    returning a float with this value.
    """

    population_size=1000

    array_of_loads=generate_array_of_loads(base_load,daily_variability,time_step_to_time_step_variability,population_size)

    array_of_loads=np.array([i for i in array_of_loads])

    sample_max=[]
    for _ in range(population_size):
        sample_select=np.random.randint(0,population_size,size=number_of_customers_in_sample)
        sample_loads=array_of_loads[sample_select,:]
        agg_sample_load=np.sum(sample_loads,axis=0)
        admd=np.max(agg_sample_load)/number_of_customers_in_sample
        sample_max.append(admd)

    admd_mean=np.mean(sample_max)
    admd_std=np.std(sample_max)

    return admd_mean,admd_std

def find_admd_values(base_load):
    """
    Find the ADMD matching
    """

    mean,std=calculate_admd(base_load,0.001,0.001)

    def diff2(mean,std):
        return np.sqrt(np.power(mean-admd_clnr_mean,2)+np.power(std-admd_clnr_std,2))

    min_diff=diff2(mean,std)

    for dayv in np.arange(0.6,1.4,0.1):
        for timev in np.arange(0.5,1,0.1):
            mean,std=calculate_admd(base_load,dayv,timev)
            temp_diff=diff2(mean,std)
            if temp_diff<min_diff:
                min_diff=temp_diff
                print(min_diff,mean,std,dayv,timev)


"""
Functions for simulating demand side response in the community smart grid.
"""

def dsr_simple_peak_shifting(
    loads,
    peak_shifted_factor=0.1,
    proportion_deferrable=0.1,
    proportion_economy7=0.1,
):
    """
    This is a simple DSR programme where the critical peak period is matched
    to the UK domestic peak (4-8pm) and the off peak period is set to the
    economy 7 period.

    Remove (peak_shifted_factor)% of load from peak period (4pm - 8pm) and
    treat (proportion_deferrable)% of it as a fully deferrable load,
    (proportion_economy7)% as load shifted to the first 3 hours of economy 7,
    and (1-proportion_economy7-proportion_deferrable)% as a load distributed
    uniformly either side of the peak period.

    Return a tuple of DataFrames containing the shifted loads.
    """

    base_load=loads.copy()

    shifted_load=loads.copy()
    shifted_load.iloc[:,:]=np.zeros((48,24))

    base_load[32:38]=loads[32:38]*(1-peak_shifted_factor)

    shiftable_load =(loads.sum() - base_load.sum())*(1-proportion_deferrable)
    deferrable_load = np.array(loads.sum() - base_load.sum())*proportion_deferrable

    # Daily average deferrable load (as used in HOMER) is weighted sum of
    # weekend/weekday amounts.
    deferrable_load = pd.Series((5./7.)*deferrable_load[::2]+(2./7.)*deferrable_load[1::2],index=month_names,name=None)

    # A proportion of the shifted load is distributed 2 hours either side
    # of the peak period. Need to loop over the months and add the right
    # amounts in.
    for month_mode in shifted_load:
        total_shifted_amount=shiftable_load[month_mode]
        shifted_load.loc["15:00":"15:30",month_mode]=total_shifted_amount/2.
        shifted_load.loc["20:00":"20:30",month_mode]=total_shifted_amount/2.

    return base_load,shifted_load,deferrable_load

def dsr_traffic_lights_find_intervals(
    grid_intensity_file,
    dsr_rate=1.0,
):
    """
    Find the intervals (in time) for each day which exceed the DSR limit
    which is a weighted average multiplied by the 'DSR rate' pased to the
    function.
    """

    # Weighting scheme for calculating moving average for grid carbon intensity
    # reference level.
    ma_weights=np.hstack([np.arange(1,22),np.ones(7)*22.])
    ma_weights=ma_weights/np.sum(ma_weights)

    gi=grid_intensity_file
    gi_daily=gi.asfreq('1D')

    gi_average=[]
    # Find weighted average of first 28 days until window becomes valid
    for i in range(27):
        gi_average.append(np.average(gi_daily[:(i+1)],weights=ma_weights[(27-i):]))

    gi_average=np.array(gi_average)

    # Calculate weighted moving average by convolution...
    # Remove transient from start and replace with simple moving average.
    gi_daily_check=np.convolve(gi_daily.values,ma_weights,mode='valid')

    gi_average=pd.Series(np.hstack((gi_average,gi_daily_check)),index=gi_daily.index)


    # Find time intervals when amber bound is exceeded ...
    amber_ranges=[]
    red_ranges=[None]

    for gi_date in gi_average.index:
        # For each day need to determine longest contigious intervals
        # in which carbon emissions rise above limit.

        gi_day=gi[str(gi_date.date())]
        gi_average_day=gi_average[gi_date]

        # Find intervals which are greater than amber bound
        # Dummy value is attached to start to line up values correctly
        range_filter=np.hstack(((gi_day>gi_average_day*dsr_rate)[0],np.diff(np.array(gi_day>gi_average_day*dsr_rate))))
        if range_filter.any():
            amber_ranges_day=gi_day.index[range_filter]

            # Where carbon intensity is still above threshold at end of day
            # (odd number of Datetime's in index) need to add end of day as bound.
            if bool(len(amber_ranges_day)%2):
                amber_ranges_day=amber_ranges_day.append(pd.DatetimeIndex([gi_date+pd.DateOffset(1)]))
                amber_ranges.append(amber_ranges_day)
            else:
                amber_ranges.append(amber_ranges_day)

        else:
            amber_ranges.append([])

    amber_rangess=[]
    # Now go back through list and pair up start and ends of intervals
    for x in amber_ranges:
        collect=[]
        if x is not []:
            for y in zip(x[::2],x[1::2]):
                collect.append(y)
        amber_rangess.append(collect)

    return amber_rangess,gi_average

def __get_co2_energy_from_interval_fraction(interval,base_load,intensity,shifted_fraction):
    """
    Get the total energy and carbon emissions shifted from the base load
    if a fraction of the load is removed.
    """

    start,end=interval
    # NOTE: The timestamp at the end is removed as it is a bound/
    # not inclusive in selecting emissions/power shifted.
    times=pd.date_range(start=start,end=end,freq='30Min')[:-1]

    intervals_energy=base_load.ix[times]
    intervals_intensity=intensity.ix[times]


    total_energy_kwh=shifted_fraction*intervals_energy.values.sum()
    total_intensity_gCO2=shifted_fraction*np.multiply(intervals_intensity.values,intervals_energy.values).sum()

    return total_energy_kwh,total_intensity_gCO2

def __get_co2_energy_from_interval_amount(interval,base_load,intensity,shifted_amount_kwh):
    """
    Get the total energy and carbon emissions shifted from the base load
    if a fixed amount (in W) of the load is removed.
    """
    start,end=interval
    # NOTE: The timestamp at the end is removed as it is a bound/
    # not inclusive in selecting emissions/power shifted.
    times=pd.date_range(start=start,end=end,freq='30Min')[:-1]

    total_energy_kwh=times.size*shifted_amount_kwh

    intervals_intensity=intensity.ix[times]

    total_intensity_gCO2=np.multiply(intervals_intensity.values,shifted_amount_kwh).sum()

    return total_energy_kwh,total_intensity_gCO2

def __get_co2_from_interval(interval,intensity):
    """
    Get the total carbon intensity in an interval (useful when shifting
    a fixed amount of the load).
    """
    start,end=interval
    # NOTE: The timestamp at the end is removed as it is a bound/
    # not inclusive in selecting emissions/power shifted.
    times=pd.date_range(start=start,end=end,freq='30Min')[:-1]

    intervals_intensity=intensity.ix[times]

    total_intensity_gCO2=intervals_intensity.values.sum()

    return total_intensity_gCO2

def __calculate_co2_energy_deferrable_load(
    homer_deferrable_load,
    intensity
):
    """
    Calculate the total CO2 emissions in a HOMER deferrable load based
    on the mean of given carbon intensity baseline data.
    """
    co2=0.
    for idx in homer_deferrable_load.index:
        co2_intensity_mean=intensity[idx].mean()
        co2+=homer_deferrable_load[idx]*co2_intensity_mean
    return co2

def calculate_total_co2_energy_for_loads(
    homer_base_load,
    intensity,
    homer_shifted_load=None,
    homer_deferrable_load=None,
):
    """
    Calculate the total CO2 and energy use in the base load using
    30 minute grid intensity data.
    """
    total_energy=homer_base_load.sum()
    total_co2=intensity.multiply(homer_base_load).sum()

    if homer_shifted_load is not None:
        total_energy+=homer_shifted_load.sum()
        total_co2+=intensity.multiply(homer_shifted_load).sum()
    if homer_deferrable_load is not None:
        total_energy+=homer_deferrable_load.sum()
        total_co2+=__calculate_co2_energy_deferrable_load(homer_deferrable_load,intensity)

    total_co2=total_co2/1e6
    return total_energy,total_co2

def dsr_find_intervals_shifted_emissions(
    grid_intensity,
    limit_length=4,
    limit_number=365,
):
    """
    NEW FUNCTION:
    """
    total_energy_kwh=0.
    total_intensity_gCO2=0.

    results=[]

    annual_intervals=dsr_generate_dummy_whole_day_intervals(2006)

    if grid_intensity.index[0].year is not 2006:
        print("WARNING: Years do not coincide -set intensity index year to 2006")

    # Loop over each day in the interval set and calculate the total
    # emissions and power shift

    for day_intervals in annual_intervals:
        if day_intervals is []:
            results.append([])
        else:
            resultss=[]
            for interval in day_intervals:
                start,end=interval
                interval_max=interval
                # First pass calculate with whole interval
                co2_max=__get_co2_from_interval(interval,grid_intensity)

                # If a limit has been placed on the length of intervals we need to find the max co2 interval
                # of that length within the interval.
                difft=(end-start).to_timedelta64()/np.timedelta64(1,'h')

                if difft>limit_length:
                    co2_max=0.

                    start_temp=start
                    while start_temp+np.timedelta64(limit_length,'h')<end:
                        interval_temp=(start_temp,start_temp+np.timedelta64(limit_length,'h'))
                        co2_temp=__get_co2_from_interval(interval_temp,grid_intensity)

                        if co2_max<co2_temp:
                            interval_max=interval_temp
                            co2_max=co2_temp


                        start_temp=start_temp+np.timedelta64(30,'m')

                start,end=interval_max
                resultss.append(interval_max)
            results.append(resultss)

    return results



def dsr_traffic_lights_find_intervals_shifted_emissions(
    annual_intervals,
    homer_base_load,
    intensity,
    shifted_fraction,
    limit_length=24,
    limit_number=365,
    interval_identification_fuzzing=False,
):
    """

    OLD FUNCTION FOR FINDING MAX CO2 INTERVALS USE
    dsr_find_intervals_shifted_emissions

    Find the intervals with the largest carbon emissions for a given load
    profile. Returns a list of intervals, one for each day or None depending
    on whether or not there were any intervals in that day to start with, and, if
    there was a valid interval, the carbon emissions and power shifted.

    To get the total energy/CO2 in the intervals pass shifted_fraction=1.
    """

    total_energy_kwh=0.
    total_intensity_gCO2=0.

    results=[]

    # Loop over each day in the interval set and calculate the total
    # emissions and power shift

    for day_intervals in annual_intervals:
        if day_intervals is []:
            results.append([])
        else:
            resultss=[]
            for interval in day_intervals:
                start,end=interval
                interval_max=interval
                # First pass calculate with whole interval
                energy_max,co2_max=__get_co2_energy_from_interval(interval,homer_base_load,intensity,shifted_fraction)

                # If a limit has been placed on the length of intervals we need to find the max co2 interval
                # of that length within the interval.
                difft=(end-start).to_timedelta64()/np.timedelta64(1,'h')
                if difft>limit_length:
                    co2_max=0.

                    start_temp=start
                    while start_temp+np.timedelta64(limit_length,'h')<end:
                        interval_temp=(start_temp,start_temp+np.timedelta64(limit_length,'h'))
                        energy_temp,co2_temp=__get_co2_energy_from_interval(interval_temp,homer_base_load,intensity,shifted_fraction)
                        if co2_max<co2_temp:
                            interval_max=interval_temp
                            co2_max=co2_temp
                            energy_max=energy_temp

                        start_temp=start_temp+np.timedelta64(30,'m')

                start,end=interval_max
                resultss.append([interval_max,energy_max,co2_max])
            results.append(resultss)

    return results

def dsr_find_total_co2_energy_in_intervals(
    intervals,
    intensity,
    shifted_amount_kwh
):

    total_emissions_gCO2=0.
    total_energy_kwh=0.

    for day_intervals in intervals:
        for interval in day_intervals:

            start,end=interval
            # NOTE: The timestamp at the end is removed as it is a bound/
            # not inclusive in selecting emissions/power shifted.
            times=pd.date_range(start=start,end=end,freq='30Min')[:-1]

            intervals_intensity=intensity.ix[times]

            total_emissions_gCO2+=np.multiply(intervals_intensity.values,shifted_amount_kwh).sum()
            total_energy_kwh+=times.size*shifted_amount_kwh


    return total_emissions_gCO2,total_energy_kwh


def dsr_calculate_co2_energy_for_interval_length_series(homer_base_load,intensity,shifted_amount):
    """
    Calculating a series of the largest emission intervals.
    """

    result=[]

    for i in [1,2,3,4,6,8,12,16,20,24]:

        ret=dsr_find_intervals_shifted_emissions(intensity,limit_length=i)
        total_co2,total_energy=dsr_find_total_co2_energy_in_intervals(ret)

        print((total_energy,total_co2))

        result.append((total_energy,total_co2))

    return result

def dsr_create_program(
    intervals,
    homer_base_load,
    shifted_amount_kwh
):
    """
    Simulate a DSR by reducing the base load by a fixed amount of kWh and
    add this removed part to the deferrable load.
    """
    new_homer_base_load=homer_base_load.copy()

    defer_loads=[]

    for day_intervals in intervals:

        day_defer_load=0.

        for interval in day_intervals:
            start,end=interval
            times=pd.date_range(start=start,end=end,freq='30Min')[:-1]
            new_homer_base_load.ix[times]=new_homer_base_load.ix[times]-shifted_amount_kwh

            day_defer_load+=times.size*shifted_amount_kwh

        defer_loads.append(day_defer_load)

    homer_deferrable_load=np.array([np.average(defer_loads)]*12)

    return new_homer_base_load,homer_deferrable_load

def dsr_generate_results_from_homer_outputs(
    grid_intensity,
    results_folder_path='./',
):
    """
    This function generates the DSR final results from a set of HOMER output
    files. It assumes the file name format is 'results_X_Y_Z.csv' where X
    is the demand flexibility in %, Y is the DSR amount in W, Z is the PV
    capacity in kW.

    The values for X,Y,Z are preset here.
    """

    import os.path

    demand_flexibility=[50.,100.]
    pv_capacity_kw=[200.,500.,1000.]
    dsr_amount_w=[0.,50.,100.,150.]

    filename_base='results_'

    results=[]

    for df in demand_flexibility:
        for dsr in dsr_amount_w:

            df_=df

            if dsr is 0.:
                df_=0.

            filename_sys=os.path.join(results_folder_path,filename_base+'sys_'+'%i_%i'%(df_,dsr)+'.csv')
            ho_sys=parse_homer_output_systems(filename_sys)

            for pv in pv_capacity_kw:

                filename=os.path.join(results_folder_path,filename_base+'%i_%i_%i'%(df_,dsr,pv)+'.csv')

                ho=parse_homer_output_files(filename)

                co2_emissions=calculate_emissions_from_homer_output(ho,grid_intensity,200,0,0,None)

                ren_frac=ho_sys['System/Ren Frac (%)'][ho_sys['Architecture/PV NG Base (kW)']==pv].values[0]

                results.append([df,dsr,pv,co2_emissions,ren_frac])

    return results


def dsr_traffic_lights(
    homer_base_load,
    amber_intervals,
    red_intervals,
    amber_rate=0.1,
    red_rate=0.2,
    deferred_factor=0.2,
    shift_forward_factor=0.67,
    shift_time_interval=3,
):
    """
    This master routine calculates the base,shifted, and deferrable loads for HOMER
    from the provided amber and red interval lists. The algorithm assumes that
    energy demand from the red interval 'spills over' into the surrounding
    intervals first. It then spills over again into the off peak period.
    A certain fixed proportion of the load from each period is removed
    and treated as deferrable load.

    The red intervals are assumed to be within the amber intervals.

    It is assumed that (deferred_factor)% of the shifted load is deferrable from
    each period and added to the deferrable load list for that month.
    This is then averaged to calculate the defrrable load array for HOMER.

    Of the remaining (1-deferred_factor)% shifted load in each period,
    (shift_forward_factor)% is assumed to be moved to the next
    (shift_time_interval) hours and (1-shift_forward_factor)% to the previous
    (shift_time_interval) hours.
    """

    homer_base_load=homer_base_load.copy()

    ref=homer_base_load.copy() #DEBUG

    year=homer_base_load.index[0].year
    day_index=pd.date_range(start=pd.datetime(year,1,1),end=pd.datetime(year,12,31))
    interval_index=pd.date_range(start=pd.datetime(year,1,1),end=pd.datetime(year+1,1,1),freq='30Min')[:-1]

    deferrable_load=pd.Series(np.zeros(day_index[-1].dayofyear),index=day_index)
    shifted_load=pd.Series(np.zeros(day_index[-1].dayofyear*48),index=interval_index)

    # Loop through red intervals, remove deferrable load
    for amber_interval0,red_interval0 in zip(amber_intervals,red_intervals):

        #Unpack
        amber_interval=amber_interval0[0]
        red_interval=red_interval0[0]

        red_start,red_end=red_interval
        red_times=pd.date_range(start=red_start,end=red_end,freq='30Min')[:-1]

        red_times_date=pd.Timestamp(red_times[0].date())

        red_total_shifted_amount=homer_base_load[red_times].sum()*red_rate

        # Reduce base load at amber times by amber factor
        homer_base_load[red_times]=homer_base_load[red_times]*(1-red_rate)

        # Add deferrable load component to daily deferrable load schedule
        deferrable_load[red_times_date]=red_total_shifted_amount*deferred_factor

        # Calculate amount to be re-allocated under DSR to surrounding times
        red_shifted_amount=red_total_shifted_amount*(1-deferred_factor)

        # Calculate time indices for shifted windows
        red_early_time_start=red_times[0]-pd.Timedelta(shift_time_interval,'h')
        red_late_time_end=red_times[-1]+pd.Timedelta(shift_time_interval,'h')

        if red_early_time_start<interval_index[0]:
            red_early_time_start=interval_index[0]

        if red_late_time_end>interval_index[-1]:
            red_late_time_end=interval_index[-1]

        red_shifted_times_early=pd.date_range(start=red_early_time_start,end=red_times[0],freq='30Min')[:-1]
        red_shifted_times_later=pd.date_range(start=red_times[-1],end=red_late_time_end,freq='30Min')[:-1]

        print(red_shifted_times_early)

        # Calculate amounts allocated to early/later window
        red_shifted_amount_early=red_shifted_amount*(1-shift_forward_factor)
        red_shifted_amount_later=red_shifted_amount*shift_forward_factor

        # Distribute these amounts equally across the shifting windows
        shifted_load[red_shifted_times_early]+=red_shifted_amount_early/red_shifted_times_early.size
        shifted_load[red_shifted_times_later]+=red_shifted_amount_later/red_shifted_times_later.size

        # Now consider larger amber windows...
        amber_start,amber_end=amber_interval
        amber_times=pd.date_range(start=amber_start,end=amber_end,freq='30Min')[:-1]

        amber_times_date=pd.Timestamp(red_times[0].date())

        #Cut out red intervals
        amber_times_cut=amber_times.difference(red_times)

        amber_total_shift=(homer_base_load[amber_times_cut].sum()+shifted_load[amber_times_cut].sum())*amber_rate

        deferrable_load[amber_times_date]+=amber_total_shift*deferred_factor

        amber_shifted_amount=amber_total_shift*(1-deferred_factor)

        # Reduce base and already shifted demand from red by amber factor
        homer_base_load[amber_times_cut]=homer_base_load[amber_times_cut]*(1-amber_rate)
        shifted_load[amber_times_cut]=shifted_load[amber_times_cut]*(1-amber_rate)

        # Take the start and ends of the amber interval and extend outwards
        # to define the times when demand will be shifted too.

        amber_early_time_start=amber_times[0]-pd.Timedelta(shift_time_interval,'h')
        amber_late_time_end=amber_times[-1]+pd.Timedelta(shift_time_interval,'h')

        if amber_early_time_start<interval_index[0]:
            amber_early_time_start=interval_index[0]

        if amber_late_time_end>interval_index[-1]:
            amber_late_time_end=interval_index[-1]

        amber_shifted_times_early=pd.date_range(start=amber_early_time_start,end=amber_times[0],freq='30Min')[:-1]
        amber_shifted_times_later=pd.date_range(start=amber_times[-1],end=amber_late_time_end,freq='30Min')[:-1]

        amber_shifted_amount_early=amber_shifted_amount*(1-shift_forward_factor)
        amber_shifted_amount_later=amber_shifted_amount*shift_forward_factor

        # Distribute these amounts equally across the shifting windows
        shifted_load[amber_shifted_times_early]+=amber_shifted_amount_early/amber_shifted_times_early.size
        shifted_load[amber_shifted_times_later]+=amber_shifted_amount_later/amber_shifted_times_later.size

    print(homer_base_load.sum()+shifted_load.sum()+deferrable_load.sum()-ref.sum()) #DEBUG

    return homer_base_load,shifted_load,deferrable_load

def dsr_unzip_intervals_energy_co2(intervals_energy_co2):
    """
    Utility function for removing intervals from combined result.
    """
    intervals=[]
    for entry in intervals_energy_co2:
        if len(entry) is 0:
            intervals.append([])
        else:

            interval,_,_=entry[0]
            intervals.append([interval])

    return intervals

def dsr_find_year_from_intervals(annual_intervals):
    """
    Find the year based on data in the intervals list.
    """
    # Year of first value should be sufficient...
    return annual_intervals[0][0][0].year

def dsr_traffic_lights_generate_heat_map(annual_intervals,heatmap_idx=None):
    """
    Construct a heat map of the intervals. The output heatmap can be passed again
    into the function with a different set of intervals to accumulate values.
    Returns a pd.Series object containing the heat map indexed to the associated
    time intervals.
    """

    year=dsr_find_year_from_intervals(annual_intervals)

    if heatmap_idx is None:
        heatmap_idx=pd.Series(np.zeros(365*48,dtype='int32'),index=pd.date_range(start="%i-01-01"%year,end="%i-12-31 23:30"%year,freq='30Min'))

    for day_intervals in annual_intervals:
        for interval in day_intervals:
            start,end=interval
            times=pd.date_range(start=start,end=end,freq='30Min')[:-1]
            heatmap_idx.ix[times]+=1

    return heatmap_idx

def __interpolate(x,y,z):
    """
    Interpolate z onto the uniform mesh defined by x,y.
    """

    import scipy.interpolate

    # Set up a regular grid of interpolation points
    xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    # Interpolate
    zi = scipy.interpolate.griddata((x, y), z, (xi, yi), method='linear')

    return xi,yi,zi


def plot_surface(x,y,z,x_label,y_label,z_label):
    """
    Plot a surace map using interpolation and the matplotlib contourf
    function.
    """
    import matplotlib.pyplot as plt

    xi,yi,zi=__interpolate(x,y,z)

    plt.contourf(xi,yi,zi)
    plt.gray()
    plt.colorbar(label=z_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def plot_systems_surface(homer_output_systems):
    """
    Plot a systems surface based on data in the exported homer output systems
    file.
    """

    ax_y={'Architecture/PV NG Base (kW)':'PV Capacity (kW)'}
    ax_x={'Architecture/1kWh LA':'Battery Capacity (kWh)'}
    #,'Architecture/NG Wind 250':'Wind Capacity (kW)','Architecture/PV NG Base (kW)':'PV Capacity (kW)'}

    z_list={'System/Ren Frac (%)':'Renewable Fraction (%)','Carbon Emissions (tCO2e/y)':'Carbon Emissions (tCO2e/y)','Cost/Initial capital (€)':'Initial Capital Spend (£)','Adj. LCOE (€/kWh)':'LCOE (£/kWh)'}

    import matplotlib.pyplot as plt

    homer_output_systems=homer_output_systems.fillna(0.)

    idx=list(ax_x.keys())
    idx.extend(list(ax_y.keys()))
    xy=homer_output_systems[idx].values

    x=xy[:,0]
    y=xy[:,1]

    fig=plt.figure()
    i=1
    for z_name in z_list.keys():
        z=homer_output_systems[z_name].values

        ax = fig.add_subplot(2,2,i)

        plot_surface(x,y,z,list(ax_x.values())[0],list(ax_y.values())[0],z_list[z_name])
        i+=1

    plt.show()

def plot_systems_line(homer_output_systems):
    """
    Plots distribution of performance metrics for single variable (e.g. PV capacity).
    """

    ax_x={'Architecture/PV NG Base (kW)':'PV Capacity (kW)'}

    z_list={'System/Ren Frac (%)':'Renewable Fraction (%)','Carbon Emissions (tCO2e/y)':'Carbon Emissions (tCO2e/y)','Cost/Initial capital (€)':'Initial Capital Spend (£)','Adj. LCOE (€/kWh)':'LCOE (£/kWh)'}

    import matplotlib.pyplot as plt

    idx=list(ax_x.keys())
    x=homer_output_systems[idx].values

    fig=plt.figure()
    i=1
    for z_name in z_list.keys():
        z=homer_output_systems[z_name].values

        ax = fig.add_subplot(2,2,i)

        plt.scatter(x,z)
        plt.gray()

        plt.xlabel(list(ax_x.values())[0])
        plt.ylabel(z_list[z_name])

        i+=1

    plt.show()

def plot_systems_compare(homer_output_systems1,homer_output_systems2):
    """
    Plots distribution of difference between performance metrics for two
    variables (e.g. PV capacity vs. battery capacity).

    Interpolates metrics onto same mesh and then subtracts them.
    """

    ax_y={'Architecture/PV NG Base (kW)':'PV Capacity (kW)'}
    ax_x={'Architecture/Batt 1kWh LA':'Battery Capacity (kWh)'}

    #,'Architecture/NG Wind 250':'Wind Capacity (kW)','Architecture/PV NG Base (kW)':'PV Capacity (kW)'}

    z_list={'System/Ren Frac (%)':'Renewable Fraction (%)','Carbon Emissions (tCO2e/y)':'Carbon Emissions (tCO2e/y)','Cost/Initial capital (€)':'Initial Capital Spend (£)','Adj. LCOE (€/kWh)':'LCOE (£/kWh)'}

    homer_output_systems1=homer_output_systems1.fillna(0.)
    homer_output_systems2=homer_output_systems2.fillna(0.)

    import matplotlib.pyplot as plt
    import scipy.interpolate

    idx=list(ax_x.keys())
    idx.extend(list(ax_y.keys()))

    xy1=homer_output_systems1[idx].values
    xy2=homer_output_systems2[idx].values

    x1=xy1[:,0]
    y1=xy1[:,1]

    x2=xy2[:,0]
    y2=xy2[:,1]

    # Set up a regular grid of interpolation points. Both outputs should
    # have same ranges for each variable so the min/max for either should
    # be compatible.
    xi, yi = np.linspace(x1.min(), x1.max(), 100), np.linspace(y1.min(), y1.max(), 100)
    xi, yi = np.meshgrid(xi, yi)

    fig=plt.figure()
    i=1
    for z_name in z_list.keys():
        z1=homer_output_systems1[z_name].values
        z2=homer_output_systems2[z_name].values

        ax = fig.add_subplot(2,2,i)

        # Interpolate (x,y,z) values onto common grid..
        zi1 = scipy.interpolate.griddata((x1, y1), z1, (xi, yi), method='linear')
        zi2 = scipy.interpolate.griddata((x2, y2), z2, (xi, yi), method='linear')

        # Subtract one value from the other
        zi=zi1-zi2

        plt.contourf(xi,yi,zi)
        plt.gray()
        plt.colorbar(label=z_list[z_name])
        plt.xlabel(list(ax_x.values())[0])
        plt.ylabel(list(ax_y.values())[0])

        i+=1

    plt.show()

    #


"""
Functions relating to excess capacity statistics and plots.
"""

def generate_excess_capacity_from_homer_output(homer_output):
    """
    Based on the detailed HOMER output calculate a series of the excess
    power/capacity in the smart grid across the year.
    """
    data=homer_output[['Total Electrical Load Served','PV NG Base Power Output','Nobel Grid Generic LA Maximum Discharge Power','NG Wind 250 Power Output']]
    excess=pd.Series((data[[3]].values+data[[2]].values+data[[1]].values-data[[0]].values).ravel())

    return excess

def calculate_expected_income_from_cm_events(
    homer_output,
    number_of_events,
    capacity_offer,
    income_per_kw,
    penalty_per_event,
):

    """
    This function calculates the potential income which can be derived
    from participation in the capacity market mechanism for a
    given offer of capacity.

    It determines the likelihood of the excess capacity being below
    a certain amount. The revenue is then calculated assuming the probability
    of having enough capacity when a capacity event occurs follows a binomial
    distribution where p is equal to the likelihood of having greater than
    or equal to the capacity offer.

    The income I is then a random variable with expectation

    E(I) = anp - n(1-p)b = p(a-b)n - nb
    STD(I)=SQRT(anpq - bnpq)=SQRT((a-b)npq)

    """
    p=0.8
    q=1-p

    revenue=capacity_offer*income_per_kw*number_of_events*p
    penalty=penalty_per_event*q*number_of_events

    income=revenue-penalty

    income_std=np.sqrt((revenue-penalty)*number_of_events*p*q)

    return income,income_std

def plot_excess_cap_cdf(homer_output):
    """
    Plot a frequency/survival curve of the excess capacity .
    """
    excess=generate_excess_capacity_from_homer_output(homer_output)
    ####excess.hist(cumulative=True,normed=1)

    #evaluate the histogram
    data=excess.values
    values, base = np.histogram(data, bins=40)
    #evaluate the cumulative
    cumulative = np.cumsum(values)
    # plot the cumulative function
    plt.plot(base[:-1], cumulative/np.sum(values), c='blue')
    #plot the survival function
    plt.plot(base[:-1], (len(data)-cumulative)/np.sum(values), c='green')

    plt.xlabel('Excess capacity (kW)')
    plt.ylabel('Cumulative probability')
    plt.gray()
    plt.show()

"""
Functions relating to visualisation of Sheffield Solar data.
"""

def parse_shef_solar(ss_meta_filename):
    ss_meta=pd.read_csv('./metadata.csv',index_col=0)
    return ss_meta

def plot_hist_orientation(ss_meta):
    (ss_meta['Orientation (deg from North)']-180).plot(kind='hist',bins=np.arange(-75,105,30))

    plt.xlabel('Orientation (degrees from south)')
    plt.ylabel('Number of systems')

    plt.show()

    return

def plot_hist_tilt(ss_meta):
    (ss_meta['Tilt (deg from horizontal)']).plot(kind='hist')

    plt.xlabel('Tilt (degrees)')
    plt.ylabel('Number of systems')

    plt.show()

    return

def plot_traffic_lights_heat_map(heatmap_idx):
    import plotly.offline as py
    import plotly.graph_objs as go

    import numpy as np

    heatmap_idx=heatmap_idx.values
    heatmap_idx=heatmap_idx.reshape((365,48))

    time_index=pd.date_range('00:00:00','23:30:00',freq='30Min')
    time_index=['%.2i:%.2i'%(i.hour,i.minute) for i in time_index]

    data = [

        go.Heatmap(
            z=heatmap_idx,
            x=time_index,
        ),

    ]
    plot_url = py.plot(data, filename='traffic_light_heat_map')

def dsr_traffic_lights_print_length_of_intervals(intervals):
    for day_intervals in intervals:
        for interval in day_intervals:
            start,end=interval
            print((end-start)/np.timedelta64(1,'h'))


def dsr_find_average_number_intervals_per_day(annual_intervals):
    """
    Counts the number of intervals in a given day and returns the average
    number per day for the whole year.
    """
    return np.mean([(int(len(i))) for i in annual_intervals])

def dsr_find_average_length_of_interval(annual_intervals):
    """
    For each day calculate the length in time and average across the year.
    Will use all intervals in each day. Returns the mean value.
    """
    annual_diffs=[]
    for day_intervals in annual_intervals:
        for interval in day_intervals:
            start,end=interval
            annual_diffs.append((end-start).to_timedelta64()/np.timedelta64(1,'h'))

    return np.mean(annual_diffs)

def dsr_find_max_co2_intervals(annual_intervals_energy_co2):
    """
    Search through the intervals in each day and find the one with the
    highest associated CO2 emissions. Return a list of lists of tuples
    of the intervals in each day with the highest co2 emissions, the
    energy shifted, and the CO2 emission reduction potential.
    """

    results=[]
    for day_intervals in annual_intervals_energy_co2:
        co2t=0.
        for interval,energy,co2 in day_intervals:

            if co2>co2t:
                co2t=co2
                interval_max=interval
                energy_max=energy

        if day_intervals:
            results.append([[interval_max,energy_max,co2t]])
        else:
            results.append([])

    return results

def dsr_find_average_length_of_max_co2_interval(annual_intervals_energy_co2):
    """
    """

    annual_diffs=[]
    annual_co2=0.
    annual_energy=0.
    for day_intervals in annual_intervals_energy_co2:
        co2t=0.
        for interval,energy,co2 in day_intervals:
            start,end=interval
            if co2>co2t:
                co2t=co2
                difft=(end-start).to_timedelta64()/np.timedelta64(1,'h')
                annual_co2t=co2
                annual_energyt=energy
        if day_intervals:
            annual_diffs.append(difft)
            annual_co2+=annual_co2t
            annual_energy+=annual_energyt


    return np.mean(annual_diffs),annual_energy,annual_co2,len(annual_diffs)

def dsr_generate_dummy_whole_day_intervals(year):
    """
    This will generate daily intervals for whole day. Useful for finding
    max CO2/energy intervals in any given day using existing functions.
    """
    starts=list(pd.date_range(start="%s-01-01 00:00"%year,end="%i-12-31 00:00"%year,freq="1D"))
    ends=list(pd.date_range(start="%s-01-01 23:30"%year,end="%i-12-31 23:30"%year,freq="1D"))

    intervals=[]
    for start,end in zip(starts,ends):
        intervals.append([(start,end)])

    return intervals

def dsr_generate_simple_peak_shift_intervals(year):
    """
    This will generate daily intervals for peak times.
    """
    starts=list(pd.date_range(start="%s-01-01 16:00"%year,end="%i-12-31 16:00"%year,freq="1D"))
    ends=list(pd.date_range(start="%s-01-01 19:30"%year,end="%i-12-31 19:30"%year,freq="1D"))

    intervals=[]
    for start,end in zip(starts,ends):
        intervals.append([(start,end)])

    return intervals

def dsr_traffic_lights_generate_bound_length_number_series(
    grid_intensity_file,
    base_load,
    shifted_fraction=0.1,
    limit_length=24,
):

    results=[]

    for dsr_rate in np.arange(0.7,1.3,0.02):
        intervals,_=dsr_traffic_lights_find_intervals(grid_intensity_file,dsr_rate=dsr_rate)

        result=dsr_traffic_lights_find_intervals_shifted_emissions(intervals,base_load,grid_intensity_file,shifted_fraction,limit_length=limit_length)

        #number=dsr_find_average_number_intervals_per_day(intervals)
        length,annual_energy,annual_co2,number=dsr_find_average_length_of_max_co2_interval(result)

        simple_intervals=dsr_generate_simple_peak_shift_intervals()

        result=dsr_traffic_lights_find_intervals_shifted_emissions(simple_intervals,base_load,grid_intensity_file,shifted_fraction)

        si_length,si_annual_energy,annual_co2_si,si_number=dsr_find_average_length_of_max_co2_interval(result)


        results.append([dsr_rate,number,length,annual_energy,annual_co2/1e6,annual_co2_si/1e6])

    results=np.array(results)

    return pd.DataFrame(results[:,1:],index=results[:,0],columns=['number of days containing interval','average length of max CO2 interval h','total energy kWh','total CO2 emissions tCO2e (amber ToU)','total CO2 emissions tCO2e (peak shift)'])

def plot_bound_variance(
    results_from_dsr_traffic_lights_generate_bound_length_number_series,
):
    input_data=results_from_dsr_traffic_lights_generate_bound_length_number_series

    import plotly.offline as py
    import plotly.graph_objs as go

    from plotly.graph_objs import Layout,XAxis,YAxis,Annotation

    common_x_axis=input_data.index.values

    trace1 = go.Scatter(
        x=common_x_axis,
        y=input_data['number of days containing interval'],
        name='number of days with DSR interval'
    )

    trace2 = go.Scatter(
        x=common_x_axis,
        y=input_data['average length of max CO2 interval h'],
        name='average length of max CO2 interval h',
        yaxis='y2'
    )

    trace3 = go.Scatter(
        x=common_x_axis,
        y=input_data['total energy kWh'],
        name='total elec use reduction potential kWh',
        yaxis='y3'
    )

    trace4 = go.Scatter(
        x=common_x_axis,
        y=input_data['total CO2 emissions tCO2e (amber ToU)'],
        name='total CO2 emission reduction potential tCO2e (DSR ToU)',
        yaxis='y4'
    )

    trace5 = go.Scatter(
        x=common_x_axis,
        y=input_data['total CO2 emissions tCO2e (peak shift)'],
        name='total CO2 emission reduction potential tCO2e (simple peak shift)',
        yaxis='y4',
        line = dict(
            color = ('rgb(0, 0, 0)'),
            width = 3,
            dash = 'dot',
        )
    )

    data = [trace1,trace2,trace3,trace4,trace5]

    layout = Layout(
        autosize=True,
        title='Variance of DSR characteristics with amber bound (10% demand shift/reduction).',
        height=500,
        width=1620,
        xaxis=XAxis(
            autorange=True,
            domain=[0.06, 0.89],
            title='Ratio of DSR Bound to Average Carbon Emissions',
            type='linear'
        ),
        yaxis=YAxis(
            autorange=True,
            tickfont=dict(
                color='#1f77b4'
            ),
            title='number of DSR periods per year',
            titlefont=dict(
                color='#1f77b4'
            ),
            type='linear'
        ),
        yaxis2=YAxis(
            anchor='free',
            autorange=True,
            overlaying='y',
            position=0,
            side='left',
            tickfont=dict(
                color='#ff7f0e'
            ),
            title='average length of max CO2 interval h',
            titlefont=dict(
                color='#ff7f0e'
            ),
            type='linear'
        ),
        yaxis3=YAxis(
            anchor='free',
            autorange=True,
            overlaying='y',
            position=0.9,
            side='right',
            tickfont=dict(
                color='rgb(44, 160, 44)'
            ),
            title='total energy shift potential kWh',
            titlefont=dict(
                color='rgb(44, 160, 44)'
            ),
            type='linear'
        ),
        yaxis4=YAxis(
            anchor='free',
            autorange=True,
            overlaying='y',
            position=0.95,
            side='right',
            tickfont=dict(
                color='#d62728'
            ),
            title='total CO2 emission reduction potential tCO2e',
            titlefont=dict(
                color='#d62728'
            ),
            type='linear'
        ),
        annotations=[
            Annotation(
                text="%f"%input_data['total CO2 emissions tCO2e (peak shift)'].values[0],
                x=0.32106457242582903,
                y=0.0625,
                xref="paper",
                yref="paper",
                ax=0,
                ay=-50,
            )
        ]
    )

    fig = go.Figure(data=data, layout=layout)
    plot_url = py.plot(fig, filename='bound_variance')

def plot_traffic_lights(
    intensity_file,
    average,
    amber_rate=1.05,
    red_rate=1.2,
):

    import plotly.offline as py
    import plotly.graph_objs as go

    from plotly.graph_objs import Line,Marker,XAxis,YAxis

    common_x_axis=intensity_file.index

    # Expand average to match number of x datum to avoid aliasing
    average=average.reindex(intensity_file.index).fillna(method='ffill')

    amber_bound = go.Scatter(
        name='DSR Threshold',
        x=common_x_axis,
        y=average*amber_rate,
        fill='tonexty',
        fillcolor='rgba(51, 204, 51, 0.3)',
        line=Line(
            width=0
        ),
        marker=Marker(
            color='444'
        ),

        mode='lines',

    )

    trace = go.Scatter(
        name='Average Grid Carbon Intensity',
        x=common_x_axis,
        y=average,
        line=Line(
            color='rgb(0, 0, 0)'
        ),
        mode='lines',
    )

    trace2 = go.Scatter(
        name='Grid Carbon Intensity',
        x=common_x_axis,
        y=intensity_file,
        line=Line(
            color='rgb(0, 0, 1)',
            width=1
        ),
        mode='lines',
        opacity=0.34,
    )

    data = [trace,amber_bound,trace2]

    layout = go.Layout(

        autosize=True,
        title=None,
        xaxis=XAxis(
            autorange=True,
            title='Date',
            type='date'
        ),
        yaxis=YAxis(
            autorange=True,
            title='Grid Carbon Intensity 2014 gCO2e/kWh',
            type='linear'
        )
    )

    fig = go.Figure(data=data, layout=layout)

    plot_url = py.plot(data, filename='intensity_traffic')

def parse_grid_intensity_files(grid_intensity_filename):
    """
    Parser for EarthNotes grid intensity files. Note: Output contains NaNs.
    """
    return pd.read_csv(grid_intensity_filename,header=3,index_col=0)

def process_grid_intensity_data(grid_intensity_file,year=None):
    """
    This function takes the grid intensity data from EarthNotes and
    re-samples it at 30 minute intervals using linear interpolation and
    filling in missing values as it goes.
    """
    # Condition grid intensity input data to fill in missing times and perform
    # linear interpolation to get 30 minute data set.
    grid_intensity_dti=grid_intensity_file.set_index(pd.to_datetime(grid_intensity_file.index)) # Create DatetimeIndex based on existing labels.

    grid_intensity_ri=grid_intensity_dti.asfreq('30min',method=None) # Reindex to 30 minute intervals.
    grid_intensity_index_last=grid_intensity_ri.index[-1] # Take final time.
    grid_intensity_ri=grid_intensity_ri.reindex(grid_intensity_ri.index.append(pd.DatetimeIndex([pd.Timedelta('30Min')+grid_intensity_index_last]))) # Add a final 30 minute interval.

    #If year passed reindex result array to that year
    if year:
        new_index=pd.date_range(start="%i-01-01 00:00"%year,end="%i-12-31 23:30"%year,freq="30Min")
        grid_intensity_ri.index=new_index
    grid_intensity_ri=grid_intensity_ri.interpolate() # Interpolate intermediate 30 minute data and missing data.

    return grid_intensity_ri['Mean intensity gCO2/kWh']

def parse_torrs_hydro_data(torrs_hydro_control_data_filename):
    """
    """
    #
    thd=pd.read_csv(torrs_hydro_control_data_filename)
    # Index df using combined Date and Time column (BEWARE date format)
    time_index=pd.to_datetime(thd['Date'].values+' '+thd['Time'].values,dayfirst=True)
    thd.index=time_index

    thd['Ex Meter'][0]=0.
    thd['Ex Meter']=thd['Ex Meter'].fillna(method='pad')

    # Remove X* columns with null data
    #thd=thd[['Pmax','Pmin','Pavg','Hmax','Hmin','Havg','Ex Meter']]

    return thd

#TODO: Need to do something about zeroes in output load files from HEUS - set to nan?
def parse_input_load_files(input_load_filename):
    """
    Parse the tables produced by HEUS extractor spreadsheet.
    """
    input_loads=pd.read_csv(input_load_filename,header=2,index_col=0)
    meta=pd.read_csv(input_load_filename,nrows=2,header=None,names=np.array(input_loads.columns))

    return input_loads,meta

def parse_zcb_datawithtime(zcb_datawithtime_filename):
    return pd.read_csv('./tenyearsdatawithtime.csv',index_col=1,header=None,names=['n','day','hour','onshore wind', 'offshore wind', 'wave', 'tidal', 'solar', 'traditional electricity demand'])

def generate_perturbation_values(daily_variability=0.1,time_step_to_time_step_variability=0.1):
    daily_perturbation_values=np.repeat(np_rand.randn(365)*daily_variability,repeats=24*2)

    time_step_perturbation_value=np_rand.randn(365*24*2)*time_step_to_time_step_variability

    perturbation_values=1+daily_perturbation_values+time_step_perturbation_value

    return perturbation_values

def __isweekend(ts):
    if ts.weekday() is 5 or ts.weekday() is 6:
        return True
    else:
        return False

def __get_daily_profile_for_day_number(input_load_profiles,day_number,year_index):
    """
    Convenience function for generating a profile for a given ISO
    day number.
    """
    ts=year_index[day_number-1]
    return input_load_profiles.iloc[:,(ts.month-1)*2+int(__isweekend(ts))]

def parse_ev_loads(folder_root='./'):

    import os.path as osp

    ev_domestic_filename='ev domestic.csv'
    ev_commercial_filename='ev commercial.csv'

    evd_filename=osp.join(folder_root,ev_domestic_filename)
    evc_filename=osp.join(folder_root,ev_commercial_filename)

    evd=pd.read_csv(evd_filename,index_col=0,skipfooter=1,engine='python')
    evc=pd.read_csv(evc_filename,index_col=0,skipfooter=1,engine='python')

    return (evd,evc)

def create_ev_loads(
    ev_domestic_load,
    ev_commercial_load,
    number_of_domestic_evs,
    number_of_commercial_evs,
    year,
):
    """
    Generate an ev load based on a combination of provided residential
    and commercial EV load profiles.
    """
    year_index=pd.date_range(start="%s-01-01"%year,end="%s-12-31"%year)
    day_index=pd.date_range(start="%s-01-01"%year,end="%s-01-01"%(year+1),freq='30Min')[:-1]
    days_in_year=year_index[-1].dayofyear

    ev_load=np.array([0.]*48*year_index[-1].dayofyear,dtype=np.float64)

    # Loop over days in year and create numpy array containing ev load inputs
    day_number=1
    while day_number<=days_in_year:
        ts=year_index[day_number-1]

        if __isweekend(ts):
            ev_load[(day_number-1)*48:(day_number-1)*48+48]=2*(number_of_domestic_evs*ev_domestic_load['Weekend Load kWh']+number_of_commercial_evs*ev_commercial_load['Weekend Load kWh']).values
        else:
            ev_load[(day_number-1)*48:(day_number-1)*48+48]=2*(number_of_domestic_evs*ev_domestic_load['Weekday Load kWh']+number_of_commercial_evs*ev_commercial_load['Weekday Load kWh']).values

        day_number+=1

    return pd.Series(ev_load,index=day_index)

heat_pump_average_daily_load_kwh = np.array(
    [
        14,
        13.5,
        10.5,
        7,
        6,
        2.5,
        1.5,
        3,
        4.5,
        7,
        12,
        12,
    ]
)

def create_homer_base_load(
    input_loads,
    year,
    number_of_households,
    heat_pumps=0.,
):
    """
    Create the HOMER base load from the provided weekly/monthly load array.
    """
    year_index=pd.date_range(start="%s-01-01"%year,end="%s-12-31"%year)
    day_index=pd.date_range(start="%s-01-01"%year,end="%s-01-01"%(year+1),freq='30Min')[:-1]
    days_in_year=year_index[-1].dayofyear

    # Pre allocate array
    homer_base_load=np.array([0.]*48*year_index[-1].dayofyear,dtype=np.float64)

    # Loop over days in year and create numpy array containing homer load inputs
    # NOTE that standard year index is used to get correct weekday/weekend pattern
    day_number=1
    extra_load=0.
    while day_number<=days_in_year:

        if heat_pumps>0:
            extra_load=heat_pumps*heat_pump_average_daily_load_kwh[year_index[day_number-1].month-1]/48.


        homer_base_load[(day_number-1)*48:(day_number-1)*48+48]=extra_load+number_of_households*__get_daily_profile_for_day_number(input_loads,day_number,homer_standard_year_index)

        day_number+=1

    return pd.Series(homer_base_load,index=day_index)

def add_heat_pumps_to_deferabble_load(
    deferrable_load,
    number_of_heat_pumps,
):
    """
    Add a model of the heat pump power consumption to the base load.
    """
    deferrable_load=deferrable_load+number_of_heat_pumps*heat_pump_average_daily_load

    return deferrable_load

def add_noise_to_homer_load(
    homer_load,
    daily_variability=0.1,
    time_step_to_time_step_variability=0.1,
    perturbation_values=None,
):
    """
    Add random variability to a homer load.
    """

    if perturbation_values is None:
        perturbation_values=generate_perturbation_values(daily_variability=0.1,time_step_to_time_step_variability=0.2)

    homer_base_load=np.multiply(homer_load,perturbation_values)

    return homer_load

def write_homer_loads(
    homer_base_load,
    write_csv=True,
    add_noise=False,
    daily_variability=0.1,
    time_step_to_time_step_variability=0.1,
):

    """
    Based on average monthly and weekday/weekend profiles create a synthetic
    annual load with day-to-day and time step to time step variability. The values
    for this are based on HOMER defaults. Return a tuple containing the file handles
    for the written output.
    """

    homer_filename_base='homer_load'

    # Add noise
    if add_noise:
        perturbation_values=generate_perturbation_values(daily_variability=0.1,time_step_to_time_step_variability=0.2)
        homer_base_load=add_noise_to_homer_load(homer_base_load,perturbation_values)

    # HOMER reads files in kW not kWh so need to multiply outputs by 2...

    if write_csv:
        np.savetxt(homer_filename_base+"_base.txt",homer_base_load*2,fmt='%.6f')

    return homer_base_load*2

"""
Functions for creating the tariff input data for HOMER.
"""

def create_simple_tariff(purchase=0.172,sellback=0.):
    """
    Create a dummy simple tariff for import into the real time rates
    tab in HOMER. This provides the advanced battery control options whilst
    using a simple tariff.

    Note that non-zero sellback rates affect the dispatch of battery storage
    resource in a way which may not be desirable (e.g. battery is sold to grid
    instead of self-consumed).
    """
    simple_tariff_a=np.array([[purchase,sellback]]*48*365)
    simple_tariff=pd.DataFrame(simple_tariff_a,index=pd.date_range(start=pd.datetime(2006,1,1),periods=365*48,freq='30Min'),columns=['power price','sellback price'])

    return simple_tariff

def create_economy7_tariff(peak_power_price=0.203,off_peak_power_price=0.0878,sellback_price=0.):
    """
    Create a economy 7 tariff schedule for use in HOMER.
    """
    import datetime
    tariff=create_simple_tariff(peak_power_price)
    time=tariff.index.time
    selector=((datetime.time(0,30)<=time) & (time<=datetime.time(7,0)))
    tariff['power price'][selector]=off_peak_power_price
    return tariff

def write_homer_price_file(prices,filename='./homer_prices.txt'):
    """
    Write a text file in the format HOMER accepts for real time grid
    price data.
    """
    prices.to_csv(filename,sep=' ',header=False,index=False)

"""
*** RESULTS ***
"""

"""
Functions for parsing the HOMER output files.
"""

def parse_homer_output_files(homer_output_filename):
    """
    Parser for HOMER output files (as created by clicking Export in results
    window).
    """
    return pd.read_csv(homer_output_filename,header=[1,2],index_col=0)

def parse_homer_output_systems(homer_output_systems_filename):
    """
    Parse the table of optimal systems. Generated by clicking 'Export' in
    the Results->Tabular on the lower table.
    """
    homer_output_systems=pd.read_csv(homer_output_systems_filename,header=1)

    #homer_output_systems_idx=list(range(homer_output_systems.shape[0]))
    #homer_output_systems_idx[0]='optimal'
    #homer_output_systems=pd.DataFrame(homer_output_systems.values,index=['optimal'],columns=homer_output_systems.columns)

    return homer_output_systems

def get_solar_capacity_from_homer_output_systems(homer_output_systems):
    """
    Get total rated capacity of solar PV in optimal system from system
    results table exported from HOMER. Assumes name of PV components
    starts with 'PV'. Return the capacity in kW as float.
    """
    solar_capacity_kw=homer_output_systems[homer_output_systems.columns[homer_output_systems.columns.str.match(r'Architecture/PV*')]].sum(axis=1)
    return solar_capacity_kw

def __get_wind_capacity_from_homer_output_systems(homer_output_systems):
    """
    Get total rated capacity of wind turbines in optimal system from system
    results table exported from HOMER. Assumes name of wind components
    starts with 'Wind'. Return the capacity in kW as float.
    """
    wind_capacity_kw=homer_output_systems[homer_output_systems.columns[homer_output_systems.columns.str.match(r'Architecture/Wind*')]].sum().values[0]
    return wind_capacity_kw

def __get_battery_capacity_from_homer_output_systems(homer_output_systems):
    """
    Get the total battery capacity from the homer output systems file
    for each result and return this as an array.
    """
    battery_capacity_kwh=homer_output_systems[homer_output_systems.columns[homer_output_systems.columns.str.match(r'Architecture/Battery*')]].sum().values[0]
    import math
    if math.isnan(battery_capacity_kwh):
        battery_capacity_kwh=0.
    return battery_capacity_kwh

def __get_solar_output_from_homer_output(homer_out):
    """
    Get the solar production in kWh from the homer output file
    and return it as a float value.
    """
    return homer_out[homer_out.columns.levels[0][homer_out.columns.levels[0].str.match(r'PV[ a-zA-Z0-9]* Power Output')]].sum()[0]/2

def get_solar_output_from_homer_output_systems(homer_output_systems):
    """
    Get the solar production in kWh from the HOMER output systems file
    for each system recorded there and return it as an array.
    """
    a=homer_output_systems

    output_clm=np.logical_and(a.columns.str.contains('PV') ,a.columns.str.contains('Production'))
    output_idx=a.columns[output_clm]
    ret=a[output_idx].sum(axis=1)
    print(ret)
    return ret

def get_wind_output_from_homer_output_systems(homer_output_systems):

    a=homer_output_systems

    output_clm=np.logical_and(a.columns.str.contains('Wind') ,a.columns.str.contains('Production'))
    output_idx=a.columns[output_clm]
    return a[output_idx].sum(axis=1)

def __get_wind_output_from_homer_output(homer_out):
    """
    Get the wind production from the homer output file and return it as
    a float.
    """
    hocl=homer_out.columns.levels[0].str.match(r'Wind[ a-zA-Z0-9]* Power Output')
    if pd.Series(hocl).any():
        rval=homer_out[homer_out.columns.levels[0][hocl]].sum()[0]/2
    else:
        rval=None
    return rval

def get_battery_type_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('Batt')
    output_idx=a.columns[output_clm]

    if not a[output_idx].empty:
        if a[output_idx].columns.str.contains('LA').any():
            rvalue='LA'
        else:# a[output_idx].columns.str.contains('LI').any():
            rvalue='LI'
    else:
        rvalue=None

    return rvalue

def get_battery_capacity_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=np.logical_and(a.columns.str.contains('Batt'),a.columns.str.contains('Quantity'))
    output_idx=a.columns[output_clm]
    return a[output_idx].sum(axis=1)

def get_coe_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('COE')
    output_idx=a.columns[output_clm]
    return a[output_idx]

def get_npc_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('NPC')
    output_idx=a.columns[output_clm]
    return a[output_idx]

def get_ic_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('Initial capital')
    output_idx=a.columns[output_clm]
    return a[output_idx]

def get_rf_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('Ren Frac')
    output_idx=a.columns[output_clm]
    return a[output_idx]

def get_load_from_homer_output_systems(homer_output_systems):
    """
    This is a fudge... uses max value in grid energy purchased column...
    may not be valid if e.g. battery charging not efficient.
    """

    a=homer_output_systems

    output_clm=np.logical_and(a.columns.str.contains('Energy'),a.columns.str.contains('Purchased'))
    output_idx=a.columns[output_clm]
    return a[output_idx].max().values

def get_grid_purchases_from_homer_output_systems(homer_output_systems):
    a=homer_output_systems

    output_clm=a.columns.str.contains('Energy Purchased')
    output_idx=a.columns[output_clm]
    return a[output_idx]

def get_max_load_from_homer_output(homer_output):
    return (homer_output['Deferrable Load Served']+homer_output['AC Primary Load']).max(),(homer_output['Deferrable Load Served']+homer_output['AC Primary Load']-homer_output['PV NG Base Power Output']).max()

# FIT values based on Jan 2017 tariffs
fit_export_jan_2017=0.0491

fit_wind_0_50_jan_2017=0.0826
fit_wind_50_100_jan_2017=0.0743
fit_wind_100_1500_jan_2017=0.0481

fit_solar_0_10_jan_2017=0.0411
fit_solar_10_50_jan_2017=0.0432

fit_hydro_0_100_jan_2017=0.0763

def calculate_lcoe_with_fit(
    current_lcoe,
    current_npc,
    annual_solar_generation,
    annual_wind_generation,
    wind_generator_capacity,
    total_load_served,
    discount_rate=3.5,
    project_lifetime=25,
):
    """
    Calculate the real LCOE including the income from the UK feed in tariff.
    Exported energy is deemed as is standard practice rather than calculated
    on the basis of the actual grid exports.
    """
    def USPWF(d,n):
        return ( (1+d)**n - 1 ) / ( d*((d+1)**n) )

    def CRF(d,n):
        return 1/USPWF(d,n)

    # Solar FIT
    solar_fit_generation=fit_solar_0_10_jan_2017
    solar_fit_revenue=annual_solar_generation*(solar_fit_generation+fit_export_jan_2017/2.)

    # Wind FIT
    if wind_generator_capacity<=50:
        wind_fit_generation=fit_wind_0_50_jan_2017
    elif wind_generator_capacity<=100:
        wind_fit_generation=fit_wind_50_100_jan_2017
    else:
        wind_fit_generation=fit_wind_100_1500_jan_2017

    wind_fit_revenue=annual_wind_generation*(wind_fit_generation+fit_export_jan_2017/2.)

    total_annual_fit_revenue=solar_fit_revenue+wind_fit_revenue

    d=discount_rate
    n=project_lifetime

    # FIT payments happen for 20 years which may not be same as project
    # lifetime. Need to evaluate PW of FIT payments then annualise.
    fit_npv=USPWF(d,20)*total_annual_fit_revenue

    # New LCOE = Old LCOE - (Annualised Tariff Income, £)/(Annual Total Elec Served, kWh)
    new_lcoe = current_lcoe - (fit_npv*CRF(d,n))/total_load_served

    new_npc=current_npc+fit_npv

    return new_lcoe[0],new_npc[0]

def rank_homer_results(
    homer_output_systems_filename,
    wind_generator_capacity,
):

    """
    Read in the homer output systems and rank them by different parameters.

    Primarily this recalculates the LCOE including the FIT information, but also
    helps in locating the optimal systems in the output.
    """

    wgc=wind_generator_capacity

    homer_output_systems=parse_homer_output_systems(homer_output_systems_filename)

    pv_out=get_solar_output_from_homer_output_systems(homer_output_systems)


    pv_out=pv_out.fillna(0.)

    wind_out=get_wind_output_from_homer_output_systems(homer_output_systems)
    wind_out=wind_out.fillna(0.)

    coe_out=get_coe_from_homer_output_systems(homer_output_systems)

    npc_out=get_npc_from_homer_output_systems(homer_output_systems)

    load=get_load_from_homer_output_systems(homer_output_systems)

    grid_purchases=get_grid_purchases_from_homer_output_systems(homer_output_systems)

    # Calculate an adjusted LCOE incorporating FIT payments.

    adj_lcoe=[]
    adj_npv=[]

    for coe,npc,pv,wind in zip(coe_out.values,npc_out.values,pv_out.values,wind_out.values):
        lcoe,npv=calculate_lcoe_with_fit(coe,npc,pv,wind,wgc,load)
        adj_lcoe.append(lcoe)
        adj_npv.append(npv)

    homer_output_systems['Adj. LCOE (€/kWh)']=adj_lcoe
    homer_output_systems['Adj. NPC (€)']=adj_npv
    homer_output_systems['Total Electrical Load (kWh)']=load[0]

    # Calculate carbon emissions using average grid intensities for 2015.
    battery_capacity=get_battery_capacity_from_homer_output_systems(homer_output_systems)
    battery_capacity=battery_capacity.fillna(0)
    batt_type=get_battery_type_from_homer_output_systems(homer_output_systems)

    emissions=[]
    for grid,pv,wind,batt in zip(grid_purchases.values,pv_out.values,wind_out.values,battery_capacity.values):

        e=calculate_simple_emissions_from_homer_output(
            grid,
            pv,
            wind,
            batt,
            batt_type
        )
        emissions.append(e[0])

    homer_output_systems["Carbon Emissions (tCO2e/y)"]=emissions

    lcoe_sys=homer_output_systems.sort('Adj. LCOE (€/kWh)')#.head(5)

    return homer_output_systems

"""
Functions for calculating the life-cycle analysis basis embedded carbon
emissions of the renewable energy system.
"""

wind_lca_emissions_g_kwh=10.
solar_lca_emissions_g_kwh=45.
la_battery_lca_emissions_kg_kwh=89.
li_battery_lca_emissions_kg_kwh=143.

def __calculate_res_emissions(
    pv_production,
    wind_production,
    battery_capacity,
    battery_type,
    project_lifetime=25,
):
    """
    Calculate emobodied/lifecycle carbon emissions for renewable energy
    systems in model.
    """
    pv_carbon=pv_production*solar_lca_emissions_g_kwh/1e6
    wind_carbon=wind_production*wind_lca_emissions_g_kwh/1e6

    battery_replacements=-1
    battery_carbon=0.
    if battery_type is "LA":
        battery_carbon=battery_capacity*la_battery_lca_emissions_kg_kwh/1e3
        battery_replacements=2
    elif battery_type is "LI":
        battery_carbon=battery_capacity*li_battery_lca_emissions_kg_kwh/1e3
        battery_replacements=1
    else:
        battery_carbon=0.

    battery_carbon_annual=(battery_replacements+1)*battery_carbon/project_lifetime

    return pv_carbon+wind_carbon+battery_carbon_annual

def calculate_simple_emissions_from_homer_output(
    grid_purchases,
    pv_production,
    wind_production,
    battery_capacity,
    battery_type,
    grid_intensity=367.,
    project_lifetime=25,
):

    """
    This calculates a simple annual emissions figure using the average
    grid carbon intensity.
    """

    res_emissions=__calculate_res_emissions(pv_production,wind_production,battery_capacity,battery_type,project_lifetime=25,)

    grid_emissions=grid_purchases*grid_intensity/1e6

    return grid_emissions+res_emissions

def calculate_emissions_from_homer_output(
    homer_output,
    grid_intensity,
    pv_production,
    wind_production,
    battery_capacity,
    battery_type,
    project_lifetime=25,
):
    """
    This calculates an annual emissions figure using an the 30 minute grid
    carbon intensity data.
    """

    res_emissions=__calculate_res_emissions(pv_production,wind_production,battery_capacity,battery_type)

    grid_purchases_kw=homer_output['Grid Purchases']
    grid_purchases_kwh=grid_purchases_kw/2.

    grid_emissions=np.dot(grid_purchases_kwh.values.flatten(),grid_intensity.values)/1e6

    return grid_emissions+res_emissions

"""
Plotting functions. Where plot is used in thesis report it is referenced here.
"""

def plot_intensity(intensity_file):
    import plotly.offline as py
    import plotly.graph_objs as go

    import numpy as np

    gi=intensity_file.values
    gi=gi.reshape((365,48))

    time_index=pd.date_range('00:00:00','23:30:00',freq='30Min')
    time_index=['%.2i:%.2i'%(i.hour,i.minute) for i in time_index]

    data = [

        go.Heatmap(
            z=gi,
            x=time_index,
        ),

    ]
    plot_url = py.plot(data, filename='intensity_heat_map')
