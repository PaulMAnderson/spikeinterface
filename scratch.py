import sys
import os

from matplotlib import pyplot as plt

# os.chdir('/home/paul/Documents/spikeinterface/src')
# sys.path.append('.')  # Add current directory to path
import spikeinterface.full as si

rec_path = '/mnt/g/To Process/PMA97/PMA97 2025-03-13 Session 1/PMA97 2025-03-13_12-21-42 Opto Config 2'

stream_names, stream_ids = si.get_neo_streams('openephysbinary', rec_path)

npix_rec = si.read_openephys(rec_path, stream_name=stream_names[1], load_sync_timestamps=True)
npix_chan_ids = npix_rec.get_channel_ids()
daq_rec = si.read_openephys(rec_path, stream_name=stream_names[0], load_sync_timestamps=True)
daq_chan_ids = daq_rec.get_channel_ids()

events     = si.read_openephys_event(rec_path,load_sync_timestamps=True)
# raw_events = si.read_openephys_event(rec_path,load_sync_timestamps=False)

event_channels = events.channel_ids
# Find the event channel that contains the digital input line 
for channel in event_channels:
    if 'PXIe-6341Digital Input Line' in channel:
        daq_channel = channel

daq_events     = events.get_events(daq_channel)
# daq_raw_events = raw_events.get_events(daq_channel)
laser_events = daq_events[daq_events['label'] == '8']
laser_times = laser_events['time']

pre = 0.001
post = 0.0075

npix_start_sample = npix_rec.time_to_sample_index(laser_times[0] - pre)
npix_end_sample = npix_rec.time_to_sample_index(laser_times[0] + post)
npix_trace = npix_rec.get_traces(channel_ids=[npix_chan_ids[0]], 
            start_frame=npix_start_sample, end_frame=npix_end_sample,return_scaled=True)
npix_times = npix_rec.get_times()
npix_t = npix_times[npix_start_sample:npix_end_sample]

daq_start_sample = daq_rec.time_to_sample_index(laser_times[0] - pre)
daq_end_sample = daq_rec.time_to_sample_index(laser_times[0] + post)
daq_trace = daq_rec.get_traces(channel_ids=[daq_chan_ids[7]],
            start_frame=daq_start_sample, end_frame=daq_end_sample, return_scaled=True)
daq_times = daq_rec.get_times()
daq_t = daq_times[daq_start_sample:daq_end_sample]


plt.figure()
plt.plot(daq_t,daq_trace*5000)
plt.plot(npix_t, npix_trace)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (uV)')


## Let's look at the average artifact from different length pulses
# import pandas as pd
# laser_df = pd.DataFrame(laser_events)
import numpy as np
import pandas as pd

# Round the durations to account for tiny variations in timing
laser_events['duration'] = np.round(laser_events['duration'], 4)
unique_durations = np.unique(laser_events['duration'])

# Create a figure to show the results
plt.figure(figsize=(15, 10))

# Store the average traces for each duration
avg_traces_by_duration = {}

# Loop through each unique pulse duration
for i, duration in enumerate(unique_durations):

    duration_ms = duration * 1000
    
    # Get the times of laser events with this duration
    pulse_times = laser_events[laser_events['duration'] == duration]['time']

    # Skip if there aren't enough pulses
    if len(pulse_times) < 3:
        print(f"Skipping duration {duration}s - not enough events ({len(pulse_times)})")
        continue

    print(f"Processing {len(pulse_times)} pulses with duration {duration}s")

    # Collect traces for all channels for each pulse time
    all_pulse_traces = []

    pre = 0.001
    post = duration + 0.01

    pre_samples  = pre * npix_rec.get_sampling_frequency()
    post_samples = post * npix_rec.get_sampling_frequency()


    for pulse_time in pulse_times:
        start_sample = npix_rec.time_to_sample_index(pulse_time) - pre_samples
        end_sample = npix_rec.time_to_sample_index(pulse_time) + post_samples

        # Get traces across all channels for this pulse
        pulse_traces = npix_rec.get_traces(start_frame=start_sample,
                                          end_frame=end_sample,
                                          return_scaled=True)

        all_pulse_traces.append(pulse_traces)

    # Average traces across all pulses of the same duration
    avg_traces = np.mean(all_pulse_traces, axis=0)
    avg_traces_by_duration[duration] = avg_traces

    # Create time vector for this segment
    trace_duration_ms = (duration + 0.011) * 1000
    t = np.linspace(-1, trace_duration_ms-1, avg_traces.shape[0])

    # Plot the average trace for a few selected channels (first, middle, last)
    plt.subplot(len(unique_durations), 1, i+1)

    channels_to_plot = [0, len(npix_chan_ids)//4, len(npix_chan_ids)//2,
                        len(npix_chan_ids)*3//4 -1]  # first, middle, last channels
    for ch_idx in channels_to_plot:
        plt.plot(t, avg_traces[:,ch_idx], label=f'Chan {npix_chan_ids[ch_idx]}')

    plt.axvline(x=0, color='r', linestyle='--', label='Laser onset')
    plt.axvline(x=duration_ms, color='g', linestyle='--', label='Laser offset')
    plt.title(f'Duration: {duration*1000:.1f} ms')
    plt.ylabel('Amplitude (uV)')
    # plt.legend()

plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

# Save the average traces by duration (optional)
# np.savez('avg_artifacts_by_duration.npz', **avg_traces_by_duration)
# Round the durations to account for tiny variations in timing
laser_events['duration'] = np.round(laser_events['duration'], 4)
unique_durations = np.unique(laser_events['duration'])

