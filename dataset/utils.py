import glob
import numpy as np
import os
import time
from tqdm import tqdm
from scipy.stats import norm


D2KM = 111.19492664455874
import math



class _Timer:
    """Timer."""

    def __init__(self, name):
        self.name = name
        self.count = 0
        self.mean = 0.0
        self.sum_squares = 0.0
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_traceback):
        elapsed_time = time.time() - self.start_time
        self.update(elapsed_time)
        self.start_time = None

    def update(self, elapsed_time):
        self.count += 1
        delta = elapsed_time - self.mean
        self.mean += delta / self.count
        delta2 = elapsed_time - self.mean
        self.sum_squares += delta * delta2

    def mean_elapsed(self):
        return self.mean

    def std_elapsed(self):
        if self.count > 1:
            variance = self.sum_squares / (self.count - 1)
            return math.sqrt(variance)
        else:
            return 0.0

class Timers:
    """Group of timers."""

    def __init__(self, activate=False):
        self.timers = {}
        self.activate = activate
    def __call__(self, name):
        if not self.activate:return _DummyTimer()
        if name not in self.timers:
            self.timers[name] = _Timer(name)
        return self.timers[name]

    def log(self, names=None, normalizer=1.0):
        """Log a group of timers."""
        assert normalizer > 0.0
        if names is None:
            names = self.timers.keys()
        print("Timer Results:")
        for name in names:
            mean_elapsed = self.timers[name].mean_elapsed() * 1000.0 / normalizer
            std_elapsed = self.timers[name].std_elapsed() * 1000.0 / normalizer
            space_num = " "*name.count('/')
            print(f"{space_num}{name}: {mean_elapsed:.2f}±{std_elapsed:.2f} ms")
class _DummyTimer:
    """A dummy timer that does nothing."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
def resample_trace(trace, sampling_rate):
    if trace.stats.sampling_rate == sampling_rate:
        return
    if trace.stats.sampling_rate % sampling_rate == 0:
        trace.decimate(int(trace.stats.sampling_rate / sampling_rate))
    else:
        trace.resample(sampling_rate)

def gaussian_mixture_fully_vectorized(x, params, eps=1e-6):
    if params.ndim == 2:
        alpha = np.reshape(params[:, 0], (1, -1))
        mu = np.reshape(params[:, 1], (1, -1))
        sigma = np.reshape(np.maximum(params[:, 2], eps), (1, -1))
        x = np.reshape(x, (-1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=1)
    elif params.ndim == 3:
        alpha = np.expand_dims(params[:, :, 0], axis=0)
        mu = np.expand_dims(params[:, :, 1], axis=0)
        sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
        x = np.reshape(x, (-1, 1, 1))
        density = alpha * 1 / (np.sqrt(2 * np.pi) * sigma) * \
            np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        density = np.sum(density, axis=2).T
    else:
        raise ValueError('Params ndim must be 2 or 3')

    return density

def filter_shard(events, shard_id, shards):
    if not shards:
        return events
    if isinstance(events, pd.DataFrame):
        events = events.iloc[shard_id::shards]
    else:
        return events[shard_id::shards]
    return events

def merge_hdf5(inputs, output, event_id, sort_key=None):
    inputs = sorted(glob.glob(inputs))
    print(inputs)
    metadata = {}
    delete_keys = []
    existing_keys = []
    catalog = None
    with h5py.File(output, 'w') as fout:
        gout_data = fout.create_group('data')
        gout_meta = fout.create_group('metadata')
        for inp in tqdm(inputs):
            with h5py.File(inp, 'r') as fin:
                for key in fin['metadata'].keys():
                    if key == 'event_metadata':
                        continue
                    tmp = fin['metadata'][key].value
                    if key in metadata:
                        if isinstance(metadata[key] == tmp, (bool, np.bool_, np.bool)):
                            assert metadata[key] == tmp
                        else:
                            assert all(metadata[key] == tmp)
                    metadata[key] = tmp

                for event in fin['data']:
                    existing_keys += [event]
                    if event in gout_data:
                        delete_keys += [event]
                        del gout_data[event]
                        continue
                    gout_event = gout_data.create_group(event)
                    for ds in fin['data'][event]:
                        gout_event.create_dataset(
                            ds, data=fin['data'][event][ds].value)

            if catalog is None:
                catalog = pd.read_hdf(inp, 'metadata/event_metadata')
            else:
                catalog = pd.concat(
                    [catalog, pd.read_hdf(inp, 'metadata/event_metadata')])

        for key, val in metadata.items():
            gout_meta.create_dataset(key, data=val)

    if sort_key is None:
        sort_key = event_id

    catalog = catalog.sort_values(by=sort_key)
    catalog = catalog[~catalog[event_id].isin(delete_keys)]
    catalog = catalog[catalog[event_id].isin(existing_keys)]
    print(f'Removed {len(delete_keys)} redundant events')
    catalog.to_hdf(output, key='metadata/event_metadata',
                   mode='a', encoding='utf-8', format='table')

def detect_location_keys(columns):
    candidates = [['LAT', 'Latitude(°)', 'Latitude'],
                  ['LON', 'Longitude(°)', 'Longitude'],
                  ['DEPTH', 'JMA_Depth(km)', 'Depth(km)', 'Depth/Km']]

    coord_keys = []
    for keyset in candidates:
        for key in keyset:
            if key in columns:
                coord_keys += [key]
                break

    if len(coord_keys) != len(candidates):
        raise ValueError('Unknown location key format')

    return coord_keys

def wait_for_file(path, sleep_seconds=600, silent=False):
    while not os.path.exists(path):
        if not silent:
            print(
                f'File {path} for weight transfer missing. Sleeping for {sleep_seconds} seconds.')
        time.sleep(sleep_seconds)

def normalize_gain(stream, inv, key='ACC'):
    delete_traces = []
    for trace in stream:
        try:
            resp = inv.get_response(trace.id, trace.stats.starttime)
        except:
            # get_response throws plain Exception, therefore bare except is required
            print(f'Missing response for {trace.id}')
            delete_traces += [trace]
            continue
        try:
            sensitivity = np.abs(
                resp.get_evalresp_response_for_frequencies([1.], key))[0]
            reported_sensitivity = resp.instrument_sensitivity.value
            if np.abs(sensitivity - reported_sensitivity) / max(sensitivity, reported_sensitivity) > 0.05:
                print(f'Sensitivity mismatch for station {trace.stats.network}.{trace.stats.station}. '
                      f'Reported: {reported_sensitivity}\tComputed: {sensitivity}\tFactor: {reported_sensitivity / sensitivity}')
                sensitivity = reported_sensitivity
            trace.data = trace.data / sensitivity
        except ValueError:
            # Illegal RESP format
            trace.data = trace.data * 0.0

    for trace in delete_traces:
        stream.remove(trace)


def gaussian_mixture(x, params, eps=1e-6, memory_save=True, fortran=True):
    if fortran and not np.isfortran(params):
        params = np.array(params, order='F')
    if not memory_save:
        return gaussian_mixture_fully_vectorized(x, params, eps)
    if params.ndim == 2:
        alpha = params[:, 0]
        mu = params[:, 1]
        sigma = np.maximum(params[:, 2], eps)
        density = np.zeros_like(x)
        for i in range(alpha.shape[0]):
            density += alpha[i] * 1 / (np.sqrt(2 * np.pi) * sigma[i]) * \
                                        np.exp(-(x - mu[i]) ** 2 / (2 * sigma[i] ** 2))
    elif params.ndim == 3:
        alpha = np.expand_dims(params[:, :, 0], axis=0)
        mu = np.expand_dims(params[:, :, 1], axis=0)
        sigma = np.expand_dims(np.maximum(params[:, :, 2], eps), axis=0)
        x = np.reshape(x, (-1, 1))
        density = np.zeros((x.shape[0], alpha.shape[1]))
        for i in range(alpha.shape[2]):
            density += alpha[:, :, i] * 1 / (np.sqrt(2 * np.pi) * sigma[:, :, i]) * np.exp(-(
                x - mu[:, :, i]) ** 2 / (2 * sigma[:, :, i] ** 2))
        density = density.T
    else:
        raise ValueError('Params ndim must be 2 or 3')

    return density


def bin_search_quantile(pred, quantile, vmin=0, vmax=10):
    min_val = vmin * np.ones((pred.shape[0], 1))
    max_val = vmax * np.ones((pred.shape[0], 1))

    for _ in range(14):
        # 14 iterations mean approximation error is below 0.001
        mean_val = (min_val + max_val) / 2
        prob = np.sum(
            pred[:, :, 0] *
            (1 - norm.cdf((mean_val - pred[:, :, 1]) / pred[:, :, 2])),
            axis=-1, keepdims=True)
        min_val = np.where(prob > quantile, mean_val, min_val)
        max_val = np.where(prob <= quantile, mean_val, max_val)

    pred = np.squeeze(min_val, axis=-1)

    return pred

def split_array_varying_lengths_np(arr, lengths):
    """
    Split a NumPy array into chunks of varying lengths.
    
    Args:
    arr (numpy.ndarray): The array to split.
    lengths (list): A list of integers, where each integer represents the length of a chunk.
    
    Returns:
    list: A list of NumPy arrays, where each array is a chunk of the original array.
    """
    if np.sum(lengths) != len(arr):
        raise ValueError(f"The sum of lengths must be equal to the length of the input array. Get {len(arr)} length array and totall lenght is {np.sum(lengths)} ")
    indices = np.cumsum(lengths)[:-1]
    return np.split(arr, indices)


def pickout_right_data_depend_on_dataset(data_path, one_data, one_event_metadata, waveform_format=None, event_set_list=None, target_type='all_in_one_source'):
    if 'STEAD' in data_path or (waveform_format is not None and waveform_format =='numpy_with_index'):
        ### source metadata
        df  = one_event_metadata
        grouped_df = df.groupby("labels")
        print(f"there are {len(grouped_df)} events in the data")
        group_size_counts = df['group_size'].value_counts().sort_index()
        # Filter the counts for group sizes from 1 to 9
        filtered_counts = group_size_counts[group_size_counts.index.isin(range(1, 20))]

        # Convert the filtered counts to a pandas DataFrame
        group_size_table = pd.DataFrame(filtered_counts).reset_index()
        group_size_table.columns = ['group_size', 'count']
        print(tabulate.tabulate(group_size_table.transpose(),headers='', tablefmt='psql', showindex=True))

        print(f"===> Notification: the old trail use wrong location information as [lat, lat , deep] rather than [lat, lon, deep]. However, the performance is good. <===")
        source_magnitude = np.array(grouped_df['Magnitude'].apply(lambda x: x.to_numpy().mean()).tolist())
        if target_type == 'all_in_one_source':
            source_latitude  = np.array(grouped_df['Latitude'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_longitude = np.array(grouped_df['Longitude'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_depth_km  = np.array(grouped_df['Depth/Km'].apply(lambda x: x.to_numpy().mean()).tolist())
            source_metadata  = {'location': np.stack([source_latitude, source_longitude, source_depth_km],-1),
                                'magnitude': source_magnitude}
            inputs_metadata = {
                            "receiver_latitude" : one_event_metadata['coords_x'].values,
                            "receiver_longitude": one_event_metadata['coords_y'].values,
                            "receiver_elevation_m": one_event_metadata['coords_z'].values
                        }
        elif target_type == 'all_in_all_delta':
            if np.abs(one_event_metadata['coords_z'].values).max() > 1000:
                print("========== [ will divide coords_z by 1000 since we hope its unit is km rather than meter  ] ==========")
                one_event_metadata['coords_z'] = one_event_metadata['coords_z']/1000
            source_metadata = {'delta_latitude' : df["delta_latitude"].values,
                               'delta_longitude': df["delta_longitude"].values,
                               'delta_deepth'   : df['delta_deepth'].values,# (delta is define as deepth - receiver_elevation_m not deepth + receiver_elevation_m)
                               'magnitude': source_magnitude}
            inputs_metadata = {
                            "receiver_latitude" : one_event_metadata['coords_x'].values,
                            "receiver_longitude": one_event_metadata['coords_y'].values,
                            "receiver_deepth": one_event_metadata['coords_z'].values,
                        }
        elif target_type == 'all_in_all_xyz_delta':
            source_metadata = {'delta_vector_x': df['delta_vector_x'].values,
                               'delta_vector_y': df['delta_vector_y'].values,
                               'delta_vector_z': df['delta_vector_z'].values,# (delta is define as deepth - receiver_z_km )
                               'magnitude': source_magnitude}
            inputs_metadata = {
                    "receiver_vector_x": one_event_metadata['receiver_x_km'].values,
                    "receiver_vector_y": one_event_metadata['receiver_y_km'].values,
                    "receiver_vector_z": one_event_metadata['receiver_z_km'].values,# (receiver_z_km is define as -receiver_elevation_m/1000 )
                }
        else:
            raise NotImplementedError(f"target_type={target_type} is not support")
        inputs_metadata["p_picks"] = one_event_metadata['p_picks'].values
        if 'offset_for_event_recorder' in one_event_metadata:
            inputs_metadata["offset_for_event_recorder"] = one_event_metadata['offset_for_event_recorder'].values
            
        #source_metadata  = np.stack([source_latitude, source_longitude, source_depth_km, source_magnitude], -1)
        #inputs_metadata = one_event_metadata[['coords_x', 'coords_y', 'coords_z', 'p_picks']].values
        # <<---- waveform is 6000 and divide to 3000
        print("========== [ notice, some dataset need let p_pick divide by 2, if the performance not good, check this ] ==========")
        if inputs_metadata['p_picks'].max() > 3000 and 'STEAD' in data_path:
            print("========== [ will divide p_pick by 2, this mean we use sample rate == 50Hz] ==========")
            print("========== [ To provide cross-dataset performance, we disable this option after 20230725] ==========")
            inputs_metadata['p_picks'] = inputs_metadata['p_picks']//2
            raise NotImplementedError(f"should not make p_picks large than 3000") ## The STEAD dataset p_arrive never bigger than 3000
        one_event_metadata['index'] = np.arange(len(one_event_metadata))
        event_set_list = one_event_metadata.groupby('labels')['index'].apply(lambda x: x.to_numpy()).tolist() 

        
        trace_name_list = one_event_metadata['EVENT'].values
        
        if isinstance(one_data, h5py._hl.files.File):
            inputs_waveform = (one_data, trace_name_list)
        else:
            one_data['all_trace_name_list'] = trace_name_list
            inputs_waveform = one_data

    elif 'TEAM' in data_path:
        inputs_waveform = np.concatenate(one_data['waveforms']) #<--format it as numpy
        source_magnitude = one_event_metadata['Magnitude'].values
        source_locations = np.squeeze(one_event_metadata[['Latitude', 'Longitude', 'Depth/Km']].values)

        if target_type == 'all_in_one_source':
            coords = np.concatenate(one_data['coords'])
            # inputs_waveform = np.concatenate(one_data['waveforms'])
            # inputs_metadata = np.concatenate([np.concatenate(one_data['coords']),
            #                                   np.concatenate(one_data['p_picks'])[:, None]], -1)
            # source_metadata = np.concatenate([one_event_metadata[['Latitude', 'Longitude', 'Depth/Km']].values,
            #                                   one_event_metadata['Magnitude'].values[:, None]
            #                                 ], -1)
            source_metadata = {'location': source_locations,'magnitude': source_magnitude}
            
            inputs_metadata = {"receiver_latitude" : coords[...,0],"receiver_longitude": coords[...,1],"receiver_elevation_m": coords[..., 2],
                               "p_picks": np.concatenate(one_data['p_picks']) }
            
        elif target_type == 'all_in_all_delta':
            #raise NotImplementedError
            for data in one_data['coords']:
                data[...,2]*=-1
            coords = np.concatenate(one_data['coords'])
            delta_location  =  [source_location - receiver_location for receiver_location, source_location in zip(one_data['coords'], source_locations)]
            delta_location  = np.concatenate(delta_location)
            
            source_metadata = {'delta_latitude' : delta_location[...,0],
                               'delta_longitude': delta_location[...,1],
                               'delta_deepth'   : delta_location[...,2],
                               'magnitude':       source_magnitude}
            print(f"===> Notification: the old trail use wrong location information as [lat, lat , deep] rather than [lat, lon, deep]. <===")
        
            inputs_metadata = {"receiver_latitude": coords[..., 0],"receiver_longitude": coords[..., 1],
                               "receiver_deepth":   coords[..., 2], # the TEAM data use minus value represent station above ground
                               "p_picks": np.concatenate(one_data['p_picks']),
                               }
        elif target_type == 'all_in_all_xyz_delta':
            raise NotImplementedError(
                f"target_type={target_type} is not support")
        else:
            raise NotImplementedError(f"target_type={target_type} is not support")
        
        
        
        if event_set_list is None:
            event_lengths = [len(t) for t in one_data['coords']]
            event_set_list = split_array_varying_lengths_np(np.arange(len(inputs_waveform)), event_lengths)
    else:

        raise NotImplementedError(f"check your path => {data_path}")
    return inputs_waveform, inputs_metadata, source_metadata, event_set_list

def read_event_data(event_name_sub_list,file_name):
    if len(event_name_sub_list)==0:return []
    with h5py.File(file_name, 'r') as f:
        g_event = np.zeros((len(event_name_sub_list), 3000, 3))
        for i,event_name in enumerate(event_name_sub_list):
            g_event[i] = np.array(f.get(f'data/{event_name}'))[::2,:]
        #g_event = [np.array(f.get(f'data/{event_name}')) for event_name in event_name_sub_list]
    return g_event

def load_wave_data(wavedata_path_list):
        if not isinstance(wavedata_path_list, list):
            return None,h5py.File(wavedata_path_list, 'r')
        data_list = []
        index_list = []
        for data_path in tqdm(wavedata_path_list):
            data = np.load(data_path)
            #tqdm.write(f"loading {data_path}.............", end = ' ')
            # print(f"nan check {np.isnan(data).any()}")
            # print(f"inf check {np.isinf(data).any()}")
            data_list.append(data)
            index_path = data_path.replace(".f16.npy", ".npy").replace(".npy", ".index.npy")
            index_list.append(np.load(index_path, allow_pickle=True))
            #tqdm.write(f"done!", end='\n')
        print("concat~!")
        now = time.time()
        data_list = np.concatenate(data_list) if len(data_list) > 1 else data_list[0]
        index_list = np.concatenate(index_list) if len(index_list) > 1 else index_list[0]
        index_map = dict([[name, i] for i, name in enumerate(index_list)])
        print(f"done, cost {time.time() - now}")

        # print("convert to share tensor")
        # now = time.time() 
        # data_list = torch.from_numpy(data_list).share_memory_()
        # print(f"done, cost {time.time() - now}")
        return index_map, data_list

