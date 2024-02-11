import pickle
import numpy as np
import os
import scipy.signal
import pandas as pd



FS = 160

def _rms(x, win):
    output = np.zeros((x.shape))
    npad = np.floor(win / 2).astype(int)
    win = int(win)
    x_ = np.pad(x, ((npad, npad), (0, 0)), 'symmetric')
    for i in range(output.shape[0]):
        output[i, :] = np.sqrt(np.sum(x_[i:i + win, :] ** 2, axis=0) / win)
    return output

def rms(x, fs=FS):
    win = 0.2 * fs
    
    return _rms(x, win)

def lpf(x, f=1., fs=FS):
    f = f / (FS / 2)
    x = np.abs(x)
   
    b, a = scipy.signal.butter(4, f, 'low')
    output = scipy.signal.filtfilt(b, a, x, axis=0, padtype='odd', padlen=3 * (max(len(b), len(a)) - 1))
    return output


def preprocess_data(emg, plot_label):

    emg = rms(emg)
    emg_min = emg.min()
    emg_max = emg.max()
    emg = (emg - emg_min) / (emg_max - emg_min)
    emg = lpf(emg)
  

  
    return emg


def prep():
    directory = "/content/drive/MyDrive/project/annotation/ActionNet Annotations"
                       
    person = []
    label_list = []
    myo_left_readings_list = []
    myo_left_timestamps_list = []
    myo_right_readings_list = []
    myo_right_timestamps_list = []
    start_list = []
    stop_list = []
    
    for parsed_emg_filename in [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]:
        filepath = os.path.join(directory, parsed_emg_filename)
       
        print('filepath:',filepath)
        data = pd.read_pickle(filepath)
        

        pkl_dict = dict()

        for index, row in data.iterrows():

                emg_left_data = row['myo_left_readings']
                emg_right_data = row['myo_right_readings']

                try:
                  myo_left_readings_list.append(preprocess_data(emg_left_data, "left"))
                  myo_right_readings_list.append( preprocess_data(emg_right_data, "right"))
                  person.append(filepath.split('/')[-1].split('.')[0])
                  label_list.append(row["description"])
                
                  myo_left_timestamps_list .append(row["myo_left_timestamps"])                  
                  myo_right_timestamps_list .append(row["myo_right_timestamps"])
                  start_list.append( row["start"])
                  stop_list.append(row["stop"])
                  
                except:
                  # emg_left_preproc=[]
                  # emg_right_preproc = []
                  # print('exp occured')
                  pass




            # SAVE dictionary in pickle file
           
     
    # pickle.dump(pkl_dict, open( "preprocess_data.pkl", "wb"))
    df = pd.DataFrame(list(zip( person,start_list, stop_list, label_list,myo_left_timestamps_list ,myo_left_readings_list ,myo_right_timestamps_list   , myo_right_readings_list  ,)),
               columns =['person','start', 'stop', 'label','myo_left_timestamps'  , 'myo_left_readings' ,'myo_right_timestamps'   ,'myo_right_readings'   , ])
    
    return(df )

# if __name__ == "__main__":
#     main()