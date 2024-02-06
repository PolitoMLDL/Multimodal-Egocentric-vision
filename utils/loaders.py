import glob
from abc import ABC
import pandas as pd
from .epic_record import EpicVideoRecord
import torch.utils.data as data
from PIL import Image
import os
import os.path
from utils.logger import logger
import numpy as np 
import pandas as pd 
import time 
import json
from colorama import Fore, Back, Style
import datetime
from numpy import random

class EpicKitchensDataset(data.Dataset, ABC):
    def __init__(self, split, modalities, mode, dataset_conf, num_frames_per_clip, num_clips, dense_sampling,
                 transform=None, load_feat=False,previous_run_pth=None, run_name='a',additional_info=False, **kwargs):
        """
        split: str (D1, D2 or D3)
        modalities: list(str, str, ...)
        mode: str (train, test/val)
        dataset_conf must contain the following:
            - annotations_path: str
            - stride: int
        dataset_conf[modality] for the modalities used must contain:
            - data_path: str
            - tmpl: str
            - features_name: str (in case you are loading features for a predefined modality)
            - (Event only) rgb4e: int
        num_frames_per_clip: dict(modality: int)
        num_clips: int
        dense_sampling: dict(modality: bool)
        additional_info: bool, set to True if you want to receive also the uid and the video name from the get function
            notice, this may be useful to do some proper visualizations!
        """
        ##########
        self.run_name = run_name
        self.s_name = split
        self.print_val = True
        self.modalities = modalities  # considered modalities (ex. [RGB, Flow, Spec, Event])
        self.mode = mode  # 'train', 'val' or 'test'
        self.dataset_conf = dataset_conf
        self.num_frames_per_clip = num_frames_per_clip
        self.dense_sampling = dense_sampling
        self.num_clips = num_clips
        self.stride = self.dataset_conf.stride
        self.additional_info = additional_info

        if self.mode == "train":
            pickle_name = split + "_train.pkl"
        elif kwargs.get('save', None) is not None:
            pickle_name = split + "_" + kwargs["save"] + ".pkl"
        else:
            pickle_name = split + "_test.pkl"
       
        self.list_file = pd.read_pickle(os.path.join(self.dataset_conf.annotations_path, pickle_name))
        logger.info(f"Dataloader for {split}-{self.mode} with {len(self.list_file)} samples generated")
        self.video_list = [EpicVideoRecord(tup, self.dataset_conf) for tup in self.list_file.iterrows()]
 
               
                       
        self.transform = transform  # pipeline of transforms
        
       
        self.load_feat = load_feat
        my_info ={}
        if self.load_feat:
            self.model_features = None

            for m in self.modalities:
                # load features for each modality
                
                s2_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_02_feature_load.txt'.format(self.run_name,self.s_name,self.mode)
                
                
                model_features = pd.DataFrame(pd.read_pickle(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name))['features'])[["uid", "features_" + m]]

                with open(s2_name, 'w') as fp:
                       fp.write(' pickle_name : {}'.format( pickle_name))
                       fp.write('\n self.dataset_conf[m]: {}'.format( self.dataset_conf[m].features_name)   )  
                       fp.write('\n  model_features.shape: {}'.format(  model_features.shape)   )            
                          
                       fp.write('\n picke_path: {}'.format(os.path.join("saved_features",
                                                                          self.dataset_conf[m].features_name + "_" +
                                                                          pickle_name)))    
                                                       
                # path = '/content/drive/MyDrive/project/mldl23-ego/saved_features/D1_uniform_D1_test.pkl'
                # model_features = pd.DataFrame(pd.read_pickle (path) ['features'])[["uid", "features_" + m]]
               
                     

                if self.model_features is None:
                     self.model_features = model_features
                    
                else:
                    self.model_features = pd.merge(self.model_features, model_features, how="inner", on="uid")
                    
                    

            self.model_features = pd.merge(self.model_features, self.list_file, how="inner", on="uid")
            
            


        s2_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_3initial.txt'.format(self.run_name,self.s_name,self.mode)
        with open(s2_name, 'w') as fp:
                          
            fp.write('self.video_list[4]uid: {}\n'.format(self.video_list[4].uid)) 
            for item in (self.video_list):
              
                 fp.write('label: {}  untrimm: {}  id: {} \n'.format(item.label,item.untrimmed_video_name,item.uid)) 
        # previous_run_path = '/content/drive/MyDrive/project/mldl23-ego/saved_features/D1_alaki_D1_train.pkl'
        if  previous_run_pth:
          previous_run_path = previous_run_pth
          if   os.path.isfile(previous_run_path):

            previous_run = pd.read_pickle(previous_run_path)

            uids  = [previous_run['features'][i]['uid'] for i in range(len(previous_run['features']))]   
            Z_cechk = [self.video_list[i].uid for i  in range(len(self.video_list))  ]
            with open(s2_name, 'a') as fp:
                            
                fp.write('len list  {} - len set {}%%%%%%%%%%%%%%%%%%%\n'.format(len(self.video_list),len(set(Z_cechk))))    
            for it in range(len(self.video_list)-1,0,-1):
                if self.video_list[it].uid in uids:
                    with open(s2_name, 'a') as fp:
                            
                      fp.write('\n index: {}  uid {} has been rempved in {}'.format(it, self.video_list[it].uid,(datetime.datetime.now() ))) 

                    self.video_list.pop(it)
                 

 




        
        # if self.print_val:
        #     s_name  = '/content/drive/MyDrive/project/mldl23-ego/saved_features/{}_{}_video_info.csv'.format(self.s_name,mode)
        
        #     dataset_conf = []
        #     end_frame = []
        #     kitchen =  []
        #     label = []
        #     num_frames = []
        #     recording = []
        #     segment_name= []
        #     start_frame = []
        #     uid = []
        #     untrimmed_video_name = []
        #     num_frames = []           
       
        #     my_dict = {}
        #     tem = []
        #     min_req_len = self.num_frames_per_clip.RGB *self.num_clips
        #     not_enough = []
          

          
        #     for i in range (len(self.video_list) ):
        #       dataset_conf.append(self.video_list[i].num_frames['RGB'])
        #       end_frame.append(self.video_list[i].end_frame)
        #       kitchen.append(self.video_list[i].kitchen)
        #       label.append(self.video_list[i].label)
              
        #       recording.append(self.video_list[i].recording)
        #       segment_name.append(self.video_list[i].segment_name)
        #       start_frame.append(self.video_list[i].start_frame)
        #       uid.append(self.video_list[i].uid)
        #       untrimmed_video_name.append(self.video_list[i].untrimmed_video_name)
        #       num_frames.append(self.video_list[i].num_frames['RGB'])
        #     df = pd.DataFrame(list(zip(untrimmed_video_name, num_frames,uid,start_frame,end_frame,segment_name,recording,label,kitchen,dataset_conf)),
        #        columns =['untrimmed_video_name', 'num_frames', 'uid', 'start_frame','end_frame', 
        #        'segment_name','recording','label','kitchen','dataset_conf'])
        #     df.to_csv(s_name, index=False)

            
            # my_dict['len' +str(i)] = self.video_list[i].num_frames['RGB']
            # tem.append(int(self.video_list[i].num_frames['RGB']))
          # tem3  = dir(self.video_list[0])
          # for j in range (len(tem3)):
          #   my_dict[j] = tem3[j]
          #   if int(self.video_list[i].num_frames['RGB'])< min_req_len:
          #     not_enough.append(i)
          # not_enough.sort(reverse = True)    
          # for it in not_enough:
          #   my_dict['pop'+str(it)] = self.video_list[it].num_frames
          #   self.video_list.pop(it)         
          # with open(s_name, 'w') as f:
          #      for key, value in  my_dict.items(): 
          #            f.write('%s:%s\n' % (key, value))
    
    def _get_train_indices(self, record, modality='RGB'):
        
        start = int(record.start_frame)
        end = (record.end_frame)        
        length_record = end-start+1
        length_clip = length_record//self.num_clips
        
        # centres_of_clips = np.linspace(start+length_clip//2,end- length_clip//2, num=self.num_clips, dtype=int)
        candidates_centre = [i for i in range(start+self.num_frames_per_clip.RGB//2+1,end- self.num_frames_per_clip.RGB//2-1)]
        centres_of_clips = random.choice(candidates_centre, size=(self.num_clips))
        centres_of_clips.sort()

        if length_clip//self.stride <self.num_frames_per_clip.RGB:
          stride = 1
        else:
          stride = self.stride
        if self.dense_sampling['RGB']:          
          num_clip_frame  = 1
          selected_ind = [0]
          tem = 1
          while num_clip_frame   <self.num_frames_per_clip.RGB:
            new_ind =  selected_ind[0]-tem*stride
            new_ind2 = selected_ind[0]+tem*stride           
            
            tem +=1
            if   new_ind+ min(centres_of_clips)>=start   :
               selected_ind.append(new_ind)
               num_clip_frame+=1
            if new_ind2 + max(centres_of_clips) <= end and num_clip_frame<self.num_frames_per_clip.RGB  :
               selected_ind.append(new_ind2)
               num_clip_frame+=1
            else:
              break
          output = []
          out_without_sub= []
          selected_ind.sort()
          wrong_index =[]
          for cent in centres_of_clips:
            for ind in  selected_ind:
              output.append(cent+ind-start)
              out_without_sub.append(cent+ind)
              if out_without_sub[-1]>end or out_without_sub[-1]< start:
                wrong_index.append(out_without_sub[-1])
          
 
          my_info ={}
          my_info['info'] = '\n start: {},end: {}, strid: {} ,num_frames_per_clip.RGB:{}'.format(start,end, stride,self.num_frames_per_clip.RGB)
          my_info['wrong_indices'] = wrong_index
          my_info['centres_of_clips'] =centres_of_clips             
          my_info ['len out'] =len(output)  
          my_info['output']= output 
          my_info['selected indices'] =selected_ind          
          my_info['out_without_subtra'] = out_without_sub          
          s_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_4_selected_index.txt'.format(self.run_name,self.s_name,self.mode)
         
          with open(s_name, 'a') as f:
            
              for key, value in  my_info.items(): 
                    f.write('%s:%s\n' % (key, value))
              f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')     

          
          return(output)



      # uniform sampling :
     
        else:
          output = []
          out_without_sub =[]
          for cent in centres_of_clips:
             
             tem = np.linspace(cent-self.num_frames_per_clip.RGB//2,cent+self.num_frames_per_clip.RGB//2, num=self.num_frames_per_clip.RGB, dtype=int)
             for t in tem:
              output.append(t-start)  
              out_without_sub.append(t)    
     
          my_info ={}
          my_info['info'] = ' start: {},end: {}, strid: {} ,num_frames_per_clip.RGB:{}'.format(start,end, stride,self.num_frames_per_clip.RGB)
          
          my_info['centres_of_clips'] =centres_of_clips             
          my_info ['len out'] =len(output)  
          my_info['output']=output 
                  
          my_info['out_without_subtra'] = out_without_sub          
          s_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_4_selected_index.txt'.format(self.run_name,self.s_name,self.mode)


         
          with open(s_name, 'a') as f:
            
              for key, value in  my_info.items(): 
                    f.write('%s:%s\n' % (key, value))
              f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
          return  output

    '''
    I add RGB to the folowing line
    '''
    def _get_val_indices(self, record, modality='RGB'):    
        # print(Fore.GREEN +'!!!!!!!!!!!!!!!!_get_val_indices')
        start = int(record.start_frame)
        end = (record.end_frame)        
        length_record = end-start+1
        length_clip = length_record//self.num_clips
        
        # centres_of_clips = np.linspace(start+length_clip//2,end- length_clip//2, num=self.num_clips, dtype=int)
        candidates_centre = [i for i in range(start+self.num_frames_per_clip.RGB//2+1,end- self.num_frames_per_clip.RGB//2-1)]
        centres_of_clips = random.choice(candidates_centre, size=(self.num_clips))
        centres_of_clips.sort()
        if length_clip//self.stride <self.num_frames_per_clip.RGB:
          stride = 1
        else:
          stride = self.stride
        if self.dense_sampling['RGB']:          
          num_clip_frame  = 1
          selected_ind = [0]
          tem = 1
          while num_clip_frame   <self.num_frames_per_clip.RGB:
            new_ind = selected_ind[0]-tem*stride
            new_ind2 = selected_ind[0]+tem*stride           
            
            tem +=1
            if   new_ind+min(centres_of_clips)>=start   :
               selected_ind.append(new_ind)
               num_clip_frame+=1
            if new_ind2 +max(centres_of_clips) <= end and num_clip_frame<self.num_frames_per_clip.RGB  :
               selected_ind.append(new_ind2)
               num_clip_frame+=1
            else:
              break
          output = []
          out_without_sub= []
          selected_ind.sort()
          wrong_index =[]
          for cent in centres_of_clips:
            for ind in  selected_ind:
              output.append(cent+ind-start)
              out_without_sub.append(cent+ind)
              if out_without_sub[-1]>end or out_without_sub[-1]< start:
                wrong_index.append(out_without_sub[-1])
          
 
          my_info ={}
          my_info['info'] = '\n start: {},end: {}, strid: {} ,num_frames_per_clip.RGB:{}'.format(start,end, stride,self.num_frames_per_clip.RGB)
          my_info['wrong_indices'] = wrong_index       
                             
          my_info ['len out'] =len(output)  
          my_info['output']=output 
          my_info['selected indices'] =selected_ind          
          my_info['out_without_subtra'] = out_without_sub          
          s_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_4_selected_index.txt'.format(self.run_name,self.s_name,self.mode)
         
          with open(s_name, 'a') as f:
            
              for key, value in  my_info.items(): 
                    f.write('%s:%s\n' % (key, value))
              f.write('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


        
          return(output)


      # uniform sampling :
     
        else:
          output = []
          out_without_sub =[]
          for cent in centres_of_clips:
             
             tem = np.linspace(cent-self.num_frames_per_clip.RGB//2,cent+self.num_frames_per_clip.RGB//2, num=self.num_frames_per_clip.RGB, dtype=int)
             for t in tem:
              output.append(t-start)  
              out_without_sub.append(t)    
          my_info ={}
          my_info['info'] = ' start: {},end: {}, strid: {} ,num_frames_per_clip.RGB:{}'.format(start,end, stride,self.num_frames_per_clip.RGB)
          
          my_info['centres_of_clips'] =centres_of_clips             
          my_info ['len out'] =len(output)  
          my_info['output']=output 
                
          my_info['out_without_subtra'] = out_without_sub          
          s_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_4_selected_index.txt'.format(self.run_name,self.s_name,self.mode)
          with open(s_name, 'w') as f:
            

              for key, value in  my_info.items(): 
                    f.write('%s:%s\n' % (key, value))
              f.write('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')


          

          return  output


    def __len__(self):
        return len(self.video_list)
    def __getitem__(self, index):
        my_info ={}
        s_name4  =  '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_3_get_item.txt'.format(self.run_name,self.s_name,self.mode)
       
       
       
        # print('start of get item')       
        frames = {}
        label = None
        # record is a row of the pkl file containing one sample/action
        # notice that it is already converted into a EpicVideoRecord object so that here you can access
        # all the properties of the sample easily
        record = self.video_list[index]
        # print('record',record )

        if self.load_feat:
            sample = {}
            # s_name2  =  '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_8_load_feat_checks.txt'.format(self.run_name,self.s_name,self.mode)
            s_name2  =  '/content/{}_{}_{}_8_load_feat_checks.txt'.format(self.run_name,self.s_name,self.mode)
            my_info[' now']=str(datetime.datetime.now() )
            my_info['shape1: '] = self.model_features.shape
            sample_row = self.model_features[self.model_features["uid"] == int(record.uid)]            
            my_info['len_list'] = len(self.model_features["uid"])
          
            my_info['sample_row'] =sample_row
            my_info['shape: '] =self.model_features.shape
            my_info ['record.uid'] =record.uid
            # for m in self.modalities:
            #   my_info ['sample_row["features_" + m].values:   '] = sample_row["features_" + m].values[0].shape
           
            
            
            with open(s_name2, 'a') as f:
              for key, value in  my_info.items(): 
                    f.write('%s:%s\n' % (key, value))
              f.write('\n %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')

            # assert len(sample_row) == 1
            for m in self.modalities:
                print('\n \n \n ')
                print(sample_row)
               
                sample[m] = sample_row["features_" + m].values[0]
      

            if self.additional_info:
                return sample, record.label, record.untrimmed_video_name, record.uid
            else:
 
                  

                return sample, record.label

        segment_indices = {}
        # notice that all indexes are sampled in the[0, sample_{num_frames}] range, then the start_index of the sample
        # is added as an offset
        for modality in self.modalities:
            
            if self.mode == "train":
                # here the training indexes are obtained with some randomization
                segment_indices[modality] = self._get_train_indices(record, modality)
            else:
                # here the testing indexes are obtained with no randomization, i.e., centered
                segment_indices[modality] = self._get_val_indices(record, modality)

        for m in self.modalities:
            
            img, label = self.get(m, record, segment_indices[m])
            frames[m] = img

        if self.additional_info:
            with open(s_name4, 'a') as f:
                   
                  f.write('\n frames shape: \n'.format(len(frames)))
            
            return frames, label, record.untrimmed_video_name, record.uid
        else:
            with open(s_name4, 'a') as f:
                   
                  f.write('\n frames shape: \n '.format(len(frames)))
            return frames, label
        
    def get(self, modality, record, indices):
        # print(Fore.YELLOW,'load data')
        images = list()
        for frame_index in indices:
            p = int(frame_index)
            # here the frame is loaded in memory
            frame = self._load_data(modality, record, p)
            images.extend(frame)
        # finally, all the transformations are applied
        process_data = self.transform[modality](images)
        return process_data, record.label
    
    def _load_data(self, modality, record, idx):
        s_name  = '/content/drive/MyDrive/project/mldl23-ego/other_info/{}_{}_{}_6loadcheck.txt'.format(self.run_name,self.s_name,self.mode)      
        with open(s_name, 'a') as f:   
                    f.write('########################') 
        # data_path = self.dataset_conf[modality].data_path
        data_path = '/content/ek_data/frames'

        
        tmpl = self.dataset_conf[modality].tmpl
        # print(Fore.YELLOW,'load data idx is :{}'.format(idx))
        # print('modality',modality)
        if modality == 'RGB' or modality == 'RGBDiff':
            # here the offset for the starting index of the sample is added
          
            idx_untrimmed = record.start_frame + idx
            # tem = os.path.join(data_path, record.untrimmed_video_name)
            # print('data_path',os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed)))
            t_e =time.time()
            st_info = ''

                    
            try:

                 img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(int(idx_untrimmed)))).convert('RGB')
                 
                         
            except OSError as error :
              st_info = st_info +os.path.join(data_path, record.untrimmed_video_name, tmpl.format(int(idx_untrimmed)))
              st_info = st_info + '  os error hapeend after: {}'.format(time.time()-t_e)
              # with open(s_name, 'a') as f:                          
                    
              #       f.write(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed)))                    
                    
              time.sleep(1)
              try:
                      
                      img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(int(idx_untrimmed)))).convert('RGB')
                     
              except:
                    
                    # try:
                    #   time.sleep(2)
                    #   img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(idx_untrimmed))).convert('RGB')
                      
                      
                          
                         
                    # except:
                    max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
          

                    if idx_untrimmed > max_idx_video:
                       
                        
                        img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(int(max_idx_video)))) \
                        .convert('RGB')
                        st_info = '\n\n\n except has hapeend'
                    else:
                         
                          raise FileNotFoundError
           




             
            except FileNotFoundError:
                
                max_idx_video = int(sorted(glob.glob(os.path.join(data_path,
                                                                  record.untrimmed_video_name,
                                                                  "img_*")))[-1].split("_")[-1].split(".")[0])
                st_info = st_info + ' FileNotFoundError:{}'.format(time.time()-t_e)
          

                if idx_untrimmed > max_idx_video:
                    img = Image.open(os.path.join(data_path, record.untrimmed_video_name, tmpl.format(int(max_idx_video)))) \
                        .convert('RGB')
                    st_info = st_info + ' idx_untrimmed > max_idx_video 2:{}'.format(time.time()-t_e)
                else:
                    raise FileNotFoundError
            with open(s_name, 'a') as f:   
                    f.write(st_info)
                    
                    f.write('######################### \n')
            return [img]
        
        else:
            with open(s_name, 'a') as f:   
                    f.write('not implemented  ')
                    f.write(st_info)
                    f.write('######################### \n')
            raise NotImplementedError("Modality not implemented")

