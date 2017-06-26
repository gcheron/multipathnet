require 'torch'
require 'fbcoco'
require 'nn'
require 'optim'
require 'xlua'
require 'mattorch'

local tnt = require 'torchnet'


local json = require 'cjson'
local py = require('fb.python')
py.exec('import numpy as np')


-- contrary to annotations (in x,y,w,h format)
-- proposal are in format: y1,x1,y2,x2

-- example from:
-- /sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/VOC2007/selective_search/train.t7
-- #data.boxes = 2501
-- #data.images = 2501
-- data.boxes[1] is [torch.DoubleTensor of size 1264x4]
-- data.images[1] = 000012.jpg


-- prepare dummy annotation file
json=require 'cjson'
data = torch.CharStorage('data/annotations/daly_trainkeyframes.json'):string()
tmp_annotations = json.decode(data)

dofile('daly_res.lua') 

local permute_tensor = torch.LongTensor{2,1,4,3}


-- track format [ X1 Y1 X2 Y2 frameIds ]
sourcedir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/track_info_forLSTM'
maximagesperset=40e3 -- create a new set when this number is reach (note that a same video will be put in the same set anyway)

function saveset(set_boxes,set_images,set_trackid,set_id)
   assert(#set_boxes==#set_images and #set_boxes==#set_trackid)
   local set_name=resprefixpath..set_id..'.t7'
   local annotation_name=annprefixpath..set_id..'.json'
   print('SAVING SET: '..set_name)
   print('Nb images: '..#set_images)
   torch.save(set_name,{boxes=set_boxes,images=set_images,trackid=set_trackid})

   -- create the dummy annotation file
   local annotations = {categories = tmp_annotations.categories,images={},annotations={}}
   for i=1,#set_images do
      local vidname = set_images[i]:match('([^/]*)/')
      local ch,cw = restab[vidname][1],restab[vidname][2]
      annotations.images[i] = {file_name = set_images[i] , id = i , width=cw, height=ch} 
      annotations.annotations[i] ={ignore=0,bbox={10,10,50,50},category_id=1,area=2500,iscrowd=0,id=i,image_id=i}
   end
   local wfile = io.open(annotation_name,'w')
   local jcontent = json.encode(annotations)
   wfile:write(jcontent)
   io.close(wfile)
end

function write_prop_file(split)
   resprefixpath='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/'..split..'tracks_set_'
   annprefixpath='/sequoia/data1/gcheron/code/torch/multipathnet/data/annotations/daly_'..split..'tracks_set_'
   vidlist='/sequoia/data2/gcheron/DALY/OF_vidlist_'..split..'1.txt'
      
   vids=io.open(vidlist,'r')
   local nbvideos=0
   local nbimages=0
   local set_id,set_boxes,set_images,set_trackid = 1,{},{},{}
   
   for vidname_exp in vids:lines() do
      local vidname = vidname_exp:match('[^%s]*')
      nbvideos=nbvideos+1
      print(('Video %03d (%s)'):format(nbvideos,vidname))
      if nbimages>maximagesperset then -- create a new set
         saveset(set_boxes,set_images,set_trackid,set_id)
         set_id=set_id+1
         set_boxes,set_images,set_trackid = {},{},{}
         collectgarbage()
      end
   
      local set_imcorres = {} -- frame idx (cix) correspondance to the cur image set (im_cor): im_cor=set_imcorres[cix]
   
      trackid=0
      tracks=io.popen('ls '..sourcedir..'/'..vidname..'/*.mat')
      for tt in tracks:lines() do
         trackid=trackid+1
         local tname = tt:match('[^/]*%.mat')
         local track=mattorch.load(tt).track:t()
   
         local imidx=track:select(2,5)
         local imboxes=track:narrow(2,1,4)
         for dd=1,track:size(1) do
            local cix=imidx[dd]
            local cbb=imboxes:narrow(1,dd,1):index(2,permute_tensor) -- y1,x1,y2,x2 put in correct input format
            local imname=('%s/image-%05d.jpg'):format(vidname,cix)
            local im_cor = set_imcorres[cix]
            if not im_cor then -- the video frame is not in the set yet
               im_cor = #set_images+1
               set_imcorres[cix] = im_cor
   
               set_images[im_cor] = imname
               set_boxes[im_cor] = cbb:double()
               set_trackid[im_cor] = torch.LongTensor{trackid}
            else -- add the box to the proposals
               assert(set_images[set_imcorres[cix]] == imname)
               set_boxes[im_cor] = set_boxes[im_cor]:cat(cbb:double(),1)
               set_trackid[im_cor] = set_trackid[im_cor]:cat(torch.LongTensor{trackid},1)
            end
         end
      end
      tracks:close()
      nbimages=#set_images
      collectgarbage()
   end
   saveset(set_boxes,set_images,set_trackid,set_id)
   vids:close()
end

write_prop_file('test')
