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
maximagesperset=1/0 -- 40e3 -- create a new set when this number is reached (note that a same video will be put in the same set anyway)
maxdetectionsperset=60000 -- create a new set when this number of boxes is reached (note that a same track will be put in the same set anyway)

assert(maxdetectionsperset == 1/0 or maximagesperset == 1/0, 'One of the 2 shouw be equal to Inf') 


function videonamesFromtrackset(setpath)
   local vidnames={}
   local setl=torch.load(setpath)
   print('vidnames founds:')
   local prevname=""
   for ii=1,#setl.images do
      cname=setl.images[ii]:match('(.*)/')
      if cname~=prevname then
         print(cname)
         vidnames[#vidnames+1]=cname
         prevname=cname
      end
   end
   return vidnames
end

function saveset(set_boxes,set_images,set_trackid,set_id,vidname2save)
   assert(#set_boxes==#set_images and #set_boxes==#set_trackid)

   local suffix=""
   if vidname2save then suffix='_VID_'..vidname2save end

   local set_name=resprefixpath..set_id..suffix..'.t7'
   local annotation_name=annprefixpath..set_id..suffix..'.json'
   print('SAVING SET: '..set_name)
   print('Nb images: '..#set_images)
   local cpt=0 ; for jj = 1,#set_boxes do cpt=cpt+set_boxes[jj]:size(1) ; end ;
   print('Nb detections: '..cpt)
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

function write_prop_file(split,vidname2save)
   -- set vidname2save in order to save tracks from this video only
   print('Create files for split: '..split)
   resprefixpath='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/'..split..'tracks_set_'
   annprefixpath='/sequoia/data1/gcheron/code/torch/multipathnet/data/annotations/daly_'..split..'tracks_set_'
   vidlist='/sequoia/data2/gcheron/DALY/OF_vidlist_'..split..'1.txt'
      
   vids=io.open(vidlist,'r')
   local nbvideos=0
   local nbimages,nbdetections=0,0
   local set_id,set_boxes,set_images,set_trackid = 1,{},{},{}
   
   for vidname_exp in vids:lines() do
      local vidname = vidname_exp:match('[^%s]*')

      if not vidname2save or vidname==vidname2save then

      nbvideos=nbvideos+1
      print(('Video %03d (%s)'):format(nbvideos,vidname))
      if nbimages>maximagesperset then -- create a new set
         saveset(set_boxes,set_images,set_trackid,set_id,vidname2save)
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
            local cix=imidx[dd] -- image idx
            local cbb=imboxes:narrow(1,dd,1):index(2,permute_tensor) -- y1,x1,y2,x2 put in correct input format
            local imname=('%s/image-%05d.jpg'):format(vidname,cix)
            local im_cor = set_imcorres[cix] -- this image corresponds to this idx in the current set
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
            nbdetections=nbdetections+1
         end
         if nbdetections>maxdetectionsperset then -- create a new set
            saveset(set_boxes,set_images,set_trackid,set_id,vidname2save)
            set_id=set_id+1
            set_boxes,set_images,set_trackid = {},{},{}
            nbdetections,set_imcorres=0,{}
            collectgarbage()
         end
      end
      tracks:close()
      nbimages=#set_images
      collectgarbage()

      end
   end
   if #set_boxes>0 then saveset(set_boxes,set_images,set_trackid,set_id,vidname2save) end
   vids:close()
end

-- write everything
--write_prop_file('train')
--write_prop_file('test')

-- write one file per video that are contained in these sets
-- usefull when set crashed because out of memory
--[[
setOfsets={
   {'/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/testtracks_set_274.t7','test'},
   {'/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/testtracks_set_177.t7','test'},
   {'/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/traintracks_set_59.t7','train'},
   {'/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/testtracks_set_63.t7','test'},
   {'/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/testtracks_set_64.t7','test'},
}
for s=1,#setOfsets do
   local cvids=videonamesFromtrackset(setOfsets[s][1])
   for v=1,#cvids do
      write_prop_file(setOfsets[s][2],cvids[v])
   end
end
]] --
cvids={
   {'2K5xBtXzSac','test'},
   {'4shlpOiiXBo','test'},
   {'qYMaowVVJrI','test'},
   {'RYM7uZeiXH0','test'},
   {'UBF1DfE0d5Y','test'},
   {'_zresAM2eXs','test'},
}
for v=1,#cvids do
   write_prop_file(cvids[v][2],cvids[v][1])
end
