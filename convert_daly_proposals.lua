require 'torch'
require 'nn'
require 'optim'
require 'xlua'

local tnt = require 'torchnet'

require 'fbcoco'

local json = require 'cjson'
local py = require('fb.python')
py.exec('import numpy as np')


-- contrary to annotations (in x,y,w,h format)
-- proposal are in format: x1,y1,x2,y2

-- example from:
-- /sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/VOC2007/selective_search/train.t7
-- #data.boxes = 2501
-- #data.images = 2501
-- data.boxes[1] is [torch.DoubleTensor of size 1264x4]
-- data.images[1] = 000012.jpg


sourcedir='/sequoia/data2/gcheron/DALY/frcnn_proposals/human_FasterRCNN_MPIIpose600small/'
resdir='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/keyframes/'
annotdir='/sequoia/data1/gcheron/code/torch/multipathnet/data/annotations/'
-- in Philippe's proposals: ( image_ID,x1,y1,x2,y2, score)  x1,y1,x2,y2 starting at 0


function write_prop_file(split)
   local data = {}
   data.boxes = {}
   data.images = {}
   local respath=resdir..split..'.t7'
   print('Compute proposals for '..split..' in: '..respath)

   -- load annotations
   local annotpath=annotdir..'daly_'..split..'.t7'
   local data_annot = torch.load(annotpath)

   local n_images = data_annot.images.file_name.data:size(1)

   local disp_step = math.ceil(n_images/20)
   for i = 1,n_images do -- for all annotated frames
      if i%disp_step==1 then print(('prop %d/%d'):format(i,n_images)) end

      local cur_im = data_annot.images.file_name.data[i]:clone():storage():string()      
      local vidname,imnum=cur_im:match('(.-)/image%-(.-)%.jpg')
      imnum=tonumber(imnum)

      -- load video proposals
      local proppath=sourcedir..vidname..'.mp4.npy'
      local cur_prop = py.eval('np.load(p)',{p=proppath})
      local idx = torch.linspace(1,cur_prop:size(1),cur_prop:size(1))[cur_prop:narrow(2,1,1):eq(imnum)]:long()
      local sel_prop = cur_prop:index(1,idx):narrow(2,2,4):add(1) -- select x1,y1,x2,y2 and make it state at 1

      data.boxes[i] = sel_prop:double()
      data.images[i] = cur_im
   end
   torch.save(respath,data)
end

write_prop_file('trainkeyframes')
write_prop_file('testkeyframes')
