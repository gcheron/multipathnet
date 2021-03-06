--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- put your paths to VOC and COCO containing subfolders with images here
-- local VOCdevkit = '/home/zagoruys/raid/datasets/VOCdevkit'
local VOCdevkit = '/mnt/disk1/glustervolume/gcheron/torch/torch/PASCAL07/VOCdevkit'
local coco_dir = '/home/zagoruys/raid/datasets/mscoco'
local daly_dir = '/sequoia/data2/gcheron/DALY'

local dalynames={'daly_trainkeyframes','daly_testkeyframes'}
for i=1,280 do dalynames[#dalynames+1]='daly_testtracks_set_'..i end
for i=1,399 do dalynames[#dalynames+1]='daly_traintracks_set_'..i end

-- load DALY sub VID sets
local vid_sets=io.popen('ls /sequoia/data1/gcheron/code/torch/multipathnet/data/annotations/daly_*tracks_set_*_VID_*_flow*') ;
for i in vid_sets:lines() do
   dalynames[#dalynames+1]=i:match('.*/([^/]*)_flow%.json') ;
end
vid_sets:close()


local restab = {
   pascal_train2007 = paths.concat(VOCdevkit, 'VOC2007/JPEGImages'),
   pascal_val2007 = paths.concat(VOCdevkit, 'VOC2007/JPEGImages'),
   pascal_test2007 = paths.concat(VOCdevkit, 'VOC2007/JPEGImages'),
   pascal_train2012 = paths.concat(VOCdevkit, 'VOC2012/JPEGImages'),
   pascal_val2012 = paths.concat(VOCdevkit, 'VOC2012/JPEGImages'),
   pascal_test2012 = paths.concat(VOCdevkit, 'VOC2012/JPEGImages'),
   coco_train2014 = paths.concat(coco_dir, 'train2014'),
   coco_val2014 = paths.concat(coco_dir, 'val2014'),
}
for _,v in ipairs(dalynames) do
   restab[v] = paths.concat(daly_dir, 'images') ;
   restab[v..'_flow'] = paths.concat(daly_dir, 'OF_closest')
end
return restab
