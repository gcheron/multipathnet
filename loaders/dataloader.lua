--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

local loader = require 'loaders.loader'

local datasets = {}

local path_config = require 'config'

local dataset_path = './data/annotations'

-- Add COCO datasets
for _,name in ipairs{'train', 'val', 'test'} do
   local file = dataset_path .. '/instances_' .. name .. '2014.json'
   datasets['coco_' .. name .. '2014'] = file
   datasets[name] = file
end
for _,name in ipairs{'test2014', 'test2015-dev', 'test2015-full'} do
   local file = dataset_path .. '/instances_' .. name .. '.json'
   datasets['coco_' .. name] = file
   datasets[name] = file
end

-- Add Pascal datasets
for _,name in ipairs{'train2007', 'train2012', 'val2007', 'val2012', 'test2007'} do
   local file = dataset_path .. '/pascal_' .. name .. '.json'
   datasets['pascal_' .. name] = file
end

-- Add ImageNet detection datasets
for _,name in ipairs{'train2014','val2013'} do
   local file = dataset_path .. '/imagenet_' .. name .. '.json'
   datasets['imagenet_' .. name] = file
end

-- Add DALY datasets
local dalynames={'trainkeyframes', 'testkeyframes','trainkeyframes_flow','testkeyframes_flow'}
for i=1,280 do dalynames[#dalynames+1]='testtracks_set_'..i end
for i=1,399 do dalynames[#dalynames+1]='traintracks_set_'..i end

-- load DALY sub VID sets
local vid_sets=io.popen('ls /sequoia/data1/gcheron/code/torch/multipathnet/data/annotations/daly_*tracks_set_*_VID_*_flow*') ;
for i in vid_sets:lines() do
   dalynames[#dalynames+1]=i:match('.*/daly_([^/]*)_flow%.json') ;
end
vid_sets:close()

for _,name in ipairs(dalynames) do
   datasets['daly_' .. name] =            dataset_path .. '/daly_' .. name .. '.json'
   datasets['daly_' .. name .. '_flow'] = dataset_path .. '/daly_' .. name .. '_flow.json'
end

-- e.g. coco.DataLoader('train') or coco.DataLoader('pascal_train2007')
local function DataLoader(dset)
   if torch.typename(dset) == 'dataLoader' then return dset end
   local file = datasets[dset]

   if not file then
      error('invalid dataset: ' .. tostring(dset))
   end
   assert(path_config[dset], 'image dir not set in config.lua')

   return loader():load(file, path_config[dset])
end

return DataLoader
