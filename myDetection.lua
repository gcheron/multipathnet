--[[----------------------------------------------------------------------------
Copyright (c) 2016-present, Facebook, Inc. All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. An additional grant
of patent rights can be found in the PATENTS file in the same directory.

------------------------------------------------------------------------------]]

-- DeepMask + MultiPathNet demo

require 'deepmask.SharpMask'
require 'deepmask.SpatialSymmetricPadding'
require 'deepmask.InferSharpMask'
require 'inn'
require 'fbcoco'
require 'image'
local model_utils = require 'models.model_utils'
local utils = require 'utils'
local coco = require 'coco'

local cmd = torch.CmdLine()
cmd:option('-np', 5,'number of proposals to save in test')
cmd:option('-si', -2.5, 'initial scale')
cmd:option('-sf', .5, 'final scale')
cmd:option('-ss', .5, 'scale step')
cmd:option('-dm', false, 'use DeepMask version of SharpMask')
cmd:option('-img','./deepmask/data/testImage.jpg' ,'path/to/test/image')
cmd:option('-imglist','' ,'path/to/test/image/list.txt')
cmd:option('-imgrespath','./res.jpg' ,'path/to/result/image')
cmd:option('-boxrespath','./resbox.jpg' ,'path/to/result/box')
cmd:option('-gtboxpath','' ,'path/to/GT/box')
cmd:option('-thr', 0.5, 'multipathnet score threshold [0,1]')
cmd:option('-maxsize', 600, 'resize image dimension')
cmd:option('-sharpmask_path', 'data/models/sharpmask.t7', 'path to sharpmask')
cmd:option('-multipath_path', 'data/models/resnet18_integral_coco.t7', 'path to multipathnet')
cmd:option('-draw', false,'draw boxes and save resulting image')
cmd:option('-start_id', 1,'start image positions in the list')

local config = cmd:parse(arg)

local draw = config.draw ;
local sharpmask = torch.load(config.sharpmask_path).model
sharpmask:inference(config.np)

local multipathnet = torch.load(config.multipath_path)
multipathnet:evaluate()
multipathnet:cuda()
model_utils.testModel(multipathnet)

local detector = fbcoco.ImageDetect(multipathnet, model_utils.ImagenetTransformer())

local dataset = paths.dofile'./DataSetJSON.lua':create'coco_val2014' -- get detection classnames

------------------- Prepare DeepMask --------------------

local meanstd = {mean = { 0.485, 0.456, 0.406 }, std = { 0.229, 0.224, 0.225 }}

local scales = {}
for i = config.si,config.sf,config.ss do table.insert(scales,2^i) end
--print(scales)

local infer = Infer{
  np = config.np,
  scales = scales,
  meanstd = meanstd,
  model = sharpmask,
  dm = config.dm,
}

------------------- Prepare image paths --------------------
imagelist = {} ;
imagereslist = {} ;
boxreslist = {} ;
gtboxlist = {} ;
if config.imglist == ""
then
   imagelist[1] = config.img
   imagereslist[1] = config.imgrespath
   boxreslist[1] = config.boxrespath
else 
   local file = io.open ('imglist.txt','r') 
   local count = 1
   while true do
      local impath = file:read()
      if impath == nil then break end
      local imname = impath:gsub('.*/',''):gsub('%..*','')
      imagelist[count] = impath
      imagereslist[count] = ('%s/%s.jpg'):format(config.imgrespath,imname)
      boxreslist[count] = ('%s/%s.t7'):format(config.boxrespath,imname)
      gtboxlist[count] = ('%s/%s.t7'):format(config.gtboxpath,imname)
      count = count + 1 
   end
   print(('%d images found!'):format(#imagelist))
   file:close()
end

local myDrawText = function(res,x1,y2,text,savepath)
   if pcall(function()
      image.drawText(res,text, x1, y2, {bg={255,255,255}, inplace=true})
   end) then ; else print('no text added in: ' .. savepath) ; end
end


local prev_time = 0 ;
local time = torch.tic() ;
for im_i = config.start_id,#imagelist do
------------------- Run DeepMask --------------------

   local img = image.load(imagelist[im_i])
   local imgOri = img:clone() ;

   local original_scale = config.maxsize/math.max(img:size(2),img:size(3))
   --print('original scale:' .. original_scale)
   --print('from size: ' .. img:size(2),img:size(3))
   img = image.scale(img, config.maxsize)
   local h,w = img:size(2),img:size(3)
   --print('to size: ' .. img:size(2),img:size(3))

   infer:forward(img)

   local masks,_ = infer:getTopProps(.2,h,w)

   local Rs = coco.MaskApi.encode(masks)
   local bboxes = coco.MaskApi.toBbox(Rs)
   bboxes:narrow(2,3,2):add(bboxes:narrow(2,1,2)) -- convert from x,y,w,h to x1,y1,x2,y2
   --print(bboxes)

------------------- Run MultiPathNet --------------------

   local detections = detector:detect(img:float(), bboxes:float())
   local prob, maxes = detections:max(2)
   --print(maxes)

   -- remove background detections
   local nonbg_idx = maxes:squeeze():gt(1):cmul(prob:gt(config.thr)):nonzero()
   local idx,scored_boxes,final_idx

   local selected_index = {} ; -- [1] = prob_id ; [2] = area_id

   local resbox = {} ;

   if not(nonbg_idx:size():size()==0) then -- if there is at least one detection
      idx = nonbg_idx:select(2,1)
      bboxes = bboxes:index(1, idx)
      maxes = maxes:index(1, idx)
      prob = prob:index(1, idx)

      scored_boxes = torch.cat(bboxes:float(), prob:float(), 2)
      final_idx = utils.nms_dense(scored_boxes, 0.3)

------------------- Select human detections --------------------

      local person_prob = -1 ; -- store best score
      local person_area = -1 ; -- sotre largest box
   
      for i,v in ipairs(final_idx:totable()) do
         local class = maxes[v][1]-1
         local name = dataset.dataset.categories[class]
         if name == "person" then
            local x1,y1,x2,y2 = table.unpack(bboxes[v]:totable())
            local wb,hb = x2-x1,y2-y1 
   
            local cprob = prob[v][1]
            local carea = wb*hb ;
            if cprob > person_prob then
               person_prob = cprob
               selected_index[1] = v ;
            end 
            if carea > person_area then
               person_area = carea 
               selected_index[2] = v ;
            end
          end
      end
      -- save all detections above th
      resbox.boxes_scores_classes=torch.cat(bboxes:double()/original_scale, torch.cat(prob:double(),maxes:double():add(-1), 2),2)
    end
------------------- Save & Draw detections --------------------
   h,w=imgOri:size(2),imgOri:size(3)

   -- save ground truth
   local gtmask,res 
   if draw then res = imgOri end

   if config.gtboxpath ~= '' then
      local gtbox = torch.load(gtboxlist[im_i])
      resbox.GT = gtbox:double() ;

      local x1g,y1g,x2g,y2g = table.unpack(gtbox:totable())
      local wbg,hbg = x2g-x1g,y2g-y1g
      gtmask = coco.MaskApi.frBbox(torch.Tensor{x1g,y1g,wbg,hbg},h,w) ;
 
      if draw then
         coco.MaskApi.drawMasks(res, coco.MaskApi.decode(gtmask))
         y2g = math.min(y2g, res:size(2)) - 10
         myDrawText(res,x1g,y2g,'GT',imagereslist[im_i])
      end
   end

   if selected_index[1] then -- if a human detection exists
      local v=selected_index[1]
      local cbox = bboxes[v]/original_scale
      resbox.score = cbox
      
      local x1,y1,x2,y2 = table.unpack(cbox:totable())
      local wb,hb = x2-x1,y2-y1
      local cmask = coco.MaskApi.frBbox(torch.Tensor{x1,y1,wb,hb},h,w) ;
      if config.gtboxpath ~= '' then resbox.score_iou = coco.MaskApi.iou(cmask,gtmask)[1][1] ; end

      if draw then
         coco.MaskApi.drawMasks(res, coco.MaskApi.decode(cmask))
         y2 = math.min(y2, res:size(2)) - 10
         myDrawText(res,x1,y2,('score %.2f'):format(resbox.score_iou),imagereslist[im_i])
      end

      v=selected_index[2]
      cbox = bboxes[v]/original_scale
      resbox.area = cbox

      x1,y1,x2,y2 = table.unpack(cbox:totable())
      wb,hb = x2-x1,y2-y1
      cmask = coco.MaskApi.frBbox(torch.Tensor{x1,y1,wb,hb},h,w) ;
      if config.gtboxpath ~= '' then resbox.area_iou = coco.MaskApi.iou(cmask,gtmask)[1][1] ; end
 
      if draw then
         coco.MaskApi.drawMasks(res, coco.MaskApi.decode(cmask))
         y2 = math.min(y2, res:size(2)) - 10
         myDrawText(res,x1,y2,('area %.2f'):format(resbox.area_iou),imagereslist[im_i])
      end
   else
      resbox.score_iou=0
      resbox.area_iou=0
   end
   torch.save(boxreslist[im_i],resbox)
   if draw then image.save(imagereslist[im_i],res) end

   local measured_time = torch.toc(time) ;
   local estimated_time = (measured_time/(im_i-config.start_id+1))*(#imagelist-im_i) ; -- estimate remaining time
   print(('%d out of %d images: %.2f s (remaining time: %.0f s)'):format(im_i,#imagelist,measured_time-prev_time,estimated_time))
   prev_time = measured_time ;
end
