----------------------
-- usage:
-- th drawbox_localization.lua -proposals data/proposals/daly/keyframes/testkeyframes.t7 -resboxes /sequoia/data2/gcheron/mutlipathnet_results/logs/fastrcnn_daly_app_run2/boxes.t7 -annotations data/annotations/daly_testkeyframes.t7 -impath /sequoia/data2/gcheron/DALY/images -respath /meleze/data0/public_html/gcheron/fastrcnnAPP -score_th 0.5 -maxsize 500
----------------------

require 'tds'
require 'image'
local coco = require 'coco'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('PART/SPLIT/AR parsing script')
cmd:text()
cmd:text('Options:')
------------ General options --------------------

cmd:option('-resboxes','', 'file containing the boxes results')
cmd:option('-proposals','', 'file containing the proposals corresponding to the boxes')
cmd:option('-annotations','', 'file containing the categories')
cmd:option('-impath','','dataset images path')
cmd:option('-respath','', 'path where to write resulting images')
cmd:option('-maxsize',1000, 'maximum image size')
cmd:option('-score_th',0.1, 'minimum detection score to display the box')
cmd:text()

local opts = cmd:parse(arg or {})


local myDrawText = function(res,x1,y2,text,savepath,color,bg)
      local color = color or {255, 0, 0}
      local bg = bg or {255, 255, 255}
      if pcall(function()
                  image.drawText(res,text, x1, y2, {bg=bg,color=color, inplace=true, size=3})
               end) then ;
      else --print('no text added in: ' .. savepath) ;
      end
   end
function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

boxes = torch.load(opts.resboxes)
prop = torch.load(opts.proposals)
local nblabels =  #boxes

categories = {}
if opts.annotations ~= '' then
   ann=torch.load(opts.annotations)
   assert(ann.categories.name.idx:size(1)-1 == nblabels)
   for i=1,nblabels do categories[i] = ann.categories.name[i] end
else
   for i=1,nblabels do categories[i] = 'class'..i end
end
assert(#prop.images == #boxes[1])

local nbimages = #prop.images


--dofile'debug.lua' ; breakpoint('indeb')

for i = 1,nbimages do
   local impath = opts.impath..'/'..prop.images[i]
   local respath = opts.respath..'/'..prop.images[i]
   assert(not file_exists(respath),'image already exists: '..respath)
   local resfolder = respath:match('.*/')
   paths.mkdir(resfolder)
   local res = image.load(impath)
   local h,w=res:size(2),res:size(3)

   -- DRAW DETECTION
   for c=1,nblabels do
      local bboxes= boxes[c][i] ;

      if bboxes:nDimension()>0 then

         local nboxes = bboxes:size(1)
         local selidx = torch.linspace(1,nboxes,nboxes)[bboxes:narrow(2,5,1):ge(opts.score_th)]:long() 
   
         if selidx:nDimension()>0 then
            bboxes=bboxes:index(1,selidx)
            nboxes = bboxes:size(1)
         else nboxes=0
         end
   
         for b=1,nboxes do
            local x1,y1,wb,hb,sc = table.unpack(bboxes[b]:totable())
            local colo={1.0,.0,.0}
            local scorestr = ('%.02f'):format(sc)
            local text=('%s (%s)'):format(categories[c],scorestr)
            local mask = coco.MaskApi.frBbox(torch.Tensor{x1,y1,wb,hb},h,w) ;
            coco.MaskApi.drawMasks(res, coco.MaskApi.decode(mask),nil,nil,torch.DoubleTensor{colo})
            y2 = math.min(y1+hb, res:size(2)) - 50
            myDrawText(res,x1,y2,text,savepath)
         end
      end
   end
   local ch,cw=res:size(2),res:size(3)
   local msiz=math.min(ch,cw)
   if msiz > opts.maxsize then
      if ch==msiz then res=image.scale(res,opts.maxsize,opts.maxsize*ch/cw)
      else res=image.scale(res,opts.maxsize*cw/ch,opts.maxsize) end
   end
   image.save(respath,res)
end
