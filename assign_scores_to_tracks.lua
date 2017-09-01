--require 'mattorch'
local mtorch=require 'mattorch'
require 'tds'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Options:')
cmd:option('-setnum',-1)
cmd:option('-fromSubVID',false,'we do not use the regular sets but the smaller one containing 1 video maximum (usefull when there is out of memory failure')
cmd:option('-endfile','raw.t7')
opts = cmd:parse(arg or {})

feature_saving = true -- either detection score or feature saving

sourcedirmattracks='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/track_info_forLSTM'


if feature_saving then
   --resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/mat_combined_features_tracks_FRCNNREPRO'
   resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/mat_combined_features_tracks_FRCNNREPRO_ROI'
   --featid=4 -- fc7
   featid=3 -- roi pooling
else
   resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/models/FRCNN_combined_action_scores'
end
local endfile=opts.endfile
scores_set_sources_app='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_app_FINAL/%s_result_set_*/'..endfile
scores_set_sources_flow='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_flow_FINAL/%s_result_set_*/'..endfile
prop_set_sourcesprefix='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/%stracks_set_%s.t7'

function get_sets_res(sets_path,split)
   local res,nbfound = {},0
   local res_sets=io.popen('ls '..sets_path:format(split)) ;
   for tt in res_sets:lines() do res[tt]=1 ; nbfound=nbfound+1 end
   -- reorder:
   local res_sorted={}
   if opts.fromSubVID then
      local cpt=0
      for k,_ in pairs(res) do
         cpt=cpt+1 ; res_sorted[cpt]=k
      end
      table.sort(res_sorted)
      return res_sorted
   end
   local spattern=sets_path:format(split):gsub('set_%*/','set_%%d/')
   for i=1,nbfound do
      local csetname=spattern:format(i)
      assert(res[csetname],'Missing SET')
      res_sorted[i]=csetname
   end
   return res_sorted
end

function save_tracks(tracks,tracks_to_check)
   for idname,nbmissing in pairs(tracks_to_check) do
      print(('%s - Empty detection, check track: %d/%d detections are missing'):format(idname,nbmissing,tracks[idname].cpt))
      print('-------------------------------------')
   end

   local ntracks=0 ; for k,v in pairs(tracks) do ntracks=ntracks+1 ; end
   local cptt=0
   for k,v in pairs(tracks) do
      cptt=cptt+1
      --io.write('track '..cptt..'/'..ntracks..'           \r'); io.flush()
      print(k)
      local ctrack=v.track
      local ori_size,filled_size=ctrack.gt_iou:size(2),v.cpt
      assert(ori_size==filled_size,'The track was not filled correctly: orig/fill sizes:'..ori_size..' and '..filled_size)
      local nbclasses=ctrack.flow_scores:size(1) -- including bck
      local s_scores=ctrack.spatial_scores
      local f_scores=ctrack.flow_scores

      -- clean track for saving
      ctrack.track=ctrack.track:double()
      ctrack.gt_tracks=nil
      ctrack.selected_action_scores=nil
      collectgarbage()

      if feature_saving then
         ctrack.gt_class_labels=nil
         ctrack.class_labels=nil
         ctrack.track_class_label=nil
         ctrack.gt_length=nil
         ctrack.gt_iou=nil
         paths.mkdir(v.savename:match('.*/'))
         collectgarbage()
         mtorch.save(v.savename,ctrack)
         ctrack=nil
         collectgarbage()

      else
      for c=1,nbclasses-1 do
         -- get combined score for the given class and save the class result
         ctrack.lstm_pred=s_scores:narrow(1,c,1):clone():add(f_scores:narrow(1,c,1):clone()):div(2)
         paths.mkdir(v.savename:format(c):match('.*/'))
         mtorch.save(v.savename:format(c),ctrack)
         collectgarbage()

         -- also do it for APP and FLOW only
         ctrack.lstm_pred=s_scores:narrow(1,c,1)
         local spath=v.savename:format(c):gsub('combined','APP')
         paths.mkdir(spath:match('.*/'))
         mtorch.save(spath,ctrack)
         collectgarbage()

         ctrack.lstm_pred=f_scores:narrow(1,c,1)
         spath=v.savename:format(c):gsub('combined','OF')
         paths.mkdir(spath:match('.*/'))
         mtorch.save(spath,ctrack)
         collectgarbage()
      end
      end
   end
   print('')
end

function write_res_file(split,fromsetnum,tosetnum)
   
   local res_sets_app=get_sets_res(scores_set_sources_app,split)
   local res_sets_flow=get_sets_res(scores_set_sources_flow,split)
   local fromsetnum = fromsetnum or 1

   local tosetnum = tosetnum or #res_sets_app
   if tosetnum>#res_sets_app then return end

   print(('APP/OF sets found: %d/%d'):format(#res_sets_app,#res_sets_flow))
   assert(#res_sets_app==#res_sets_flow)

   for i_set=fromsetnum,tosetnum do -- for each set
      collectgarbage()
      -- load the set of detections
      print('Load set '..i_set);
      print(res_sets_app[i_set])
      local loadapp = torch.load(res_sets_app[i_set])
      local loadflow= torch.load(res_sets_flow[i_set])
      print('Load set '..i_set..' LOADED!');
      collectgarbage()

      -- get scores
      local res_app = loadapp[1]
      local res_flow=loadflow[1]
      local reg

      if opts.fromSubVID then
         reg='set_.*/'
      else reg='set_[0-9]*/'
      end

      local set_number = res_sets_app[i_set]:match(reg):sub(5,-2)
      assert(set_number==res_sets_flow[i_set]:match(reg):sub(5,-2))

      if not opts.fromSubVID then set_number=tonumber(set_number) end

      local feat_app,feat_flow
      if feature_saving then
         feat_app = loadapp[featid]
         feat_flow=loadflow[featid]
      end

      loadapp,loadflow=nil,nil
      collectgarbage()

      local nblabels = res_app[1]:size(2) -- including bkg
      local nbdets = #res_app

      -- load corresponding proposals
      print('Load proposals '..(prop_set_sourcesprefix):format(split,set_number))
      local cur_prop = torch.load((prop_set_sourcesprefix):format(split,set_number))
      assert(#cur_prop.trackid==nbdets)

      local cur_tracks_to_fill,tracks_to_check = {},{}
      local cur_vid_to_fill,not_at_first = '',false
      collectgarbage()
      for i_det_set=1,nbdets do -- parse detections into tracks
         -- get detection info
         local alltrackids = cur_prop.trackid[i_det_set]

         local vidname = cur_prop.images[i_det_set]:match('(.*)/')

         if not_at_first and cur_vid_to_fill~=vidname then -- if the video has changed, save the prev tracks
            print(split,cur_vid_to_fill,'set '..i_set)
            save_tracks(cur_tracks_to_fill,tracks_to_check)
            cur_tracks_to_fill = {}
            tracks_to_check = {}
            collectgarbage()
         end
         cur_vid_to_fill,not_at_first = vidname,true

         -- it can have more track id than detection results due to proposal with area <= 2,
         -- check they indeed have <=2 area
         if not (alltrackids:size(1) == res_app[i_det_set]:size(1)) then

            local boxes=cur_prop.boxes[i_det_set]

            -- compute as in FRCNN code: area is no exactly <=2 (+1 is missing)
            local hw = boxes:narrow(2,3,2):clone():add(-1, boxes:narrow(2,1,2))
            local s = hw:select(2,1):cmul(hw:select(2,2))
            local idx = s:le(2) -- area is <= 2
            local missing_idx=torch.range(1,alltrackids:size(1))[idx]:long()
            --local missing_idx=torch.range(1,alltrackids:size(1))[hw:eq(0):sum(2):gt(0)]:long()

            assert(missing_idx:size(1)==alltrackids:size(1)-res_app[i_det_set]:size(1))
            if missing_idx:nDimension()==0 or not (missing_idx:size(1)==alltrackids:size(1)-res_app[i_det_set]:size(1)) then
               assert(false)
               -- we have not found the right number of 0 area tracks
               -- so it should have a small area somewhere, recompute missing_ix with a th:
               local cth=0.4
               for jj=1,6 do -- slowly increase the th to find the right missing number
                  cth=cth+0.1
                  missing_idx=torch.range(1,alltrackids:size(1))[hw:lt(cth):sum(2):gt(0)]:long()
                  if missing_idx:nDimension()>0 and (missing_idx:size(1)>=alltrackids:size(1)-res_app[i_det_set]:size(1)) then break end
               end

               -- check if we now have the correct number of 0 area
               if missing_idx:nDimension()==0 or not (missing_idx:size(1)==alltrackids:size(1)-res_app[i_det_set]:size(1)) then
                  print('==============================================')
                  print('CHECK at: i_set '..i_set,'i_det_set '..i_det_set)
                  print('Prediction sizes:')
                  print(
                     res_app[i_det_set]:size(),res_flow[i_det_set]:size(),
                     feat_app[i_det_set]:size(),feat_flow[i_det_set]:size())
                  print('Pred:')
                  print(res_app[i_det_set])
                  print('Boxes and their sizes:')
                  print(boxes,hw)
                  print('missing_idx',missing_idx)
   
                  --dofile('debug.lua') ; breakpoint('',{alltrackids,boxes,hw,missing_idx,res_app[i_det_set]})
                  assert(false,'we have not found the right number of 0 area tracks')
               end
            end

            -- add dummy numbers to detections
            local tensInsert = function(X,pos)
               assert(X)
               local xSize=X:size(1)
               local dummyInfo = X:narrow(1,1,1):clone():zero()
               dummyInfo[1][dummyInfo:size(2)]=1
               if pos==1 then -- insert at the first positions
                  X=dummyInfo:cat(X,1):clone()
               elseif pos==xSize+1 then -- append at the end
                  X=X:cat(dummyInfo,1):clone()
               else -- insert in the middle
                  X=torch.cat(X:narrow(1,1,pos-1),dummyInfo,1):cat(X:narrow(1,pos,xSize-pos+1),1):clone()
               end
               dummyInfo = nil
               collectgarbage()
               return X
            end
            for i_miss=1,missing_idx:size(1) do
               local i_det=missing_idx[i_miss]
               local trackid = alltrackids[i_det]
               local idname = vidname..'_'..tonumber(trackid)

               if not tracks_to_check[idname] then
                  tracks_to_check[idname]=1
               else
                  tracks_to_check[idname]=tracks_to_check[idname]+1 -- increase missing count
               end

               res_app[i_det_set]=tensInsert(res_app[i_det_set],i_det)
               res_flow[i_det_set]=tensInsert(res_flow[i_det_set],i_det)
               feat_app[i_det_set]=tensInsert(feat_app[i_det_set],i_det)
               feat_flow[i_det_set]=tensInsert(feat_flow[i_det_set],i_det)
            end

         end

         for i_det=1,alltrackids:size(1) do -- for all detections in this image
            local trackid = alltrackids[i_det]
            local idname = vidname..'_'..tonumber(trackid)

            if not cur_tracks_to_fill[idname] then -- if we have not loaded the track yet
               cur_tracks_to_fill[idname]={}
               local vidtrackname=(vidname..'/track%05d.mat'):format(trackid)
               cur_tracks_to_fill[idname].track=mtorch.load(sourcedirmattracks..'/'..vidtrackname) ; collectgarbage()
               cur_tracks_to_fill[idname].cpt=0
               if feature_saving then cur_tracks_to_fill[idname].savename=resdir..'/'..vidtrackname
               else cur_tracks_to_fill[idname].savename=resdir..'_CLASS%d/predictions/MATLAB/'..vidtrackname end
               local ctrack=cur_tracks_to_fill[idname].track
               ctrack.flow_scores:zero()
               ctrack.spatial_scores:zero()

               if feature_saving then
                  -- init the feature tensors
                  if not nFeatDims then nFeatDims=feat_flow[1][1]:nDimension() end
                  local tracksize = ctrack.flow_scores:size(2)
                  local sizeFeat={} ; for i_s=1,nFeatDims do sizeFeat[i_s] = feat_app[1][1]:size(nFeatDims-i_s+1) end
                  sizeFeat[nFeatDims+1]=tracksize -- size is reversed for mattorch saving

                  ctrack.feat_spatial = torch.DoubleTensor(torch.LongStorage(sizeFeat)):zero()
                  ctrack.feat_flow = torch.DoubleTensor(torch.LongStorage(sizeFeat)):zero()
               end
            end
            cur_tracks_to_fill[idname].cpt=cur_tracks_to_fill[idname].cpt+1 -- we add a detection to the track

            -- fill scores (and feaetures eventually)
            local ctrack,ccpt=cur_tracks_to_fill[idname].track,cur_tracks_to_fill[idname].cpt

            for _,v in pairs({
               {ctrack.spatial_scores,res_app ,ctrack.feat_spatial,feat_app},
               {ctrack.flow_scores   ,res_flow,ctrack.feat_flow   ,feat_flow}
               }) do
               -- action scores
               local score_,res_ = v[1]:narrow(2,ccpt,1),v[2][i_det_set][i_det]
               score_:narrow(1,1,nblabels-1):copy(res_:narrow(1,2,nblabels-1))
               score_[nblabels]=res_[1] -- bkg to the end

               if feature_saving then
                  local cur_feat=v[4][i_det_set][i_det]
                  local sav_position=v[3]:narrow(nFeatDims+1,ccpt,1)
                  sav_position:copy(cur_feat)
               end
            end
         end
      end
      print(split,cur_vid_to_fill,'set '..i_set)
      save_tracks(cur_tracks_to_fill,tracks_to_check) -- save the last tracks
   end
end

local from_i,to_i
if opts.setnum > 0 then -- avoid memory leak bug from mattorch
   from_i,to_i=opts.setnum,opts.setnum
end
if feature_saving then write_res_file('train',from_i,to_i) end
write_res_file('test',from_i,to_i)
print("ALL TRACKS SUCCESSFULLY ASSIGNED")
