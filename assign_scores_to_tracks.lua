require 'mattorch'
require 'tds'

feature_saving = true -- either detection score or feature saving

sourcedirmattracks='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/track_info_forLSTM'
if feature_saving then
   resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/mat_combined_features_tracks_FRCNNREPRO'
   featid=4 -- fc7
   --featid=3 -- roi pooling
else
   resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/models/FRCNN_combined_action_scores'
end
scores_set_sources_app='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_app_FINAL/%s_result_set_*/raw.t7'
scores_set_sources_flow='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_flow_FINAL/%s_result_set_*/raw.t7'
prop_set_sourcesprefix='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/%stracks_set_%d.t7'

function get_sets_res(sets_path,split)
   local res = {}
   local res_sets=io.popen('ls '..sets_path:format(split)) ;
   for tt in res_sets:lines() do res[#res+1]=tt end
   return res
end

function save_tracks(tracks,tracks_to_check)
   for idname,nbmissing in pairs(tracks_to_check) do
      print(('%s - Empty detection, check track: %d/%d detections are missing'):format(idname,nbmissing,tracks[idname].cpt))
   end

   for k,v in pairs(tracks) do
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

      if feature_saving then
         ctrack.gt_class_labels=nil
         ctrack.class_labels=nil
         ctrack.track_class_label=nil
         ctrack.gt_length=nil
         ctrack.gt_iou=nil
         paths.mkdir(v.savename:match('.*/'))
         mattorch.save(v.savename,ctrack)
         collectgarbage()

         return
      end
      for c=1,nbclasses-1 do
         -- get combined score for the given class and save the class result
         ctrack.lstm_pred=s_scores:narrow(1,c,1):clone():add(f_scores:narrow(1,c,1):clone()):div(2)
         paths.mkdir(v.savename:format(c):match('.*/'))
         mattorch.save(v.savename:format(c),ctrack)
         collectgarbage()

         -- also do it for APP and FLOW only
         ctrack.lstm_pred=s_scores:narrow(1,c,1)
         local spath=v.savename:format(c):gsub('combined','APP')
         paths.mkdir(spath:match('.*/'))
         mattorch.save(spath,ctrack)
         collectgarbage()

         ctrack.lstm_pred=f_scores:narrow(1,c,1)
         spath=v.savename:format(c):gsub('combined','OF')
         paths.mkdir(spath:match('.*/'))
         mattorch.save(spath,ctrack)
         collectgarbage()
      end
   end
end

function write_res_file(split)
   
   local res_sets_app=get_sets_res(scores_set_sources_app,split)
   local res_sets_flow=get_sets_res(scores_set_sources_flow,split)
   print(('APP/OF sets found: %d/%d'):format(#res_sets_app,#res_sets_flow))
   assert(#res_sets_app==#res_sets_flow)

   for i_set=1,#res_sets_app do -- for each set
      -- load the set of detections
      local loadapp = torch.load(res_sets_app[i_set])
      local loadflow= torch.load(res_sets_flow[i_set])

      -- get scores
      local res_app = loadapp[1]
      local res_flow=loadflow[1]
      local set_number = tonumber(res_sets_app[i_set]:match('set_[0-9]*/'):sub(5,-2))
      assert(set_number==tonumber(res_sets_flow[i_set]:match('set_[0-9]*/'):sub(5,-2)))

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
            print(split,cur_vid_to_fill)
            save_tracks(cur_tracks_to_fill,tracks_to_check)
            cur_tracks_to_fill = {}
            tracks_to_check = {}
            collectgarbage()
         end
         cur_vid_to_fill,not_at_first = vidname,true

         -- it can have more track id than detection results due to proposal with 0 area,
         -- check they indeed have 0 area
         if not (alltrackids:size(1) == res_app[i_det_set]:size(1)) then

            local boxes=cur_prop.boxes[i_det_set]
            -- compute as in FRCNN code: area is no exactly 0 (+1 is missing)
            local hw = boxes:narrow(2,3,2):clone():add(-1, boxes:narrow(2,1,2))
            local missing_idx=torch.range(1,alltrackids:size(1))[hw:eq(0):sum(2):gt(0)]:long()
            -- we found the right number of 0 area tracks
            assert(missing_idx:size(1)==alltrackids:size(1)-res_app[i_det_set]:size(1))

            -- add dummy numbers to detections
            local tensInsert = function(X,pos)
               assert(X)
               local xSize=X:size(1)
               local dummyInfo = X:narrow(1,1,1):clone():zero()
               if pos==1 then -- insert at the first positions
                  X=dummyInfo:cat(X,1)
               elseif pos==xSize+1 then -- append at the end
                  X=X:cat(dummyInfo,1)
               else -- insert in the middle
                  X=torch.cat(X:narrow(1,1,pos-1),dummyInfo,1):cat(X:narrow(1,pos,xSize-pos+1),1)
               end
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
               cur_tracks_to_fill[idname].track=mattorch.load(sourcedirmattracks..'/'..vidtrackname) ; collectgarbage()
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
      print(split,cur_vid_to_fill)
      save_tracks(cur_tracks_to_fill,tracks_to_check) -- save the last tracks
   end
end

write_res_file('test')
if feature_saving then write_res_file('train') end
