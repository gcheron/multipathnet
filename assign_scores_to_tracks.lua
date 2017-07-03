require 'mattorch'
require 'tds'

sourcedirmattracks='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/track_info_forLSTM'
resdir='/sequoia/data2/gcheron/lstm_time_detection_datasets/DALY_philippe_tracks/models/FRCNN_combined_action_scores'
scores_set_sources_app='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_app_FINAL/result_set_*/raw.t7'
scores_set_sources_flow='/sequoia/data2/gcheron/multipathnet_results/logs/fastrcnn_daly_flow_FINAL/result_set_*/raw.t7'
prop_set_sourcesprefix='/sequoia/data1/gcheron/code/torch/multipathnet/data/proposals/daly/tracks/testtracks_set_'

function get_sets_res(sets_path)
   local res = {}
   local res_sets=io.popen('ls '..sets_path) ;
   for tt in res_sets:lines() do res[#res+1]=tt end
   return res
end

function save_tracks(tracks)
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

function write_res_file()
   
   local res_sets_app=get_sets_res(scores_set_sources_app)
   local res_sets_flow=get_sets_res(scores_set_sources_flow)
   assert(#res_sets_app==#res_sets_flow)

   for i_set=1,#res_sets_app do -- for each set
      -- load the set of detections
      local res_app = torch.load(res_sets_app[i_set])[1]
      local res_flow = torch.load(res_sets_flow[i_set])[1]
      local set_number = tonumber(res_sets_app[i_set]:match('set_[0-9]*/'):sub(5,-2))
      assert(set_number==tonumber(res_sets_flow[i_set]:match('set_[0-9]*/'):sub(5,-2)))

      local nblabels = res_app[1]:size(2) -- including bkg
      local nbdets = #res_app

      -- load corresponding proposals
      local cur_prop = torch.load(prop_set_sourcesprefix..set_number..'.t7')
      assert(#cur_prop.trackid==nbdets)

      local cur_tracks_to_fill = {}
      local cur_vid_to_fill,not_at_first = '',false
      for i_det_set=1,nbdets do -- parse detections into tracks
         -- get detection info
         local alltrackids = cur_prop.trackid[i_det_set]
         local vidname = cur_prop.images[i_det_set]:match('(.*)/')

         if not_at_first and cur_vid_to_fill~=vidname then -- if the video has changed, save the prev tracks
            print(cur_vid_to_fill)
            save_tracks(cur_tracks_to_fill)
            cur_tracks_to_fill = {}
            collectgarbage()
         end
         cur_vid_to_fill,not_at_first = vidname,true

         for i_det=1,alltrackids:size(1) do -- for all detections in this image
            local trackid = alltrackids[i_det]
            local idname = vidname..'_'..tonumber(trackid)

            if not cur_tracks_to_fill[idname] then -- if we have not loaded the track yet
               cur_tracks_to_fill[idname]={}
               local vidtrackname=(vidname..'/track%05d.mat'):format(trackid)
               cur_tracks_to_fill[idname].track=mattorch.load(sourcedirmattracks..'/'..vidtrackname) ; collectgarbage()
               cur_tracks_to_fill[idname].cpt=0
               cur_tracks_to_fill[idname].savename=resdir..'_CLASS%d/predictions/MATLAB/'..vidtrackname
               local ctrack=cur_tracks_to_fill[idname].track
               ctrack.flow_scores:zero()
               ctrack.spatial_scores:zero()
            end
            cur_tracks_to_fill[idname].cpt=cur_tracks_to_fill[idname].cpt+1 -- we add a detection to the track
            local ctrack,ccpt=cur_tracks_to_fill[idname].track,cur_tracks_to_fill[idname].cpt
--dofile('debug.lua') ; breakpoint('r',{tracks,ctrack})
            for _,v in pairs({{ctrack.spatial_scores,res_app},{ctrack.flow_scores,res_flow}}) do
               local score_,res_ = v[1]:narrow(2,ccpt,1),v[2][i_det_set][i_det]
               score_:narrow(1,1,nblabels-1):copy(res_:narrow(1,2,nblabels-1))
               score_[nblabels]=res_[1] -- bkg to the end
            end
         end
      end
      print(cur_vid_to_fill)
      save_tracks(cur_tracks_to_fill) -- save the last tracks
   end
end

write_res_file()
