g_count_imagesaving=0
function draw_prop_target_from_sample(sample,revim)

   if not ut then ut=dofile('utils.lua') end

   local nbimages=sample.input[1]:size(1)
   local respath='/meleze/data0/public_html/gcheron/frcnn_targprop/'

   local nb_prop_prob=10000000

   for i = 1,nbimages do
      local src=sample.input[1][i]:clone():float();
      print(src[1]:mean(),src[2]:mean(),src[3]:mean(),src:min(),src:max())
      print(src[1]:std(),src[2]:std(),src[3]:std())
      src:add(-src:min()) ; src:div(src:max());
      if revim then  src=src:index(1,torch.LongTensor{3,2,1}) end
      local nbprop=sample.input[2]:size(1)
      for j = 1,nbprop do
         local prop=sample.input[2][j]
         if i==prop[1] then -- this prop is for the current image 
            local _,x1, y1, x2, y2 = table.unpack(prop:totable())
            local class = sample.target[1][j]
            assert(class==sample.target[2][1][j])

            if class==1 then -- this is bkg
               assert(sample.target[2][2][j]:clone():abs():sum()==0) -- no regression assign
               if math.random() < nb_prop_prob/nbprop then
                  src=image.drawRect(src, x1, y1, x2, y2,{color={255, 0, 0}})
               end
            else
               local reg = sample.target[2][2][j]:narrow(1,(class-1)*4+1,4) -- get the regression target
               local reg_un = reg:clone()
               :cmul(state.network:get(6):get(2).std):add(state.network:get(6):get(2).mean) -- unormalize
               local gtbox=reg_un:clone()
               ut.convertFrom(gtbox,prop:narrow(1,2,4),reg_un) ; -- convert the proposal to its gt box
               local x1g, y1g, x2g, y2g = table.unpack(gtbox:totable())
               src=image.drawRect(src, x1g, y1g, x2g, y2g,{color={0, 255, 0}})
               src=image.drawRect(src, x1, y1, x2, y2,{color={0, 0, 255}})
               src=image.drawText(src,'C'..class,math.max(1,x1+1),math.max(1,y1+1),{color={0, 0, 255},size=1.5})
            end
         end
      end

      g_count_imagesaving=g_count_imagesaving+1
      image.save(respath..g_count_imagesaving..'.jpg',src)
print(src:size())
   end
end 
