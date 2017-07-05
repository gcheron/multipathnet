for i in `seq -w 001 280` ;
do
   ii=$(echo $i | sed -e 's/^0*//g')
   echo "save_roipooling_fc7=true track_setid=$ii ../test_fastrcnn_daly_app.sh" > s${i}_app_dalytracktest.sh
   echo "save_roipooling_fc7=true track_setid=$ii ../test_fastrcnn_daly_flow.sh" > s${i}_flow_dalytracktest.sh
done
for i in `seq -w 001 399` ;
do
   ii=$(echo $i | sed -e 's/^0*//g')
   echo "track_split=train save_roipooling_fc7=true track_setid=$ii ../test_fastrcnn_daly_app.sh" > s${i}_app_dalytracktrain.sh
   echo "track_split=train save_roipooling_fc7=true track_setid=$ii ../test_fastrcnn_daly_flow.sh" > s${i}_flow_dalytracktrain.sh
done



cd /sequoia/data1/gcheron/code/torch/training_tools/job_manager
for i in `seq -w 001 399` ;
do
   ii=$(echo $i | sed -e 's/^0*//g') ;
   if [[ $ii -le 280 ]] ;
   then
      SH_generate_TITANGAIA_config.sh /sequoia/data1/gcheron/code/torch/multipathnet/scripts/jobs/s${i}_app_dalytracktest.sh ;
      SH_generate_TITANGAIA_config.sh /sequoia/data1/gcheron/code/torch/multipathnet/scripts/jobs/s${i}_flow_dalytracktest.sh ;
   fi
   SH_generate_TITANGAIA_config.sh /sequoia/data1/gcheron/code/torch/multipathnet/scripts/jobs/s${i}_app_dalytracktrain.sh
   SH_generate_TITANGAIA_config.sh /sequoia/data1/gcheron/code/torch/multipathnet/scripts/jobs/s${i}_flow_dalytracktrain.sh
   sleep 1 ;
   echo $i
done
