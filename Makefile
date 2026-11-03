# Run all commands in one shell
.ONESHELL:

# 1) first run the pickling step (make data-pickling) to generate the data for SRM
# 2) then run the eSRM step (make model-srm) to generate figures for eSRM denoising
# and across subject encoding
# 3) For the pooling analysis, run make srm-aug to generate the data for pooling analysis
# and then run make plotting-pooling to generate the figures for pooling analysis

PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))


%-pickling: subjects ?=  625 676 798	 ## 625 676 7170 798
%-pickling: window_length = 500  ## 500 ms
%-pickling: model_name= 'gpt2-xl'  ## 'glove50'|'gpt2-xl'
%-pickling: lags ?= 250 	#0 50 100 #
%-pickling: production = 0        ## 1 means production, 0 means comprehension
%-pickling: elec_list = 'new_path'   ## 'old_path'|'new_path'|'erp-corr'
%-pickling: folds = 10
%-pickling: CMD = sbatch --job-name=model_srm-$(shell date +%Y%m%d-%H%M%S)-sub-proportion-$(production) submit.sh
# %-pickling: CMD = python 
# %-pickling: CMD = echo


data-pickling:
	
		$(CMD) $(PROJECT_DIR)/pickling.py \
		--production $(production) \
		--subject $(subjects) \
		--window_length $(window_length) \
		--lags $(lags) \
		--elec_list $(elec_list) \
		--folds $(folds) \
		--model_name $(model_name) \
		




%-srm: subjects ?=  625 676 798	 ## 625 676 7170 798
%-srm: window_length = 200  ## 200 ms
%-srm: lags ?= -250 	#0 50 100 #
%-srm: total_conv_numbers ?= 54 78 24 15  ### [54,78,24,15] for [625,676,7170,798] 
%-srm: production = 1        ## 1 means production, 0 means comprehension
%-srm: elec_list = 'new_path'   ## 'old_path'|'new_path'|'erp-corr'
%-srm: srm_k = 10  ## number of srm features
%-srm: proportion_analysis = 'percentage'  ## 'percentage'|'fixed_number'
%-srm: org= --original_srm_encoding
# %-srm: across = --across_subject_encoding
# %-srm: across_2= --across_subject_shared_space
%-srm: across_3= --across_subject_with_replacing_srm
# %-srm: across_4= --across_subject_with_various_train_data_proportion
# %-srm: all_sub= --encoding_with_all_subjects_data
%-srm: plotting = --plotting
# %-srm: mat_file_saving = --mat_file_saving
# %-srm: select_elec= --selected_elec_id
%-srm: folds = 10
# %-srm: desired_fold = 0
# %-srm: CMD = sbatch --job-name=model_srm-sub-proportion-$(production) submit.sh
%-srm: CMD = sbatch --job-name=model_srm-$(shell date +%Y%m%d-%H%M%S)-sub-proportion-$(production) submit.sh
# %-srm: CMD = python 
# %-srm: CMD = echo
# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh


model-srm:

	$(CMD) $(PROJECT_DIR)/csrm_ecog_2.py\
    --production $(production) \
    --subject $(subjects) \
    --window_length $(window_length) \
    --total_conv_numbers $(total_conv_numbers) \
    --lags $(lags) \
    --elec_list $(elec_list) \
    --srm_k $(srm_k) \
    --folds $(folds) \
	--proportion_analysis $(proportion_analysis) \
	$(if $(org),$(org)) \
	$(if $(across),$(across)) \
	$(if $(across_2),$(across_2)) \
	$(if $(across_3),$(across_3)) \
	$(if $(across_4),$(across_4)) \
	$(if $(all_sub),$(all_sub)) \
	$(if $(select_elec),$(select_elec)) \
	$(if $(plotting),$(plotting)) \
	$(if $(mat_file_saving),$(mat_file_saving)); \
		
		

random-srm:

	for i in {1..5}; do \
		$(CMD) $(PROJECT_DIR)/csrm_ecog_2.py \
		--production $(production) \
		--subject $(subjects) \
		--window_length $(window_length) \
		--total_conv_numbers $(total_conv_numbers) \
		--lags $(lags) \
		--elec_list $(elec_list) \
		--random_seed $$i \
		--srm_k $(srm_k) \
		--folds $(folds) \
		--proportion_analysis $(proportion_analysis) \
		$(if $(org),$(org)) \
		$(if $(across),$(across)) \
		$(if $(across_2),$(across_2)) \
		$(if $(across_3),$(across_3)) \
		$(if $(across_4),$(across_4)) \
		$(if $(all_sub),$(all_sub)) \
		$(if $(select_elec),$(select_elec)) \
		$(if $(plotting),$(plotting)) \
		$(if $(mat_file_saving),$(mat_file_saving)); \
	done


%-aug: subjects ?=  625 676 798	 ## 625 676 7170 798
%-aug: data_size ?=  100 500 1000 2500 5000 10000 'all'	 ## 100 500 1000 2500 5000 10000 'all'
%-aug: lags ?= -250 	#0 50 100 #
%-aug: production = 1        ## 1 means production, 0 means comprehension
# %-aug: elec_list = 'new_path'   ## 'old_path'|'new_path'|'erp-corr'
%-aug: srm_k = 5  ## number of srm features
%-aug: num_random_run = 10

%-aug: zero_shot = --zero_shot
%-aug: folds = 10

%-aug: CMD = sbatch --job-name=pooling-$(shell date +%Y%m%d-%H%M%S)-sub-size submit.sh
# %-aug: CMD = python 
# %-aug: CMD = echo


srm-aug:

	for i in $(data_size); do \
		$(CMD) $(PROJECT_DIR)/low_data_regime.py \
		--subject $(subjects) \
		--production $(production) \
		--lags $(lags) \
		--srm_k $(srm_k) \
		--folds $(folds) \
		--num_random_run $(num_random_run) \
		$(if $(zero_shot),--zero_shot) \
		--data_size $$i; \
	done



%-pooling: data_size ?=  100 500 1000 2500 5000 10000 'all'	 ## 100 500 1000 2500 5000 10000 'all'

%-pooling: production = 1        ## 1 means production, 0 means comprehension

%-pooling: srm_k = 5  ## number of srm features
%-pooling: num_random_run = 10
# %-pooling: zero_shot = --zero_shot
%-pooling: CMD = python 
# %-aug: CMD = echo
# %-srm: JOB_NAME = $(subst /,-,$(desired_fold))
# %-srm: CMD = sbatch --job-name=$(production)-$(JOB_NAME)-across submit.sh


plotting-pooling:

	
		$(CMD) $(PROJECT_DIR)/plotting_pooling.py \
		--production $(production) \
		--srm_k $(srm_k) \
		--num_random_run $(num_random_run) \
		$(if $(zero_shot),--zero_shot) \
		--data_size $(data_size); \
