%.png: %.dot
	sfdp -Tpng -Gsize=20,20\! -Gdpi=300 -Gnodesep=10 -Nfontsize=40 -Goverlap=scale $< -o $@

all: kraje.patent.png cpc.patent.png

cpc.png: cpc.dot
	twopi -Tpdf -Gsize=8,8 -Nfontsize=40 -Goverlap=scale cpc.dot -o cpc.pdf

clean:
	rm kraje.patent.png
	rm cpc.patent.png



PLG_PATH = plgkrusek@prometheus.cyfronet.pl:/net/scratch/people/plgkrusek/patenty


to_plg_data:
	#rsync -av --inplace do_modelu_grawitacyjnego.csv $(PLG_PATH)/
	rsync -av dane $(PLG_PATH)/
	
to_plg_code:
	rsync -av --inplace *.sh code Makefile $(PLG_PATH)

to_plg: to_plg_data to_plg_code

from_plg:
	rsync -av --inplace --exclude '*.out' --exclude '*.err' '$(PLG_PATH)/checkpoint' plg/


3comp_others:
	mkdir -p gen/$@
	python3 code/gravity.py \
		--nboot 1 \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--notreinablezero

mcmc:
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 50000 \
		--num_burnin_steps 90000
# Małe c0
mcmc2:
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 20000 \
		--num_burnin_steps 20000

# Lepsza inicjalizacja w
mcmc3:
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 16000 \
		--num_burnin_steps 16000

test_mcmc:
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4 \
		--num_chains 2 \
		--num_adaptation 16 \
		--num_burnin_steps 16 \
		--priorsample gen/mcmc3/samples.pkl \
		--toyear 2015

paper:
	mkdir -p gen/$@
	python3 code/results.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--paperdir gen/$@ \
		--mcmcpickle gen/mcmc3/samples.pkl


# ANlizy czasowe


#y2015:
#	mkdir -p gen/$@
#	python3 code/bayes.py \
#		--pickle dane/clean.pickle \
#		--out gen/$@ \
#		--others \
#		--nnz 2 \
#		--feature_type ALL \
#		--num_results 4000 \
#		--num_chains 16 \
#		--num_adaptation 16000 \
#		--num_burnin_steps 16000 \
#		--toyear 2015
#
#
#y2016: y2015
#	mkdir -p gen/$@
#	python3 code/bayes.py \
#		--pickle dane/clean.pickle \
#		--out gen/$@ \
#		--others \
#		--nnz 2 \
#		--feature_type ALL \
#		--num_results 4000 \
#		--num_chains 16 \
#		--num_adaptation 16000 \
#		--num_burnin_steps 16000 \
#		--priorsample gen/y2015/samples.pkl \
#		--toyear 2016
#
#y2017: y2016
#	mkdir -p gen/$@
#	python3 code/bayes.py \
#		--pickle dane/clean.pickle \
#		--out gen/$@ \
#		--others \
#		--nnz 2 \
#		--feature_type ALL \
#		--num_results 4000 \
#		--num_chains 16 \
#		--num_adaptation 16000 \
#		--num_burnin_steps 16000 \
#		--priorsample gen/y2016/samples.pkl \
#		--toyear 2017
y2018:
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 16000 \
		--num_burnin_steps 16000 \
		--toyear 2018 \
		--loglambda_zero -12.0

y2019: y2018
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 16000 \
		--num_burnin_steps 16000 \
		--priorsample gen/y2018/samples.pkl \
		--toyear 2019

y2020: y2019
	mkdir -p gen/$@
	python3 code/bayes.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--num_results 4000 \
		--num_chains 16 \
		--num_adaptation 16000 \
		--num_burnin_steps 16000 \
		--priorsample gen/y2019/samples.pkl \
		--toyear 2020

paper2020:
	mkdir -p gen/$@
	python3 code/results.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--paperdir gen/$@ \
		--mcmcpickle gen/y2020/samples.pkl \
		--toyear 2020

paper2019:
	mkdir -p gen/$@
	python3 code/results.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--paperdir gen/$@ \
		--mcmcpickle gen/y2019/samples.pkl \
		--toyear 2019
paper2018:
	mkdir -p gen/$@
	python3 code/results.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--paperdir gen/$@ \
		--mcmcpickle gen/y2018/samples.pkl \
		--toyear 2018
paper2017:
	mkdir -p gen/$@
	python3 code/results.py \
		--pickle dane/clean.pickle \
		--out gen/$@ \
		--others \
		--nnz 2 \
		--feature_type ALL \
		--paperdir gen/$@ \
		--mcmcpickle gen/y2017/samples.pkl \
		--toyear 2017

time_evolution: paper2018 paper2019 paper2020
	echo "time_evolution"


trend:
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py --pickle dane/clean.pickle --nkl 16384 --steps 800 --seed 127445 #792848 # --nocoldstart

trend_local:
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py --pickle dane/clean.pickle --nkl 120 --steps 500 --seed 22

gen/trends/% :
	mkdir -p $@
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py \
		--pickle dane/clean.pickle \
		--nkl 16384 \
		--steps 2000 \
		--seed $* \
		--out $@

gen/trends/paper :
	mkdir -p $@
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py \
		--pickle dane/clean.pickle \
		--nkl 16384 \
		--steps 1000 \
		--seed 792848 \
		--out $@

gen/trends/review :
	mkdir -p $@
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py \
		--pickle dane/clean_update.pickle \
		--nkl 16384 \
		--steps 1000 \
		--seed 792848 \
		--out $@

gen/trends/review2 :
	mkdir -p $@
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py \
		--pickle dane/clean_update.pickle \
		--nkl 16384 \
		--steps 1000 \
		--seed 792848 \
		--train_test_date="2021-04-01" \
		--out $@

gen/trends/review3 :
	mkdir -p $@
	PYTHONPATH=${PYTHONPATH} python3 code/logistic_growth.py \
		--pickle dane/clean_update.pickle \
		--nkl 16384 \
		--steps 1000 \
		--seed 792848 \
		--train_test_date="2021-09-01" \
		--out $@

seeded_trends: gen/trends/982995 gen/trends/127445 gen/trends/635725 gen/trends/792848 gen/trends/16917 gen/trends/773737 gen/trends/979000  gen/trends/318589

paper_trends: gen/trends/paper
review_trends: gen/trends/review gen/trends/review2 gen/trends/review3

from_plg_trend:
	rsync -av --inplace  '$(PLG_PATH)/gen/trends' plg/