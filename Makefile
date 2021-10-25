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
		--priorsample gen/mcmc3/samples.pkl

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
