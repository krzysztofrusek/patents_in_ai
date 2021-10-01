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
	rsync -av --inplace do_modelu_grawitacyjnego.csv $(PLG_PATH)/
	
to_plg_code:
	rsync -av --inplace *.sh code  $(PLG_PATH)

to_plg: to_plg_data to_plg_code

from_plg:
	rsync -av --inplace --exclude '*.out' --exclude '*.err' '$(PLG_PATH)/checkpoint' plg/
