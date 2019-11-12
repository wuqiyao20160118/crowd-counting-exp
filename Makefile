.RECIPEPREFIX +=

PYTHON=python3
ROOT=../crowd-counting-revise/data
TRAINDATA=$(ROOT)/scvd_processed/train
VALDATA=$(ROOT)/scvd_processed/val
TESTDATA=$(ROOT)/scvd/test
TRAINXML=$(ROOT)/scvd/train
VALXML=$(ROOT)/scvd/test
TESTXML=$(ROOT)/scvd/test
EPOCH=100

CHECKPOINT=weight_all/checkpoint_60.pth

main: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --debug

resume: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --debug

evaluate: 
        $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT)

test: 
        $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT)

main_scale:
        $(PYTHON) main_scale.py $(TRAINDATA) $(VALDATA) $(TRAINXML) $(VALXML) --dataset-root $(ROOT)

resume_scale:
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) $(TRAINXML) $(VALXML) --dataset-root $(ROOT) --resume $(CHECKPOINT)

test_scale:
        $(PYTHON) evaluate.py $(TESTDATA) $(TESTXML) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT)