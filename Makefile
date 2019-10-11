.RECIPEPREFIX +=

PYTHON=python3
ROOT=data/scvd
TRAINDATA=$(ROOT)/train
VALDATA=$(ROOT)/test
TESTDATA=$(ROOT)/test
EPOCH=100

CHECKPOINT=weight/checkpoint_35.pth

main: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --debug

resume: 
        $(PYTHON) main.py $(TRAINDATA) $(VALDATA) --dataset-root $(ROOT) --resume $(CHECKPOINT) --debug

evaluate: 
        $(PYTHON) evaluate.py $(VALDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split val

test: 
        $(PYTHON) evaluate.py $(TESTDATA) --dataset-root $(ROOT) --checkpoint $(CHECKPOINT) --split test
