.PHONY: deps test train play play-gui play-human

LEVEL="snakeai/levels/10x10-blank.json"

deps:
	python3.6 -m pip install --upgrade -r requirements.txt

test:
	PYTHONPATH=$(PYTHONPATH):. py.test snakeai/tests

train:
	./train.py --level "snakeai/levels/10x10-blank.json" --num-episodes 1000

play:
	./play.py --interface cli --level "snakeai/levels/10x10-blank.json" --num-episodes 100

play-gui:
	./play.py --interface gui --level "snakeai/levels/10x10-blank.json" --num-episodes 10

play-human:
	./play.py --interface gui --agent human --level "snakeai/levels/6x5-blank.json" --num-episodes 1
