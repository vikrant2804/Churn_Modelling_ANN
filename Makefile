train:
	python src/train.py

evaluate:
	python src/evaluate.py

test:
	pytest tests/

docker-build:
	docker build -t mlops-churn .

docker-run:
	docker run mlops-churn