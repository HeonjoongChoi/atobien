build:
	docker build --no-cache --tag user_rest_api:0.1 .

start-api:
	docker run -d --name user_rest_api -p 58027:8000 user_rest_api:0.1
