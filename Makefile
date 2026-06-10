.PHONY: help build up down logs clean deploy backup

help:
	@echo "Available commands:"
	@echo "  make build    - Build Docker images"
	@echo "  make up       - Start all services"
	@echo "  make down     - Stop all services"
	@echo "  make logs     - View logs"
	@echo "  make clean    - Remove containers and volumes"
	@echo "  make deploy   - Deploy to VPS"
	@echo "  make backup   - Backup data"

build:
	docker-compose build

up:
	docker-compose up -d
	@echo "Services started. Monitor with: make logs"

down:
	docker-compose down

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	docker system prune -f

deploy:
	@echo "Deploying to VPS..."
	rsync -avz --exclude 'data/*.csv' --exclude '__pycache__' --exclude '*.pyc' \
		-e "ssh" ./ root@YOUR_VPS_IP:/root/federated-learning/
	ssh root@YOUR_VPS_IP "cd /root/federated-learning && docker-compose up -d --build"

backup:
	@mkdir -p backup
	tar -czf backup/fl_backup_$(shell date +%Y%m%d_%H%M%S).tar.gz \
		--exclude='backup' \
		--exclude='data/*.csv' \
		.
	@echo "Backup created in backup/ directory"