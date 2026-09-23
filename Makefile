.PHONY: up local-up local-down status decision-health owned-train owned-evaluate logs test test-ai test-backend test-frontend smoke samples batch

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=200

test: test-ai test-backend test-frontend

test-ai:
	cd ai-service && pytest -q tests

test-backend:
	cd backend && mvn -B test

test-frontend:
	cd frontend && npm run build && npm test -- --no-progress

smoke:
	bash scripts/run_compose_smoke.sh

samples:
	python samples/scripts/generate_corpus.py

batch:
	python samples/scripts/run_batch.py --url http://localhost:8000

local-up:
	bash scripts/clinevo-local.sh up

local-down:
	bash scripts/clinevo-local.sh down

status:
	python tools/clinevo.py status

decision-health:
	curl -fsS http://localhost:8000/decision/health


owned-train:
	docker compose run --rm ai-service python -m ml.train_owned --size $${OWNED_TRAIN_SIZE:-128} --epochs $${OWNED_TRAIN_EPOCHS:-2}

owned-evaluate:
	docker compose run --rm ai-service python -m ml.evaluate_owned --checkpoint /cache/clinevo-owned/latest.pt --size $${OWNED_EVAL_SIZE:-256}
