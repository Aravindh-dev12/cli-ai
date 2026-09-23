.PHONY: up local-up local-down status decision-health owned-train owned-evaluate owned-benchmark owned-calibrate owned-teacher-label owned-distill rlcd-train frontier-train frontier-rlcd frontier-evaluate frontier-export hf-upload owned-register owned-promote logs test test-ai test-backend test-frontend smoke samples batch

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
	docker compose run --rm ai-service python -m ml.evaluate_owned --checkpoint /cache/clinevo-owned/latest.pt --size ${OWNED_EVAL_SIZE:-256}

signal-gan:
	docker compose run --rm ai-service python -m ml.signal_gan --epochs ${GAN_EPOCHS:-10} --batch-size ${GAN_BATCH_SIZE:-32}

owned-benchmark:
	docker compose run --rm ai-service python -m ml.benchmark --checkpoint /cache/clinevo-owned/latest.pt --size $${OWNED_BENCHMARK_SIZE:-256}

owned-calibrate:
	docker compose run --rm ai-service python -m ml.calibrate --checkpoint /cache/clinevo-owned/latest.pt --size ${OWNED_CALIBRATE_SIZE:-256}

owned-teacher-label:
	docker compose run --rm ai-service python -m ml.teacher_label --teachers ${OWNED_TEACHERS:-laya} --size ${OWNED_TEACHER_SIZE:-256}

owned-distill:
	docker compose run --rm ai-service python -m ml.distill --teacher-jsonl /cache/clinevo-owned/teacher-decisions.jsonl --output /cache/clinevo-owned/distilled.pt --epochs ${OWNED_DISTILL_EPOCHS:-2} --temperature ${OWNED_DISTILL_TEMPERATURE:-2.0} --lora

rlcd-train:
	docker compose run --rm ai-service python -m ml.rlcd_train --size ${RLCD_SIZE:-512} --epochs ${RLCD_EPOCHS:-1} --init /cache/clinevo-owned/latest.pt --output /cache/clinevo-owned/rlcd.pt

frontier-train:
	docker compose run --rm ai-service python -m ml.train_frontier --size ${FRONTIER_TRAIN_SIZE:-256} --steps ${FRONTIER_TRAIN_STEPS:-500} --output /cache/clinevo-owned/frontier.pt

frontier-rlcd:
	docker compose run --rm ai-service python -m ml.frontier_rlcd --size ${FRONTIER_RLCD_SIZE:-512} --epochs ${FRONTIER_RLCD_EPOCHS:-1} --init /cache/clinevo-owned/frontier.pt --output /cache/clinevo-owned/frontier-rlcd.pt

frontier-evaluate:
	docker compose run --rm ai-service python -m ml.evaluate_frontier --checkpoint /cache/clinevo-owned/frontier.pt --size ${FRONTIER_EVAL_SIZE:-128}

frontier-export:
	docker compose run --rm ai-service python -m ml.export_frontier --checkpoint /cache/clinevo-owned/frontier.pt --output /cache/clinevo-owned/hub

hf-upload:
	docker compose run --rm ai-service python -m ml.hf_upload --repo-id ${HF_REPO_ID:-Aravindhan11/cli-ai} --folder /cache/clinevo-owned/hub

owned-register:
	docker compose run --rm ai-service python -m ml.mlops register --checkpoint /cache/clinevo-owned/latest.pt --metrics /cache/clinevo-owned/metrics.json --dataset-version $${OWNED_DATASET_VERSION:-synthetic-v1} --model-version $${OWNED_MODEL_VERSION:-latest}

owned-promote:
	docker compose run --rm ai-service python -m ml.mlops promote --model-version ${OWNED_MODEL_VERSION:-latest} --min-accuracy ${OWNED_MIN_ACCURACY:-0.80} --min-macro-f1 ${OWNED_MIN_MACRO_F1:-0.80} --max-ece ${OWNED_MAX_ECE:-0.15} --max-p95-ms ${OWNED_MAX_P95_MS:-1500}
