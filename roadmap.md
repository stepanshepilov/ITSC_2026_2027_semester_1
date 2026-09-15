### ФАЗА 1: Классический и базовый Deep RL (Недели 1-2)

*Цель: пройти путь от MDP и табличных методов до нейросетевых value- и policy-based агентов, сохранив понимание того, что именно оптимизируется.*

**Неделя 1: MDP, Dynamic Programming и Model-Free RL**

- [ ] **Теория:** MDP, состояния, действия, награды, переходы, policy, value/Q-function, уравнение Беллмана, Monte Carlo, TD-learning, exploration vs exploitation, on-policy и off-policy.
- [ ] **Проект:** реализовать на NumPy настраиваемый `GridWorld` с детерминированными и стохастическими переходами, терминальными состояниями и штрафом за шаг. Написать `Policy Iteration`, `Value Iteration`, `SARSA` и `Q-Learning`, затем сравнить policy, sample efficiency и устойчивость к шуму.
- [ ] **Среды:** собственный `GridWorld`, `FrozenLake-v1` с несколькими значениями `slippery`, `Taxi-v3` и `CliffWalking-v1` из `Gymnasium`.
- [ ] **MLOps:** оформить пакет с конфигурацией эксперимента в YAML, фиксировать seed, версии Python-зависимостей и гиперпараметры. Логировать среднюю награду, длину эпизода, epsilon и success rate в W&B или MLflow; сохранить лучшую policy как артефакт и добавить smoke-тест для одного короткого запуска.

**Неделя 2: DQN и Policy Gradient в одном сравнительном эксперименте**

- [ ] **Теория:** аппроксимация Q-функции, replay buffer, target network, Huber loss, target leakage, Policy Gradient Theorem, REINFORCE, baseline и variance reduction.
- [ ] **Проект:** написать на PyTorch минимальный DQN с воспроизводимым replay buffer и REINFORCE с categorical policy. Обучить DQN на `CartPole-v1`, `Acrobot-v1` и `LunarLander-v3`, а REINFORCE сравнить с DQN на `CartPole-v1` и `LunarLander-v3`. Провести абляции: размер buffer, частота обновления target network и наличие baseline.
- [ ] **Среды:** `CartPole-v1` для быстрой отладки, `Acrobot-v1` для дискретного управления и `LunarLander-v3` для более сложного reward shaping.
- [ ] **MLOps:** сделать CLI вида `train`, `evaluate`, `plot`; сохранять checkpoints, конфиги и кривые обучения. Добавить evaluation-скрипт на фиксированном наборе seeds, registry моделей и CI-проверку, которая запускает короткое обучение и проверяет, что checkpoint можно загрузить.

---

### ФАЗА 2: Практичные Deep RL-алгоритмы (Недели 3-4)

*Цель: освоить алгоритмы, которые регулярно встречаются в production- и research-системах, и научиться оценивать их инженерные компромиссы.*

**Неделя 3: Actor-Critic и PPO**

- [ ] **Теория:** A2C/A3C, advantage function, GAE, entropy bonus, trust region, clipping, on-policy data и проблема распределённого сбора траекторий.
- [ ] **Проект:** реализовать учебный A2C, затем использовать проверенную реализацию PPO из `CleanRL` или `Stable-Baselines3`. Сначала пройти `CartPole-v1`, затем обучить агента на `BipedalWalker-v3`; сравнить влияние GAE lambda, clip range, entropy coefficient и числа environments.
- [ ] **Среды:** `CartPole-v1` для проверки loss и advantage, `BipedalWalker-v3` для непрерывного управления, опционально `LunarLanderContinuous-v3` для второго benchmark.
- [ ] **MLOps:** вести experiment matrix по гиперпараметрам, логировать policy loss, value loss, entropy, explained variance и episode return. Добавить автоматическую оценку после тренировки, хранение видео эпизодов и threshold-gate: модель публикуется только при прохождении заданного среднего return.

**Неделя 4: SAC и TD3 для непрерывного управления**

- [ ] **Теория:** deterministic policy gradient, target policy smoothing, delayed actor updates, entropy-regularized objective, temperature alpha, off-policy replay и reward scaling.
- [ ] **Проект:** изучить и запустить SAC и TD3 в `Stable-Baselines3`, а затем реализовать небольшой критик/actor training loop самостоятельно. Сравнить sample efficiency, стабильность и чувствительность к seed на `Pendulum-v1` и `HalfCheetah-v5` (или `Ant-v5` при наличии MuJoCo).
- [ ] **Среды:** `Pendulum-v1` как быстрый sanity check, `HalfCheetah-v5` как основной benchmark, `Humanoid-v5` только как расширенный эксперимент при наличии вычислительных ресурсов.
- [ ] **MLOps:** вынести среды и алгоритмы в единый конфиг, использовать Docker для воспроизводимого запуска, сохранять replay/checkpoint metadata и системные метрики. Подготовить отчёт с confidence intervals по нескольким seeds и небольшой inference API, который возвращает действие для заданного observation.

---

### ФАЗА 3: RL4LLM (Недели 5-7)

*Цель: понять, как preference optimization и reward-driven обучение применяются к языковым моделям, и собрать воспроизводимый pipeline на малой модели.*

**Неделя 5: Preference data, Reward Modeling и DPO**

- [ ] **Теория:** preference pairs, reference model, KL-регуляризация, reward model, RLHF pipeline, DPO objective и отличие offline preference optimization от PPO.
- [ ] **Проект:** взять небольшую instruction-модель, например `Qwen/Qwen2.5-0.5B-Instruct`, подготовить небольшой поднабор `Anthropic/hh-rlhf` или `HuggingFaceH4/ultrafeedback_binarized`, обучить reward model на pairwise loss и выполнить DPO через `TRL`. Дообучение должно поддерживать LoRA/QLoRA.
- [ ] **Среды и данные:** локальный CPU smoke-тест на 20-50 примерах, затем GPU-эксперимент на 1-5 тысячах пар; отдельный validation split и фиксированный набор prompts для качественной проверки.
- [ ] **MLOps:** версионировать dataset snapshot, tokenizer и LoRA adapter; логировать длины ответов, reward margin, train/eval loss и KL к reference model. Собирать модель и конфиг в один артефакт, добавить проверку утечки validation prompts и таблицу before/after evaluation.

**Неделя 6: KTO/ORPO и оценка качества LLM**

- [ ] **Теория:** обучение на positive/negative feedback без обязательных preference pairs, KTO loss, ORPO, alignment tax, reward hacking, length bias и ограничения автоматических LLM-as-a-judge оценок.
- [ ] **Проект:** на одном и том же SFT checkpoint сравнить DPO, KTO и ORPO на компактном датасете предпочтений. Провести controlled ablation по beta/learning rate и проверить поведение на полезности, отказах от небезопасных запросов, длине ответа и повторяемости.
- [ ] **Среды и данные:** `TRL` с Hugging Face `transformers`, `datasets`, `peft` и `accelerate`; для оценки использовать отдельный набор prompts, held-out pairwise preferences и небольшой ручной blind review.
- [ ] **MLOps:** организовать pipeline `prepare-data -> train -> evaluate -> register`, сохранять lineage датасета и хеш конфигурации. Публиковать сравнительный отчёт, latency/VRAM budget и regression gate, который блокирует модель при падении базовых метрик.

**Неделя 7: PPO для LLM, GRPO и production inference**

- [ ] **Теория:** token-level reward, KL penalty, value head, rollout generation, PPO instability, group-relative advantages в GRPO, verifiable rewards и отличие reasoning/RL от обычного SFT.
- [ ] **Проект:** собрать небольшой эксперимент с PPO или GRPO на задачах с проверяемым ответом: arithmetic, GSM8K-подмножество или генерация структурированного JSON. Сначала обучить reward/verifier function, затем сравнить SFT, DPO и PPO/GRPO на held-out задачах; не использовать большую модель без необходимости.
- [ ] **Среды и данные:** `TRL`/`accelerate`, локальный verifier для арифметики или JSON Schema, модель 0.5B-1.5B с LoRA. Зафиксировать лимит токенов, batch size и бюджет GPU, чтобы эксперимент можно было повторить.
- [ ] **MLOps:** упаковать inference в FastAPI или vLLM-compatible сервис, измерить latency, throughput и memory usage, добавить health endpoint и structured logging. Вести evaluation dashboard, хранить prompts/responses с обезличиванием и сделать rollback на предыдущий adapter.

---

### ФАЗА 4: Финальный проект по RL (Недели 8-10)

**Неделя 8: Постановка задачи и baseline**

- [ ] **Проект:** выбрать прикладной сценарий с наблюдаемой средой и ограничениями: оптимизация очереди/ресурсов, управление запасами, маршрутизация, bidding или собственная симуляция. Описать MDP, reward, ограничения, business KPI и offline/online режим.
- [ ] **Практика:** создать Gymnasium-compatible environment, валидатор observation/action spaces, deterministic baseline и random policy. Сгенерировать train/validation/test scenarios, отдельно проверить отсутствие leakage между ними.
- [ ] **MLOps:** создать репозиторий эксперимента с `pyproject.toml`, Dockerfile, pre-commit, unit/integration tests, конфигами Hydra или YAML, DVC/хешированием данных и W&B/MLflow tracking. Зафиксировать Definition of Done и бюджет вычислений.

**Неделя 9: Обучение, сравнение и устойчивость**

- [ ] **Проект:** обучить минимум два подходящих алгоритма, например PPO против SAC/TD3 для continuous control или DQN против PPO для discrete control. Провести минимум 5 seeds, ablation reward design и проверку на изменённых условиях среды, включая perturbation и worst-case scenarios.
- [ ] **Практика:** сравнить с heuristic и supervised/imitation baseline, построить learning curves с доверительными интервалами, оценить sample efficiency, constraint violations, latency и стоимость одного эпизода.
- [ ] **MLOps:** реализовать автоматический training job, checkpoint selection, model registry, evaluation report и quality gate. Добавить reproducibility test из чистого Docker-окружения и отдельный скрипт для загрузки зарегистрированной модели.

**Неделя 10: Доставка, мониторинг и защита проекта**

- [ ] **Проект:** подготовить inference service или batch decision pipeline, replay demo и технический отчёт: постановка, reward design, алгоритмы, эксперименты, ограничения и план дальнейшего улучшения. Показать failure cases и объяснить, когда policy нельзя применять.
- [ ] **Практика:** провести shadow/offline evaluation на новых сценариях, нагрузочный тест сервиса и rollback к предыдущему checkpoint. Добавить экспорт метрик Prometheus-compatible формата или эквивалентный dashboard.
- [ ] **MLOps:** оформить CI/CD workflow: lint, tests, environment build, training smoke-test, evaluation gate и публикация артефактов. Финальный результат должен запускаться одной командой, иметь README с командами воспроизведения и содержать версию модели, данных и кода.

---