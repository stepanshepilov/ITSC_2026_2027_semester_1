# 3. Практика и MLOps недели

## 3.1. Цель эксперимента

Реализовать на PyTorch два минимальных агента:

- DQN: MLP, replay buffer, Huber loss, target network, epsilon-greedy;
- REINFORCE: categorical policy, Monte Carlo returns, опциональный baseline.

Основная среда для smoke-теста — `CartPole-v1`. После проверки можно перенести конфигурацию на `Acrobot-v1` и `LunarLander-v3`, если установлены соответствующие зависимости.

## 3.2. Структура

```text
week_2/
  01_dqn.md
  02_policy_gradient.md
  03_practice_and_mlops.md
  week_2_deep_rl.ipynb
  configs/
  artifacts/
  reports/
```

## 3.3. Минимальная конфигурация

```yaml
env:
  id: CartPole-v1
  max_episode_steps: 500

training:
  seed: 42
  episodes: 300
  gamma: 0.99
  learning_rate: 0.001
  batch_size: 64
  replay_capacity: 10000
  warmup_steps: 500
  target_update_frequency: 100
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  use_baseline: true
```

Параметры должны передаваться в функции обучения, а не быть разбросаны по notebook. Для быстрого smoke-теста уменьшите `episodes`, `warmup_steps` и `replay_capacity`.

## 3.4. Порядок работы

1. Проверить shape observation/action space и seed среды.
2. Запустить random policy и убедиться, что episode завершается.
3. Обучить DQN на короткой конфигурации.
4. Обучить REINFORCE без baseline.
5. Повторить REINFORCE с baseline и сравнить variance returns.
6. Сравнить DQN и REINFORCE на одинаковых seeds.
7. Выполнить абляции и сохранить таблицу результатов.

## 3.5. Абляции

Минимальная матрица:

| Абляция | Варианты | Что измеряем |
|---|---|---|
| Replay capacity | 1 000 / 10 000 / 50 000 | скорость и устойчивость DQN |
| Target update | 10 / 100 / 1 000 steps | колебания Q-values и return |
| Baseline | без / с value baseline | std policy loss и return |

Каждый вариант запускайте минимум на 3 seeds. Для отчёта показывайте mean и standard deviation, а не только лучший запуск.

## 3.6. Evaluation и checkpoints

Разделяйте `train` и `evaluate`: во время evaluation для DQN используйте greedy action, а для REINFORCE либо greedy `argmax`, либо заранее выбранный stochastic protocol. Сохраняйте:

- `state_dict` модели;
- конфигурацию и seed;
- имя среды и версии библиотек;
- историю return;
- метрики evaluation.

Чекпоинт можно принять как лучший по moving average evaluation return, но критерий должен быть объявлен до запуска.

## 3.7. CLI и CI

Рекомендуемый интерфейс:

```bash
python train.py --algorithm dqn --env CartPole-v1 --config configs/cartpole.yaml
python evaluate.py --checkpoint artifacts/dqn.pt --episodes 20
python plot.py --input reports/results.csv
```

Smoke-test должен проверять, что короткое обучение завершается, файл checkpoint создаётся и загружается, а evaluation возвращает числовой mean return. В CI не следует требовать высокой reward: проверяйте контракт запуска, формы тензоров и отсутствие NaN.

## 3.8. Критерии готовности

- обе реализации обучаются и сохраняют checkpoint;
- DQN использует replay buffer и отдельную target network;
- REINFORCE умеет работать с baseline через параметр;
- evaluation не обучает модель и отключает epsilon exploration;
- есть график mean return по seeds;
- проведены три заданные абляции или явно зафиксирована причина пропуска;
- notebook запускается сверху вниз после установки зависимостей.

## 3.9. Вопросы для отчёта

1. В каком диапазоне параметров DQN наиболее чувствителен к target update frequency?
2. Насколько baseline уменьшил дисперсию policy loss?
3. Как изменилось поведение при переходе от CartPole к LunarLander?
4. Какой алгоритм лучше использует ограниченный budget взаимодействий и почему?
