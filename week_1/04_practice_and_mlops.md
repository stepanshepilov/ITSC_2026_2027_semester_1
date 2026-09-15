# 4. Практика недели и MLOps-контур

## 4.1. Рекомендуемый порядок работы

1. Реализовать `GridWorld` и проверить его вручную на 2–3 состояниях.
2. Запустить Value Iteration и Policy Iteration на детерминированной среде.
3. Добавить случайные переходы и проверить чувствительность к `slip_probability`.
4. Реализовать SARSA и Q-Learning с epsilon-greedy.
5. Сравнить алгоритмы на `FrozenLake-v1`, `Taxi-v3` и `CliffWalking-v1`.
6. Повторить каждый эксперимент на нескольких seeds.

## 4.2. Минимальная структура эксперимента

```text
week_1/
  01_mdp_and_bellman.md
  02_dynamic_programming.md
  03_model_free_rl.md
  04_practice_and_mlops.md
  week_1_rl_basics.ipynb
  configs/
    gridworld.yaml
  artifacts/
  reports/
```

## 4.3. Конфигурация

Не зашивайте параметры в код алгоритма. Минимальный конфиг должен содержать:

```yaml
env:
  rows: 5
  cols: 5
  slip_probability: 0.0
  step_reward: -0.04
  goal_reward: 1.0

training:
  gamma: 0.95
  alpha: 0.1
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995
  episodes: 3000
  seed: 42
```

## 4.4. Воспроизводимость

- фиксируйте seed Python, NumPy и среды;
- сохраняйте конфиг рядом с результатом;
- записывайте версии зависимостей;
- отделяйте обучение от evaluation;
- используйте несколько seeds для итогового сравнения;
- сохраняйте `Q-table` и policy как артефакты.

## 4.5. Критерии готовности

Работа считается завершённой, если:

- policy визуально отображается в GridWorld;
- Value Iteration и Policy Iteration находят маршрут к цели;
- SARSA и Q-Learning обучаются на малой среде;
- результаты можно повторить одной командой с конфигом;
- есть таблица или график среднего return по seeds;
- короткий smoke-test проверяет запуск обучения и загрузку артефакта.

## 4.6. Вопросы для отчёта

1. Почему SARSA может выбрать более безопасный маршрут, чем Q-Learning?
2. Как меняется policy при увеличении штрафа за шаг?
3. Что происходит с Q-values при слишком большом `alpha`?
4. Почему один seed недостаточен для вывода о качестве алгоритма?
