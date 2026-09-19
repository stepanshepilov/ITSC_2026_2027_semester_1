# 2. REINFORCE и Policy Gradient

## 2.1. Политика как распределение

Policy-based агент не строит Q-table. Он параметризует распределение действий:

$$\pi_\theta(a \mid s).$$

Для дискретных действий нейросеть выдаёт logits, а `Categorical(logits=...)` превращает их в вероятности. Действие сэмплируется из распределения, поэтому policy естественно поддерживает exploration.

## 2.2. Policy Gradient Theorem

Цель — максимизировать ожидаемый discounted return:

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0].$$

Градиент можно оценить через log-derivative trick:

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\sum_t \nabla_\theta \log \pi_\theta(a_t\mid s_t) G_t\right].$$

На практике минимизируют loss:

$$L_{PG} = -\sum_t \log \pi_\theta(a_t\mid s_t) G_t.$$

REINFORCE собирает полный эпизод, вычисляет returns и делает обновление после его завершения. Метод прост и unbiased при корректной выборке, но его оценки часто имеют большую дисперсию.

## 2.3. Discounted returns

Для шага `t`:

$$G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \dots.$$

При `gamma` близком к 1 агент сильнее учитывает далёкие последствия. В учебной реализации удобно вычислять returns справа налево и затем нормализовать их внутри эпизода или batch:

```python
returns = (returns - returns.mean()) / (returns.std() + 1e-8)
```

Нормализация не меняет ожидаемое направление градиента, но уменьшает численную нестабильность.

## 2.4. Baseline и advantage

К return можно вычесть любую функцию состояния `b(s)`, не меняя ожидаемый градиент:

$$\mathbb{E}[\nabla \log \pi(a\mid s)b(s)] = 0.$$

Простейший baseline — средний return batch. Более полезный вариант — обучаемая value network `V_phi(s)`:

$$A_t = G_t - V_\phi(s_t).$$

Тогда policy loss:

$$L_\pi = -\sum_t \log \pi_\theta(a_t\mid s_t) \operatorname{stopgrad}(A_t).$$

Важно не пропускать policy gradient через value network: advantage для actor обновления обычно отделяют `detach()`.

## 2.5. Entropy bonus

Entropy поощряет не слишком раннее схлопывание policy в одно действие:

$$H(\pi(\cdot\mid s)) = -\sum_a \pi(a\mid s)\log\pi(a\mid s).$$

Итоговый loss может быть записан как:

$$L = L_\pi + c_v L_V - \beta H.$$ 

Для чистого REINFORCE value loss и entropy bonus можно включать по отдельности. В этом материале baseline сравнивается как контролируемая абляция.

## 2.6. DQN против REINFORCE

| Свойство | DQN | REINFORCE |
|---|---|---|
| Тип | value-based, off-policy | policy-based, on-policy |
| Опыт | replay buffer переиспользуется | эпизод используется один раз |
| Action space | дискретный | дискретный или непрерывный при другой policy distribution |
| Target | TD bootstrap | полный Monte Carlo return |
| Главный риск | нестабильные Q-targets | высокая дисперсия градиента |
| Exploration | epsilon-greedy | sampling из policy |

Нельзя сравнивать только одну seed или один последний эпизод. Нужны несколько запусков, moving average и фиксированный evaluation protocol.

## 2.7. Вопросы

1. Почему REINFORCE не может обновить policy до конца эпизода?
2. Как baseline уменьшает variance, но не меняет expected gradient?
3. Что происходит при слишком большом entropy coefficient?
4. Почему normalized returns полезны, но не заменяют правильный reward design?
