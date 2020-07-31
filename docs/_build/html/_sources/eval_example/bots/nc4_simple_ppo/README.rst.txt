예제 4. Simple PPO
===================

sc2minigame에서 가장 중요한 의사결정 사항 두 가지는 다음에 어떤 유닛을 생산해야하는지, 언제 공격해야 하는지 두 가지 이다.
Simple, Simple2, 그리고 Simple3는 이 두 가지 문제를 해결할 수 있는 간단한 AI를 구현한 예이다.
Simple3는 Simple이나 Simple2에 비해 성능이 향상되었지만, 규칙을 수정해서 성능을 향상시킬 수 있는 여지가 충분히 있다.
유닛의 생산 비율을 조정하거나, 공격시점을 결정하는 규칙을 수정하는 것만으로도 승률을 높일 수 있다.
그러나, 규칙을 수정하고 평가하는 과정을 반복하는 것은 매우 번거로운 작업이고 
일정 수준이상의 성능에 도달하기 위해서는 보다 체계적이고 자동화된 방법이 필요하다.
Simple PPO는 공격시점을 결정하는 규칙을 강화학습(RL, Reinforcement Learning) 알고리즘 중 하나인 
PPO (Proximal Policy Optimization)를 사용해 개선하는 예를 보여준다.


학습 실행
-----------

.. code-block:: bash

   # Trainer 실행
   (sc2) ~/NCF2020 $ python -m python -m bots.nc4_simple_ppo.train


학습 코드를 실행하려면 우선 Traniner process를 실행시킨다.
Trainer가 성공적으로 실행되면 "READY" 메시지가 출력된 채로 대기 한다.
Traniner는 학습 알고리즘을 실행하기만 하고, 게임을 플레이해서 데이터를 생성하지 않는다.
게임을 플레이해서 데이터를 생성하는 역할을하는 Actor process를 추가로 실행해야 한다.

.. code-block:: bash

   # Actor 실행
   (sc2) ~/NCF2020 $ python -m python -m bots.nc4_simple_ppo.train --attach={trainer-ip} --n_actors=2

Actor는 Trainer와 비슷하게 실행하지만, Actor가 접속할 Trainer의 IP 주소를 argument로 같이 입력하면,
Actor모드로 실행된다.
여러 Actor가 한 Trainer에 접속해서 학습을 수행할 수 있고, 많은 Actor가 접속할 수록 학습 속도를 향상시킬 수 있다.
실질적으로는 학습에 참가한 PC 사양에 따라 네트워크(LAN), CPU, GPU등에서 심각한 병목이 발생하여 
무한히 학습 속도를 향상시키는 것은 어렵지만, 일반적인 데스크톱 PC에서는 PC 10대 정도 까지는 문제없이 성능이 향상된다.

tensorboard 출력결과는 runs 폴더에 저장되고, 학습한 인공신경망 모델은 bots/nc4_simple_ppo/model.pt 파일로 저장된다.


