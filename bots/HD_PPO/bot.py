
__author__ = '이다영, 박혜진'

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pathlib
import pickle
import time

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint
from sc2.position import Point2, Point3
from enum import Enum
from random import *

from .consts import CommandType, TacticStrategy

nest_asyncio.apply()


class Model(nn.Module):
    def __init__(self):
        super().__init__()
         #5가 state 개수, 12가 유닛종류(economy)
        self.fc1 = nn.Linear(5, 64)
        self.norm1 = nn.LayerNorm(64)
        self.fc2 = nn.Linear(64, 64)
        self.norm2 = nn.LayerNorm(64)
        self.vf = nn.Linear(64, 1)
        self.tactic_head = nn.Linear(64, len(TacticStrategy))

    def forward(self, x):
        x = F.relu(self.norm1(self.fc1(x)))
        x = F.relu(self.norm2(self.fc2(x)))
        value = self.vf(x)
        tactic_logp = torch.log_softmax(self.tactic_head(x), -1)
        bz = x.shape[0]
        logp = (tactic_logp.view(bz, -1, 1)).view(bz, -1)
        return value, logp

    
class NukeManager(object):
    """
    사령부 핵 담당 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.pos=0
        self.ghost_pos1_x=32.5
        self.ghost_pos2_x=95.5
        self.ghost_pos=0
        self.enemy_pos=0

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.ghost_pos = self.ghost_pos1_x
            self.enemy_pos= self.ghost_pos2_x
        else :
            self.ghost_pos = self.ghost_pos2_x
            self.enemy_pos= self.ghost_pos1_x
        
        
        

    # 마지막 명령이 발행된지 10초가 넘었으면 리워드 -1?
    # 택틱마다 일정시간 지나도 명령 발행 안되면 리워드 -1 

    async def step(self):
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.bot.get_available_abilities(cc)
        ghosts = self.bot.units(UnitTypeId.GHOST)
        
        if ghosts.amount > 0:
            ghost = ghosts.first
            if AbilityId.BUILD_NUKE in cc_abilities:
                # 전술핵 생산 가능(자원이 충분)하면 전술핵 생산
                actions.append(cc(AbilityId.BUILD_NUKE))
                print(ghosts.first.position)

            if ghost.distance_to(Point2((self.ghost_pos,55))) > 3 and self.pos == 0:
                actions.append(ghost.move(Point2((self.ghost_pos,55))))
                self.pos=1

            if ghost.distance_to(Point2((self.ghost_pos,55))) == 0 and self.pos == 1:
                print(ghost.position)
                actions.append(ghosts.first.move(Point2((self.enemy_pos,55))))
                self.pos=2

            if ghost.distance_to(Point2((self.enemy_pos,55))) < 3:
                print(ghost.position)
                self.pos=3

            if self.pos==3 :
                ghost_abilities = await self.bot.get_available_abilities(ghosts.first)
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and ghosts.first.is_idle:
                    # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                    actions.append(ghosts.first(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                    actions.append(ghosts.first(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=self.bot.enemy_start_locations[0]))
                    print("핵쏨")

        return actions


class ReconManager(object):
    """
    정찰부대 운용을 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.perimeter_radious = 10
        self.pos=0
        self.pos1_x=32.5
        self.pos2_x=95.5

    def reset(self):
        if self.bot.enemy_start_locations[0].x == 95.5:
            self.pos = self.pos1_x
        else :
            self.pos = self.pos2_x

    # def position(self):
        
    async def step(self):
        actions = list()
        recon_tag = self.bot.units.filter(
            lambda unit: unit.tag in self.bot.reconArray
        )

        for unit in recon_tag: 
            # 근처에 적들이 있는지 파악
            unit.move(Point2((self.pos,60)))
            threaten = self.bot.known_enemy_units.closer_than(
                    self.perimeter_radious, unit.position)
            
            if unit.type_id == UnitTypeId.RAVEN:
                if unit.health_percentage > 0.8 and unit.energy >= 50:
                    print("유닛오더? ",unit.orders)
                    if threaten.amount > 0: # 근처에 적이 하나라도 있으면
                        alert = 1
                        if unit.orders and unit.orders[0].ability.id != AbilityId.RAVENBUILD_AUTOTURRET:
                            closest_threat = threaten.closest_to(unit.position)
                            pos = unit.position.towards(closest_threat.position, 5)
                            pos = await self.bot.find_placement(
                                UnitTypeId.AUTOTURRET, pos)
                            order = unit(AbilityId.BUILDAUTOTURRET_AUTOTURRET, pos)
                            actions.append(order)
                '''else:
                    if unit.distance_to(self.target) > 5:
                        order = unit.move(self.target)
                        actions.append(order)'''

            elif unit.type_id == UnitTypeId.MARINE:
                if self.bot.known_enemy_units.exists:
                    enemy_unit = self.bot.known_enemy_units.closest_to(unit)
                    actions.append(unit.attack(enemy_unit))

        return actions


class CombatManager(object):
    """
    일반 전투 부대 컨트롤(공격+수비)
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
    
    def reset(self):
        self.evoked = dict()

    async def step(self):
        actions = list()

        ##-----변수, 그룹 등 선언-----
        combat_tag = self.bot.units.filter(
            lambda unit: unit.tag in self.bot.combatArray
        )
        combat_units = combat_tag.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        wounded_units = self.bot.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.bot.enemy_start_locations[0]  # 적 시작 위치

        #
        # -----유닛 명령 생성-----
        ## 마이크로 컨트롤
        #
        for unit in combat_tag:  # 건물이 아닌 유닛만 선택
            
            ##-----타겟 설정-----
            enemy_unit = self.bot.enemy_start_locations[0]
            if self.bot.known_enemy_units.exists:
                enemy_unit = self.bot.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            ##-----유닛 스킬-----

            ##-----전투 유닛 전체-----
            if unit.type_id is not UnitTypeId.MEDIVAC:
                #print(combat_units.amount)
                if combat_units.amount > 20 and self.bot.units.of_type([UnitTypeId.SIEGETANKSIEGED, UnitTypeId.SIEGETANK]).amount > 1:
                    # 전투가능한 유닛 수가 20을 넘으면 + 탱크 2대 이상 적 본진으로 공격              
                    actions.append(unit.attack(target))
                    use_stimpack = True
                    siege = False
                else:
                    # 적 사령부 방향에 유닛 집결
                    target = self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)
                    actions.append(unit.attack(target))
                    use_stimpack = False
                    siege = True

                ##-----해병과 불곰-----
                if unit.type_id in (UnitTypeId.MARINE, UnitTypeId.MARAUDER):
                    if use_stimpack and unit.distance_to(target) < 15:
                        # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                        if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                            # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                            if self.bot.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                                # 1초 이전에 스팀팩을 사용한 적이 없음
                                actions.append(unit(AbilityId.EFFECT_STIM))
                                self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.bot.time

                ##-----탱크-----
                if unit.type_id in (UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED):
                    if self.bot.known_enemy_units.exists:
                        #타겟설정: 해병과 불곰 우선
                        if self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).exists:
                            target = self.bot.known_enemy_units.of_type([UnitTypeId.MARINE, UnitTypeId.MARAUDER]).closest_to(unit)
                    
                        if siege: #다 안모여서 대기중
                            if unit.distance_to(self.bot.start_location + 0.25 * (enemy_cc.position - self.bot.start_location)) < 5:
                                if unit.type_id == UnitTypeId.SIEGETANK:
                                    order = unit(AbilityId.SIEGEMODE_SIEGEMODE)
                                    actions.append(order) 
                        else: #다모여서 돌격
                            if unit.distance_to(target) > 10 and unit.type_id == UnitTypeId.SIEGETANKSIEGED:
                                order = unit(AbilityId.UNSIEGE_UNSIEGE)
                                actions.append(order)
                                


            ##-----비전투 유닛==메디박-----
            if unit.type_id is UnitTypeId.MEDIVAC:
                ##힐을 가장 우선시
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                    #print(wounded_unit.name, "을 회복중")
                elif combat_units.amount < 1 :
                    actions.append(unit.move(self.bot.cc - 5))
                else: 
                    actions.append(unit.move(combat_units.center))
                    #print("대기중")

        # -----유닛 명령 생성 끝-----
        return actions


class MuleManager(object):
    """
    지게로봇을 이용한 사령부 자힐 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai

    async def step(self):
        actions = list()

        cc = self.bot.units(UnitTypeId.COMMANDCENTER).first
        enemy_cc = self.bot.enemy_start_locations[0]
        cc_abilities = await self.bot.get_available_abilities(cc)
        mule = self.bot.units(UnitTypeId.MULE)
        pos = cc.position.towards(enemy_cc.position, -5)

        if cc.health_percentage < 1:
            if mule.amount == 0:
                print("뮬없음")
                if AbilityId.CALLDOWNMULE_CALLDOWNMULE in cc_abilities:
                    # 지게로봇 생산가능하면 생산
                    actions.append(cc(AbilityId.CALLDOWNMULE_CALLDOWNMULE, pos))
                    print("뮬생산")
            elif mule.amount > 0:
                #actions.append(mule.first(AbilityId.REPAIR_MULE(cc)))
                actions.append(mule.first(AbilityId.EFFECT_REPAIR_MULE, cc))
                print("힐")
        else: #택틱이 뮬인데 치료할거리가 없을경우->그럴수가 있나?
            if mule.amount > 0:
                mule_unit=mule.random
                actions.append(mule_unit.move(pos))
                print("이동")

        return actions


class StepManager(object):
    """
    스텝 레이트 유지를 담당하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot = bot_ai
        self.seconds_per_step = 0.35714  # on_step이 호출되는 주기
        self.reset()
    
    def reset(self):
        self.step = -1
        # 마지막으로 on_step이 호출된 게임 시간
        self.last_game_time_step_evoked = 0.0 

    def invalid_step(self):
        """
        너무 빠르게 on_step이 호출되지 않았는지 검사
        """
        elapsed_time = self.bot.time - self.last_game_time_step_evoked
        if elapsed_time < self.seconds_per_step:
            return True
        else:
            # print(C.blue(f'man_step_time: {elapsed_time}'))
            self.step += 1
            self.last_game_time_step_evoked = self.bot.time
            return False



class RatioManager(object):
    """
    tactic에 따라 생성할 유닛 비율 결정하는 매니저
    """
    def __init__(self, bot_ai):
        self.bot= bot_ai

    def next_unit_select(self):
        """
        tactic에 따라 다음에 생산할 유닛 return
        """

        unit_counts = dict()
        
        if self.bot.tactic_strategy == TacticStrategy.ATTACK or self.bot.tactic_strategy == TacticStrategy.MULE:
            print("들어옴: 111111")
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 25,
                UnitTypeId.MARAUDER: 0,
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 5,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN: 0,
                UnitTypeId.BATTLECRUISER: 0,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.combatArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1
            
        elif self.bot.tactic_strategy == TacticStrategy.RECON:
            print("들어옴: 2222222")
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 0,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 10,
                UnitTypeId.GHOST: 0,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:0,
                UnitTypeId.BATTLECRUISER: 0,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.reconArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1

        elif self.bot.tactic_strategy == TacticStrategy.NUKE:
            print("들어옴: 333333333")
            self.target_unit_counts = {
                UnitTypeId.COMMANDCENTER: 0,  # 추가 사령부 생산 없음
                UnitTypeId.MARINE: 0,
                UnitTypeId.MARAUDER: 0, 
                UnitTypeId.REAPER: 0,
                UnitTypeId.GHOST: 3,
                UnitTypeId.HELLION: 0,
                UnitTypeId.SIEGETANK: 0,
                UnitTypeId.THOR: 0,
                UnitTypeId.MEDIVAC: 0,
                UnitTypeId.VIKINGFIGHTER: 0,
                UnitTypeId.BANSHEE: 0,
                UnitTypeId.RAVEN:0,
                UnitTypeId.BATTLECRUISER: 0,
            }
            for unit in self.bot.units:
                if unit.tag in self.bot.nukeArray: # 유닛의 태그가 어레이에 포함되어있으면
                    unit_counts[unit.type_id] = unit_counts.get(unit.type_id, 0) + 1

        

        self.evoked = dict()
        
        
        
        ### 여기서 unit_counts에 각각 다른 유닛집단 들어가야하지않나??

        target_unit_counts = np.array(list(self.target_unit_counts.values()))
        target_unit_ratio = target_unit_counts / (target_unit_counts.sum() + 1e-6)  # 목표로 하는 유닛 비율
        current_unit_counts = np.array([unit_counts.get(tid, 0) for tid in self.target_unit_counts.keys()])
        current_unit_ratio = current_unit_counts / (current_unit_counts.sum() + 1e-6)  # 현재 유닛 비율
        unit_ratio = (target_unit_ratio - current_unit_ratio).clip(0, 1)  # 목표 - 현재 유닛 비율
        
        next_unit = list(self.target_unit_counts.keys())[unit_ratio.argmax()]  # 가장 부족한 유닛을 다음에 훈련

        print("뽑힌애 : ",next_unit)
        return next_unit


class AssignManager(object):
    """
    유닛을 부대에 배치하는 매니저
    """
    def __init__(self, bot_ai, *args, **kwargs):
        self.bot = bot_ai
        

    def reset(self):
        pass

    def assign(self):


        units_tag = self.bot.units.tags #전체유닛
        #  and연산으로 살아있는 유닛으로만 구성
        self.bot.combatArray = self.bot.combatArray & units_tag 
        self.bot.reconArray = self.bot.reconArray & units_tag
        self.bot.nukeArray = self.bot.nukeArray & units_tag

        #tactic이 combat이면 combatarray에 주고 나머지엔 combat에서 쓰는게 제외하고 할당 가능?ㅇㅇ
        units_tag = units_tag - self.bot.combatArray - self.bot.reconArray - self.bot.nukeArray
        
        if self.bot.tactic_strategy == TacticStrategy.ATTACK or self.bot.tactic_strategy == TacticStrategy.MULE:
            self.bot.combatArray = self.bot.combatArray | units_tag
        elif self.bot.tactic_strategy == TacticStrategy.RECON:
            self.bot.reconArray = self.bot.reconArray | units_tag
        elif self.bot.tactic_strategy == TacticStrategy.NUKE:
            self.bot.nukeArray = self.bot.nukeArray | units_tag

# 사령부 주변에 적 없고 사령부 피 2/3 남았을때 지게로봇 소환


class Bot(sc2.BotAI):
    """
    빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    개별 전투 유닛이 적사령부에 바로 공격하는 대신, 15이 모일 때까지 대기
    """
    def __init__(self, step_interval=5.0, host_name='', sock=None):
        super().__init__()
        self.step_interval = step_interval
        self.host_name = host_name
        self.sock = sock
        if sock is None:
            try:
                self.model = Model()
                model_path = pathlib.Path(__file__).parent / 'model.pt'
                self.model.load_state_dict(
                    torch.load(model_path, map_location='cpu')
                )
            except Exception as exc:
                import traceback; traceback.print_exc() 

        self.step_manager = StepManager(self)
        self.combat_manager = CombatManager(self)
        self.assign_manager = AssignManager(self)
        self.recon_manager = ReconManager(self)
        self.nuke_manager = NukeManager(self)
        self.ratio_manager = RatioManager(self)
        self.mule_manager = MuleManager(self)
        #부대별 유닛 array
        self.combatArray = set()
        self.reconArray = set()
        self.nukeArray = set()
        

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.step_interval = self.step_interval
        self.last_step_time = -self.step_interval

        self.tactic_strategy = TacticStrategy.ATTACK

        self.cc = self.units(UnitTypeId.COMMANDCENTER).first  # 전체 유닛에서 사령부 검색

        self.step_manager.reset()
        self.combat_manager.reset()
        self.assign_manager.reset()
        self.assign_manager.assign()
        self.nuke_manager.reset()
        self.recon_manager.reset()
        self.assign_manager.assign()


        # Learner에 join
        self.game_id = f"{self.host_name}_{time.time()}"
        

    async def on_step(self, iteration: int):
        """
        :param int iteration: 이번이 몇 번째 스텝인self.assign_manager = AssignManager(self)지를 인자로 넘겨 줌

        매 스텝마다 호출되는 함수
        주요 AI 로직은 여기에 구현
        """

        if self.step_manager.invalid_step():
            return list()

        actions = list() # 이번 step에 실행할 액션 목록
        #print("1111택틱: ", self.tactic_strategy)

        if self.time - self.last_step_time >= self.step_interval:
            self.tactic_strategy= self.set_strategy()
            self.last_step_time = self.time
            print("22222택틱: ", self.tactic_strategy)
        
        

        if self.step_manager.step % 2 == 0:
            self.assign_manager.assign()
            
            next_unit = self.ratio_manager.next_unit_select() #RatioManager에서 받아옴
            if self.can_afford(next_unit) and self.time - self.ratio_manager.evoked.get((self.cc.tag, 'train'), 0) > 1.0:
                # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
                actions.append(self.cc.train(next_unit))
                self.ratio_manager.evoked[(self.cc.tag, 'train')] = self.time
                self.assign_manager.assign()
            #if cc.health_percentage < 0.9:    #mule 테스트용 코드
            #if self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC]).amount > 5:
                #self.tactics = Tactics(3)
                #actions += await self.mule_manager.step()
            #else:
                #self.tactics = Tactics(4)


        
        #actions += await self.attack_team_manager.step()
        actions += await self.recon_manager.step() 
        actions += await self.combat_manager.step()   
        actions += await self.nuke_manager.step()
        

        ## -----명령 실행-----
        await self.do_actions(actions)
        #print("한거: ",actions)



    def set_strategy(self):
        #
        # 특징 추출
        #
        state = np.zeros(5, dtype=np.float32)
        state[0] = self.cc.health_percentage
        state[1] = min(1.0, self.minerals / 1000)
        state[2] = min(1.0, self.vespene / 1000)
        state[3] = min(1.0, self.time / 360)
        state[4] = min(1.0, self.state.score.total_damage_dealt_life / 2500)
        state = state.reshape(1, -1)

        # NN
        data = [
            CommandType.STATE,
            pickle.dumps(self.game_id),
            pickle.dumps(state.shape),
            state,
        ]
        if self.sock is not None:
            self.sock.send_multipart(data)
            data = self.sock.recv_multipart()
            value = pickle.loads(data[0])
            action = pickle.loads(data[1])
        else:
            with torch.no_grad():
                value, logp = self.model(torch.FloatTensor(state))
                value = value.item()
                action = logp.exp().multinomial(num_samples=1).item()

        tactic_strategy = TacticStrategy(action)
        return tactic_strategy


    def on_end(self, game_result):
        if self.sock is not None:
            score = 1. if game_result is Result.Victory else -1.
            self.sock.send_multipart((
                CommandType.SCORE, 
                pickle.dumps(self.game_id),
                pickle.dumps(score),
            ))
            self.sock.recv_multipart()



    '''def on_end(self, game_result):
        for tag in self.combatArray:
            print("111전투 : ", tag)
        print("-----")
        for tag in self.reconArray:
            print("2222222정찰 : ", tag)
        print("-----")
        for tag in self.nukeArray:
            print("핵 : ", tag)
        print("-----")'''


